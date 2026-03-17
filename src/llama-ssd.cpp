#include "llama-ssd.h"
#include "llama-iouring.h"
#include "llama-impl.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#endif

llama_ssd_manager::llama_ssd_manager() = default;

llama_ssd_manager::~llama_ssd_manager() {
    for (int fd : dio_fds) {
        if (fd >= 0) close(fd);
    }
    for (int fd : reg_fds) {
        if (fd >= 0) close(fd);
    }
    if (dummy_buf) {
        ggml_backend_buffer_free(dummy_buf);
        dummy_buf = nullptr;
    }
    for (int i = 0; i < 2; i++) {
        if (buffers[i]) {
            free(buffers[i]);
            buffers[i] = nullptr;
        }
    }
}

bool llama_ssd_manager::is_expert_weight(const char * name) {
    const char * p = strstr(name, "ffn_");
    if (!p) return false;

    const char * exps = strstr(p, "_exps");
    if (!exps) return false;

    // Not scales (_exps_s) or biases (_exps_b)
    const char * after = exps + 5;
    if (*after == '_') return false;

    // Not shared experts
    if (strstr(p, "shexp")) return false;

    // Not the router
    if (strstr(p, "gate_inp")) return false;

    return true;
}

static int parse_layer_idx(const char * name) {
    int il = -1;
    if (strncmp(name, "blk.", 4) == 0) {
        il = atoi(name + 4);
    }
    return il;
}

void llama_ssd_manager::pre_alloc_scan(struct ggml_context * ctx) {
    // Scan all tensors in this context, identify SSD-offloaded expert tensors,
    // and record their metadata (but NOT file offsets yet — those come from register_tensor)
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (!is_expert_weight(ggml_get_name(t))) continue;

        int layer_idx = parse_layer_idx(ggml_get_name(t));
        if (layer_idx < 0) continue;

        // Ensure we have enough layer slots
        while ((int)layers.size() <= layer_idx) {
            ssd_layer_info li;
            li.layer_idx = (int)layers.size();
            layers.push_back(li);
        }

        auto & layer = layers[layer_idx];
        int tensor_idx = (int)layer.tensors.size();

        ssd_tensor_info ti;
        ti.tensor = t;
        ti.n_expert = (int)t->ne[2];
        // experts vector will be populated by register_tensor later

        layer.tensors.push_back(std::move(ti));
        tensor_map[ggml_get_name(t)] = {layer_idx, tensor_idx};
    }
}

void llama_ssd_manager::finalize_layout(ggml_backend_buffer_type_t buft) {
    // Compute tensor offsets within the layer buffer and total layer sizes
    size_t max_layer_bytes = 0;
    for (auto & layer : layers) {
        size_t offset = 0;
        for (auto & ti : layer.tensors) {
            ti.tensor_offset = offset;
            size_t tensor_bytes = ggml_nbytes(ti.tensor);
            offset += tensor_bytes;
        }
        layer.total_bytes = offset;
        max_layer_bytes = std::max(max_layer_bytes, offset);
    }

    if (max_layer_bytes == 0) {
        LLAMA_LOG_WARN("%s: no SSD tensors found, SSD offloading disabled\n", __func__);
        return;
    }

    buf_size = max_layer_bytes;

    // Allocate double buffers with page alignment for potential O_DIRECT
    size_t alignment = 4096;
    for (int i = 0; i < 2; i++) {
        int ret = posix_memalign(&buffers[i], alignment, buf_size);
        if (ret != 0 || !buffers[i]) {
            LLAMA_LOG_ERROR("%s: failed to allocate SSD buffer %d (%.1f MB)\n",
                __func__, i, (double)buf_size / (1024.0 * 1024.0));
            throw std::runtime_error("Failed to allocate SSD double buffer");
        }
        memset(buffers[i], 0, buf_size);
    }

    // Initialize buffer states
    for (int b = 0; b < 2; b++) {
        buf_states[b].layer_idx = -1;
        buf_states[b].expert_loaded.clear();
    }

    // Initialize prediction state
    last_selected.resize(layers.size());

    // Create a dummy buffer so backend scheduler recognizes these as weight tensors
    dummy_buf = ggml_backend_buft_alloc_buffer(buft, 0);
    if (dummy_buf) {
        ggml_backend_buffer_set_usage(dummy_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // Set tensor->data to point into buffer 0 so the ggml allocator skips them.
    // Also set tensor->buffer to the dummy buffer.
    // This is the key trick: tensors with non-NULL data are not allocated by
    // ggml_backend_alloc_ctx_tensors_from_buft, saving RAM.
    for (auto & layer : layers) {
        for (auto & ti : layer.tensors) {
            ti.tensor->data = (uint8_t *)buffers[0] + ti.tensor_offset;
            if (dummy_buf) {
                ti.tensor->buffer = dummy_buf;
            }
        }
    }

    size_t n_tensors = 0;
    for (auto & layer : layers) {
        n_tensors += layer.tensors.size();
    }

    LLAMA_LOG_INFO("%s: SSD offloading: %zu layers, %zu tensors, buffer size %.1f MB x2 (saved %.1f MB RAM)\n",
        __func__, layers.size(), n_tensors,
        (double)buf_size / (1024.0 * 1024.0),
        (double)(max_layer_bytes * layers.size() - buf_size * 2) / (1024.0 * 1024.0));
}

void llama_ssd_manager::register_tensor(ggml_tensor * tensor, uint16_t file_idx, size_t file_offset, int layer_idx) {
    GGML_ASSERT(!initialized && "Cannot register tensors after init()");

    // Find the pre-scanned tensor info
    auto it = tensor_map.find(ggml_get_name(tensor));
    if (it == tensor_map.end()) {
        LLAMA_LOG_WARN("%s: tensor '%s' not found in pre-scan, skipping\n", __func__, ggml_get_name(tensor));
        return;
    }

    auto & ti = layers[it->second.layer_idx].tensors[it->second.tensor_idx];
    GGML_ASSERT(ti.tensor == tensor);

    // Compute per-expert slice info
    size_t expert_stride = tensor->nb[2];
    ti.experts.clear();
    for (int e = 0; e < ti.n_expert; e++) {
        ssd_expert_slice slice;
        slice.file_idx    = file_idx;
        slice.file_offset = file_offset + e * expert_stride;
        slice.slice_size  = expert_stride;
        ti.experts.push_back(slice);
    }

    LLAMA_LOG_DEBUG("%s: registered SSD tensor '%s' layer=%d n_expert=%d slice_size=%.1fMB\n",
        __func__, ggml_get_name(tensor), layer_idx, ti.n_expert,
        (double)expert_stride / (1024.0 * 1024.0));
}

void llama_ssd_manager::init(const std::vector<std::string> & file_paths) {
    GGML_ASSERT(!initialized);

    // Open file descriptors for reading expert data
    // Try O_DIRECT first for each file, fall back to regular fd
    use_direct_io = true;
    for (const auto & path : file_paths) {
        int dio_fd = -1;
        int reg_fd = -1;

#ifdef __linux__
        dio_fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (dio_fd >= 0) {
            // O_DIRECT requires alignment to logical block size (typically 512 or 4096),
            // NOT st_blksize which returns the optimal I/O size (can be very large on RAID)
            // Use 4096 as a safe default that works on all modern Linux filesystems
            dio_alignment = 4096;
        } else {
            use_direct_io = false;
        }
#else
        use_direct_io = false;
#endif
        // Always open a regular fd as fallback for unaligned reads
        reg_fd = open(path.c_str(), O_RDONLY);
        if (reg_fd < 0) {
            throw std::runtime_error("Failed to open model file: " + path);
        }

        dio_fds.push_back(dio_fd);
        reg_fds.push_back(reg_fd);
    }

    // O_DIRECT requires aligned offsets and sizes. Expert slice sizes are typically
    // aligned (quantized blocks), but file offsets depend on GGUF header layout.
    // We handle unaligned offsets by reading aligned chunks into a bounce buffer.
    if (use_direct_io) {
        // Check if sizes are aligned (offsets handled at read time)
        bool sizes_aligned = true;
        for (const auto & layer : layers) {
            for (const auto & ti : layer.tensors) {
                for (const auto & slice : ti.experts) {
                    if (slice.slice_size % dio_alignment != 0) {
                        sizes_aligned = false;
                        break;
                    }
                }
                if (!sizes_aligned) break;
            }
            if (!sizes_aligned) break;
        }
        if (!sizes_aligned) {
            LLAMA_LOG_WARN("%s: expert slice sizes not aligned to %zu bytes, disabling O_DIRECT\n",
                __func__, dio_alignment);
            use_direct_io = false;
            for (int fd : dio_fds) {
                if (fd >= 0) close(fd);
            }
            dio_fds.assign(dio_fds.size(), -1);
        } else {
            LLAMA_LOG_INFO("%s: O_DIRECT enabled, alignment=%zu\n", __func__, dio_alignment);
        }
    }

    // Create io_uring instance for async I/O
    io = std::make_unique<llama_io_uring>(64);

    initialized = true;

    LLAMA_LOG_INFO("%s: SSD offloading initialized, %zu layers, buffer size %.1f MB x2, async=%s, direct_io=%s\n",
        __func__, layers.size(), (double)buf_size / (1024.0 * 1024.0),
        io->is_async() ? "io_uring" : "sync",
        use_direct_io ? "on" : "off");
}

int llama_ssd_manager::get_read_fd(uint16_t file_idx) const {
    if (use_direct_io && file_idx < dio_fds.size() && dio_fds[file_idx] >= 0) {
        return dio_fds[file_idx];
    }
    return reg_fds[file_idx];
}

void llama_ssd_manager::load_expert_sync(int layer_idx, int tensor_idx, int expert_idx, int buf_idx) {
    const auto & ti = layers[layer_idx].tensors[tensor_idx];
    const auto & slice = ti.experts[expert_idx];

    uint8_t * dest = (uint8_t *)buffers[buf_idx] + ti.tensor_offset + expert_idx * slice.slice_size;

    if (use_direct_io && dio_fds[slice.file_idx] >= 0) {
        // O_DIRECT path: handle potentially unaligned offset
        size_t offset_misalign = slice.file_offset % dio_alignment;
        off_t  aligned_offset  = (off_t)(slice.file_offset - offset_misalign);
        size_t aligned_size    = ((slice.slice_size + offset_misalign + dio_alignment - 1) / dio_alignment) * dio_alignment;

        if (offset_misalign == 0) {
            // Offset and size are both aligned — read directly into dest
            ssize_t ret = pread(dio_fds[slice.file_idx], dest, slice.slice_size, aligned_offset);
            if (ret < 0 || (size_t)ret != slice.slice_size) {
                LLAMA_LOG_ERROR("%s: O_DIRECT pread failed: ret=%zd, errno=%s\n", __func__, ret, strerror(errno));
            }
        } else {
            // Offset is unaligned — read into aligned bounce buffer, then copy
            thread_local void * bounce_ptr = nullptr;
            thread_local size_t bounce_cap = 0;
            if (bounce_cap < aligned_size) {
                if (bounce_ptr) free(bounce_ptr);
                posix_memalign(&bounce_ptr, dio_alignment, aligned_size);
                bounce_cap = aligned_size;
            }

            ssize_t ret = pread(dio_fds[slice.file_idx], bounce_ptr, aligned_size, aligned_offset);
            if (ret < 0 || (size_t)ret < slice.slice_size + offset_misalign) {
                LLAMA_LOG_ERROR("%s: O_DIRECT bounce pread failed: ret=%zd, errno=%s\n", __func__, ret, strerror(errno));
            } else {
                memcpy(dest, (uint8_t *)bounce_ptr + offset_misalign, slice.slice_size);
            }
        }
    } else {
        // Regular pread path
        ssize_t ret = pread(reg_fds[slice.file_idx], dest, slice.slice_size, (off_t)slice.file_offset);
        if (ret < 0 || (size_t)ret != slice.slice_size) {
            LLAMA_LOG_ERROR("%s: pread failed: ret=%zd, errno=%s\n", __func__, ret, strerror(errno));
        }
    }

    stats_.n_loads++;
    stats_.bytes_loaded += slice.slice_size;
}

uint64_t llama_ssd_manager::load_expert_async(int layer_idx, int tensor_idx, int expert_idx, int buf_idx) {
    const auto & ti = layers[layer_idx].tensors[tensor_idx];
    const auto & slice = ti.experts[expert_idx];

    uint8_t * dest = (uint8_t *)buffers[buf_idx] + ti.tensor_offset + expert_idx * slice.slice_size;

    // For async reads with O_DIRECT and unaligned offsets, use the regular fd
    // (io_uring with O_DIRECT requires aligned offsets; bounce buffer approach
    // doesn't work well with async since we'd need to track the bounce buffer)
    int fd;
    if (use_direct_io && dio_fds[slice.file_idx] >= 0 && slice.file_offset % dio_alignment == 0) {
        fd = dio_fds[slice.file_idx];
    } else {
        fd = reg_fds[slice.file_idx];
    }

    uint64_t ticket = io->submit_read(fd, dest, slice.slice_size, (off_t)slice.file_offset);

    in_flight[ticket] = {layer_idx, tensor_idx, expert_idx, buf_idx};

    stats_.n_loads++;
    stats_.bytes_loaded += slice.slice_size;

    return ticket;
}

void llama_ssd_manager::ensure_ready(int il, const int32_t * selected_experts, int n_selected) {
    if (il < 0 || il >= (int)layers.size()) return;

    // Determine which buffer to use for this layer.
    // If prefetch targeted this layer, swap to the prefetched buffer.
    int use_buf;
    if (prefetch_target_layer == il && prefetch_target_buf >= 0) {
        use_buf = prefetch_target_buf;
        active_buf = use_buf;
        prefetch_target_layer = -1;
        prefetch_target_buf = -1;
    } else {
        use_buf = active_buf;
    }

    const auto & layer = layers[il];
    auto & bs = buf_states[use_buf];

    if (bs.layer_idx != il) {
        bs.layer_idx = il;
        bs.expert_loaded.resize(layer.tensors.size());
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            bs.expert_loaded[t].assign(layer.tensors[t].n_expert, false);
        }
    }

    // Reap any completed async reads and mark experts as loaded
    if (!in_flight.empty()) {
        std::vector<uint64_t> completed;
        io->reap_completed(completed);
        for (uint64_t ticket : completed) {
            auto it = in_flight.find(ticket);
            if (it != in_flight.end()) {
                auto & info = it->second;
                if (info.buf_idx == use_buf && info.layer_idx == il) {
                    bs.expert_loaded[info.tensor_idx][info.expert_idx] = true;
                }
                in_flight.erase(it);
            }
        }

        // Wait for remaining in-flight reads targeting this layer+buffer
        int64_t t0 = ggml_time_us();
        std::vector<uint64_t> to_wait;
        for (auto & [ticket, info] : in_flight) {
            if (info.buf_idx == use_buf && info.layer_idx == il) {
                to_wait.push_back(ticket);
            }
        }
        for (uint64_t ticket : to_wait) {
            io->wait_for(ticket);
            auto it = in_flight.find(ticket);
            if (it != in_flight.end()) {
                auto & info = it->second;
                bs.expert_loaded[info.tensor_idx][info.expert_idx] = true;
                in_flight.erase(it);
            }
        }
        if (!to_wait.empty()) {
            stats_.t_wait_us += (double)(ggml_time_us() - t0);
        }
    }

    // Load any missing expert slices synchronously (mispredictions)
    for (int s = 0; s < n_selected; s++) {
        int expert_idx = selected_experts[s];
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (expert_idx < 0 || expert_idx >= layer.tensors[t].n_expert) continue;
            if (bs.expert_loaded[t][expert_idx]) {
                stats_.n_prefetch_hits++;
                continue;
            }
            stats_.n_prefetch_misses++;
            int64_t t0 = ggml_time_us();
            load_expert_sync(il, (int)t, expert_idx, use_buf);
            stats_.t_io_us += (double)(ggml_time_us() - t0);
            bs.expert_loaded[t][expert_idx] = true;
        }
    }
}

void llama_ssd_manager::activate_layer(int il) {
    if (il < 0 || il >= (int)layers.size()) return;

    const auto & layer = layers[il];

    for (const auto & ti : layer.tensors) {
        ti.tensor->data = (uint8_t *)buffers[active_buf] + ti.tensor_offset;
    }
}

void llama_ssd_manager::prefetch_start(int il) {
    if (il < 0 || il >= (int)layers.size()) return;

    int next_buf = 1 - active_buf;
    auto predicted = predict_experts(il);
    if (predicted.empty()) return;

    const auto & layer = layers[il];
    auto & bs = buf_states[next_buf];

    bs.layer_idx = il;
    bs.expert_loaded.resize(layer.tensors.size());
    for (size_t t = 0; t < layer.tensors.size(); t++) {
        bs.expert_loaded[t].assign(layer.tensors[t].n_expert, false);
    }

    // Record prefetch target so ensure_ready knows which buffer to use
    prefetch_target_buf = next_buf;
    prefetch_target_layer = il;

    // Submit async reads for predicted experts (non-blocking)
    for (int expert_idx : predicted) {
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (expert_idx < 0 || expert_idx >= layer.tensors[t].n_expert) continue;
            load_expert_async(il, (int)t, expert_idx, next_buf);
        }
    }
}

std::vector<int> llama_ssd_manager::predict_experts(int il) const {
    if (il < 0 || il >= (int)last_selected.size()) return {};
    if (last_selected[il].empty()) {
        return {0, 1};
    }
    return last_selected[il];
}

void llama_ssd_manager::update_prediction(int il, const int32_t * selected_experts, int n_selected) {
    if (il < 0 || il >= (int)last_selected.size()) return;
    last_selected[il].assign(selected_experts, selected_experts + n_selected);
}
