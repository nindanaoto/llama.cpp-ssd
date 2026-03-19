#include "llama-ssd.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#endif

llama_ssd_manager::llama_ssd_manager(int n_buf_slots, int n_io_threads)
    : n_buf_slots_(n_buf_slots), n_io_threads_(std::max(n_io_threads, 1)),
      buffers(n_buf_slots, nullptr), buf_states(n_buf_slots),
      io_ready(n_buf_slots, -1) {
    GGML_ASSERT(n_buf_slots_ >= 2 && "need at least 2 buffer slots for double-buffering");
}

llama_ssd_manager::~llama_ssd_manager() {
    // Stop I/O thread
    {
        std::lock_guard<std::mutex> lock(io_mutex);
        io_thread_stop = true;
    }
    io_cv.notify_all();
    if (io_thread.joinable()) {
        io_thread.join();
    }

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
    for (int i = 0; i < n_buf_slots_; i++) {
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
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (!is_expert_weight(ggml_get_name(t))) continue;

        int layer_idx = parse_layer_idx(ggml_get_name(t));
        if (layer_idx < 0) continue;

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

        layer.tensors.push_back(std::move(ti));
        tensor_map[ggml_get_name(t)] = {layer_idx, tensor_idx};
    }
}

void llama_ssd_manager::finalize_layout(ggml_backend_buffer_type_t buft) {
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

    size_t alignment = 4096;
    for (int i = 0; i < n_buf_slots_; i++) {
        int ret = posix_memalign(&buffers[i], alignment, buf_size);
        if (ret != 0 || !buffers[i]) {
            LLAMA_LOG_ERROR("%s: failed to allocate SSD buffer slot %d (%.1f MB)\n",
                __func__, i, (double)buf_size / (1024.0 * 1024.0));
            throw std::runtime_error("Failed to allocate SSD circular buffer");
        }
        memset(buffers[i], 0, buf_size);
    }

    for (int b = 0; b < n_buf_slots_; b++) {
        buf_states[b].layer_idx = -1;
        buf_states[b].expert_loaded.clear();
    }

    last_selected.resize(layers.size());

    dummy_buf = ggml_backend_buft_alloc_buffer(buft, 0);
    if (dummy_buf) {
        ggml_backend_buffer_set_usage(dummy_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

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

    LLAMA_LOG_INFO("%s: SSD offloading: %zu layers, %zu tensors, buffer size %.1f MB x%d (saved %.1f MB RAM)\n",
        __func__, layers.size(), n_tensors,
        (double)buf_size / (1024.0 * 1024.0), n_buf_slots_,
        (double)(max_layer_bytes * layers.size() - buf_size * n_buf_slots_) / (1024.0 * 1024.0));
}

void llama_ssd_manager::register_tensor(ggml_tensor * tensor, uint16_t file_idx, size_t file_offset, int layer_idx) {
    GGML_ASSERT(!initialized && "Cannot register tensors after init()");

    auto it = tensor_map.find(ggml_get_name(tensor));
    if (it == tensor_map.end()) {
        LLAMA_LOG_WARN("%s: tensor '%s' not found in pre-scan, skipping\n", __func__, ggml_get_name(tensor));
        return;
    }

    auto & ti = layers[it->second.layer_idx].tensors[it->second.tensor_idx];
    GGML_ASSERT(ti.tensor == tensor);

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

    use_direct_io = true;
    for (const auto & path : file_paths) {
        int dio_fd = -1;
        int reg_fd = -1;

#ifdef __linux__
        dio_fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (dio_fd >= 0) {
            dio_alignment = 4096;
        } else {
            use_direct_io = false;
        }
#else
        use_direct_io = false;
#endif
        reg_fd = open(path.c_str(), O_RDONLY);
        if (reg_fd < 0) {
            throw std::runtime_error("Failed to open model file: " + path);
        }

        dio_fds.push_back(dio_fd);
        reg_fds.push_back(reg_fd);
    }

    if (use_direct_io) {
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

    initialized = true;

    // Start the background I/O thread
    io_thread_stop = false;
    io_thread = std::thread(&llama_ssd_manager::io_thread_func, this);

    LLAMA_LOG_INFO("%s: SSD offloading initialized, %zu layers, buffer size %.1f MB x%d, io_threads=%d, direct_io=%s\n",
        __func__, layers.size(), (double)buf_size / (1024.0 * 1024.0), n_buf_slots_,
        n_io_threads_, use_direct_io ? "on" : "off");
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
        size_t offset_misalign = slice.file_offset % dio_alignment;
        off_t  aligned_offset  = (off_t)(slice.file_offset - offset_misalign);
        size_t aligned_size    = ((slice.slice_size + offset_misalign + dio_alignment - 1) / dio_alignment) * dio_alignment;

        if (offset_misalign == 0) {
            ssize_t ret = pread(dio_fds[slice.file_idx], dest, slice.slice_size, aligned_offset);
            if (ret < 0 || (size_t)ret != slice.slice_size) {
                LLAMA_LOG_ERROR("%s: O_DIRECT pread failed: ret=%zd, errno=%s\n", __func__, ret, strerror(errno));
            }
        } else {
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
        ssize_t ret = pread(reg_fds[slice.file_idx], dest, slice.slice_size, (off_t)slice.file_offset);
        if (ret < 0 || (size_t)ret != slice.slice_size) {
            LLAMA_LOG_ERROR("%s: pread failed: ret=%zd, errno=%s\n", __func__, ret, strerror(errno));
        }
    }

    // Note: stats updated by caller, not here (thread safety for parallel I/O)
}

// Load all predicted experts for a layer into a buffer slot.
// Uses n_io_threads_ parallel pread threads to saturate RAID0 bandwidth.
void llama_ssd_manager::load_layer_predicted(int il, int buf_idx) {
    auto predicted = predict_experts(il);
    if (predicted.empty()) return;

    const auto & layer = layers[il];
    auto & bs = buf_states[buf_idx];

    bs.layer_idx = il;
    bs.expert_loaded.resize(layer.tensors.size());
    for (size_t t = 0; t < layer.tensors.size(); t++) {
        bs.expert_loaded[t].assign(layer.tensors[t].n_expert, false);
    }

    // Collect all (tensor_idx, expert_idx) work items
    struct work_item { int tensor_idx; int expert_idx; };
    std::vector<work_item> work;
    for (int expert_idx : predicted) {
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (expert_idx < 0 || expert_idx >= layer.tensors[t].n_expert) continue;
            work.push_back({(int)t, expert_idx});
        }
    }
    if (work.empty()) return;

    // Parallel load: N threads each pick work items via atomic index
    int n_workers = std::min(n_io_threads_, (int)work.size());
    std::atomic<size_t> cursor(0);
    auto worker_fn = [&]() {
        while (true) {
            size_t idx = cursor.fetch_add(1, std::memory_order_relaxed);
            if (idx >= work.size()) break;
            load_expert_sync(il, work[idx].tensor_idx, work[idx].expert_idx, buf_idx);
        }
    };

    if (n_workers <= 1) {
        worker_fn();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n_workers - 1);
        for (int i = 0; i < n_workers - 1; i++) {
            threads.emplace_back(worker_fn);
        }
        worker_fn(); // coordinator thread also does work
        for (auto & th : threads) {
            th.join();
        }
    }

    // Mark loaded + update stats (single-threaded, no races)
    for (const auto & w : work) {
        bs.expert_loaded[w.tensor_idx][w.expert_idx] = true;
        stats_.n_loads++;
        stats_.bytes_loaded += layer.tensors[w.tensor_idx].experts[w.expert_idx].slice_size;
    }
}

// Background I/O thread: continuously prefetches layers ahead of compute
void llama_ssd_manager::io_thread_func() {
    while (true) {
        int layer_to_load = -1;
        int buf_idx = -1;
        int gen = -1;

        {
            std::unique_lock<std::mutex> lock(io_mutex);

            // Wait until there's work to do
            io_cv.wait(lock, [this] {
                if (io_thread_stop) return true;
                if (io_cursor >= io_target) return false;

                int slot = io_cursor % n_buf_slots_;
                if (io_ready[slot] >= 0 && io_ready[slot] >= compute_cursor) {
                    return false; // slot still in use by compute
                }
                return true;
            });

            if (io_thread_stop) break;

            layer_to_load = io_cursor;
            buf_idx = layer_to_load % n_buf_slots_;
            gen = io_generation; // capture generation before releasing lock

            io_ready[buf_idx] = -1;
            io_cursor++;
        }

        // Load layer outside the lock — this is the slow I/O part
        load_layer_predicted(layer_to_load, buf_idx);

        // Mark slot as ready — but ONLY if this is still the current token.
        // If begin_token() was called while we were loading, this completion
        // is stale and must be discarded to prevent deadlock.
        {
            std::lock_guard<std::mutex> lock(io_mutex);
            if (gen == io_generation) {
                io_ready[buf_idx] = layer_to_load;
            }
            // else: stale load from previous token, discard silently
        }
        compute_cv.notify_all();
    }
}

void llama_ssd_manager::begin_token() {
    std::lock_guard<std::mutex> lock(io_mutex);
    io_generation++;    // invalidate any in-flight loads from previous token
    io_cursor = 0;
    compute_cursor = 0;
    io_target = (int)layers.size();
    for (int i = 0; i < n_buf_slots_; i++) {
        io_ready[i] = -1;
    }
    io_cv.notify_all(); // wake I/O thread
}

void llama_ssd_manager::ensure_ready(int il, const int32_t * selected_experts, int n_selected) {
    if (il < 0 || il >= (int)layers.size()) return;

    int use_buf = il % n_buf_slots_;
    int64_t t0 = ggml_time_us();

    // Wait for the I/O thread to finish loading this layer.
    // compute_cursor = il means "layers 0..il-1 have finished compute, their slots are free."
    // We do NOT set il+1 here because layer il's MoE compute hasn't happened yet —
    // the slot must remain valid until the NEXT layer's ensure_ready fires.
    {
        std::unique_lock<std::mutex> lock(io_mutex);
        compute_cursor = il; // layers before il are done, their slots can be reused
        io_cv.notify_all();  // wake I/O thread in case it's waiting for a slot

        compute_cv.wait(lock, [this, il, use_buf] {
            return io_ready[use_buf] == il;
        });
    }

    stats_.t_wait_us += (double)(ggml_time_us() - t0);

    // Check for mispredictions: load any experts the I/O thread didn't prefetch
    const auto & layer = layers[il];
    auto & bs = buf_states[use_buf];
    int n_misses = 0;

    for (int s = 0; s < n_selected; s++) {
        int expert_idx = selected_experts[s];
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (expert_idx < 0 || expert_idx >= layer.tensors[t].n_expert) continue;
            if (bs.expert_loaded[t][expert_idx]) {
                stats_.n_prefetch_hits++;
                continue;
            }
            stats_.n_prefetch_misses++;
            int64_t t1 = ggml_time_us();
            load_expert_sync(il, (int)t, expert_idx, use_buf);
            stats_.t_io_us += (double)(ggml_time_us() - t1);
            stats_.n_loads++;
            stats_.bytes_loaded += layer.tensors[t].experts[expert_idx].slice_size;
            bs.expert_loaded[t][expert_idx] = true;
            n_misses++;
        }
    }
}

void llama_ssd_manager::activate_layer(int il) {
    if (il < 0 || il >= (int)layers.size()) return;

    int buf_idx = il % n_buf_slots_;
    const auto & layer = layers[il];

    for (const auto & ti : layer.tensors) {
        ti.tensor->data = (uint8_t *)buffers[buf_idx] + ti.tensor_offset;
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

void llama_ssd_manager::preload_all_layers() {
    (void)this;
}

void llama_ssd_manager::scan_graph_for_topk(struct ggml_cgraph * gf) {
    for (auto & layer : layers) {
        layer.topk_tensor = nullptr;
    }

    int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        const char * name = ggml_get_name(node);
        if (strncmp(name, "ffn_moe_topk-", 13) == 0) {
            int il = atoi(name + 13);
            if (il >= 0 && il < (int)layers.size()) {
                layers[il].topk_tensor = node;
            }
        }
    }
}

bool llama_ssd_manager::update_predictions_from_graph() {
    bool any_mismatch = false;

    for (int il = 0; il < (int)layers.size(); il++) {
        const auto & layer = layers[il];
        if (!layer.topk_tensor || !layer.topk_tensor->data) continue;

        const int32_t * actual = (const int32_t *)layer.topk_tensor->data;
        int n_selected = (int)layer.topk_tensor->ne[0];

        const auto & predicted = last_selected[il];
        if ((int)predicted.size() != n_selected) {
            any_mismatch = true;
        } else {
            for (int i = 0; i < n_selected; i++) {
                bool found = false;
                for (int j = 0; j < (int)predicted.size(); j++) {
                    if (actual[i] == predicted[j]) { found = true; break; }
                }
                if (!found) { any_mismatch = true; break; }
            }
        }

        update_prediction(il, actual, n_selected);
    }

    return any_mismatch;
}
