#include "llama-ssd.h"
#include "llama-iouring.h"
#include "llama-mmap.h"
#include "llama-impl.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

llama_ssd_manager::llama_ssd_manager() = default;

llama_ssd_manager::~llama_ssd_manager() {
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

    // Open own file handles for reading expert data
    for (const auto & path : file_paths) {
        files_.emplace_back(new llama_file(path.c_str(), "rb"));
    }

    // Create io_uring instance for async I/O
    io = std::make_unique<llama_io_uring>(64);

    initialized = true;

    LLAMA_LOG_INFO("%s: SSD offloading initialized, %zu layers, buffer size %.1f MB x2, async=%s\n",
        __func__, layers.size(), (double)buf_size / (1024.0 * 1024.0),
        io->is_async() ? "io_uring" : "sync");
}

void llama_ssd_manager::load_expert_sync(int layer_idx, int tensor_idx, int expert_idx, int buf_idx) {
    const auto & ti = layers[layer_idx].tensors[tensor_idx];
    const auto & slice = ti.experts[expert_idx];

    uint8_t * dest = (uint8_t *)buffers[buf_idx] + ti.tensor_offset + expert_idx * slice.slice_size;

    llama_file * file = files_[slice.file_idx].get();
    file->seek(slice.file_offset, SEEK_SET);
    file->read_raw(dest, slice.slice_size);

    stats_.n_loads++;
    stats_.bytes_loaded += slice.slice_size;
}

uint64_t llama_ssd_manager::load_expert_async(int layer_idx, int tensor_idx, int expert_idx, int buf_idx) {
    const auto & ti = layers[layer_idx].tensors[tensor_idx];
    const auto & slice = ti.experts[expert_idx];

    uint8_t * dest = (uint8_t *)buffers[buf_idx] + ti.tensor_offset + expert_idx * slice.slice_size;

    int fd = files_[slice.file_idx]->file_id();
    uint64_t ticket = io->submit_read(fd, dest, slice.slice_size, (off_t)slice.file_offset);

    in_flight[ticket] = {layer_idx, tensor_idx, expert_idx, buf_idx};

    stats_.n_loads++;
    stats_.bytes_loaded += slice.slice_size;

    return ticket;
}

void llama_ssd_manager::ensure_ready(int il, const int32_t * selected_experts, int n_selected) {
    if (il < 0 || il >= (int)layers.size()) return;

    const auto & layer = layers[il];
    auto & bs = buf_states[active_buf];

    bool same_layer = (bs.layer_idx == il);

    if (!same_layer) {
        bs.layer_idx = il;
        bs.expert_loaded.resize(layer.tensors.size());
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            bs.expert_loaded[t].assign(layer.tensors[t].n_expert, false);
        }
    }

    // First: wait for any in-flight async reads that target this layer+buffer
    // and mark them as loaded
    if (!in_flight.empty()) {
        std::vector<uint64_t> completed;
        io->reap_completed(completed);
        for (uint64_t ticket : completed) {
            auto it = in_flight.find(ticket);
            if (it != in_flight.end()) {
                auto & info = it->second;
                if (info.buf_idx == active_buf && info.layer_idx == il) {
                    bs.expert_loaded[info.tensor_idx][info.expert_idx] = true;
                }
                in_flight.erase(it);
            }
        }

        // Wait for remaining in-flight reads targeting this layer+buffer
        std::vector<uint64_t> to_wait;
        for (auto & [ticket, info] : in_flight) {
            if (info.buf_idx == active_buf && info.layer_idx == il) {
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
    }

    // Load any missing expert slices (synchronous — these are mispredictions)
    for (int s = 0; s < n_selected; s++) {
        int expert_idx = selected_experts[s];
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (expert_idx < 0 || expert_idx >= layer.tensors[t].n_expert) continue;
            if (bs.expert_loaded[t][expert_idx]) {
                stats_.n_prefetch_hits++;
                continue;
            }
            stats_.n_prefetch_misses++;
            load_expert_sync(il, (int)t, expert_idx, active_buf);
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

    // Submit async reads for predicted experts (non-blocking)
    for (int expert_idx : predicted) {
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (expert_idx < 0 || expert_idx >= layer.tensors[t].n_expert) continue;
            load_expert_async(il, (int)t, expert_idx, next_buf);
            // Don't mark as loaded yet — will be confirmed in ensure_ready
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
