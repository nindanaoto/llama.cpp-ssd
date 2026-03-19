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

llama_ssd_manager::llama_ssd_manager(int n_buf_slots)
    : n_buf_slots_(n_buf_slots), buffers(n_buf_slots, nullptr), buf_states(n_buf_slots),
      io_ready(n_buf_slots, -1) {
    GGML_ASSERT(n_buf_slots_ >= 2);
}

llama_ssd_manager::~llama_ssd_manager() {
    {
        std::lock_guard<std::mutex> lock(io_mutex);
        io_thread_stop = true;
    }
    io_cv.notify_all();
    if (io_thread.joinable()) {
        io_thread.join();
    }

    for (int fd : dio_fds) { if (fd >= 0) close(fd); }
    for (int fd : reg_fds) { if (fd >= 0) close(fd); }
    if (dummy_buf) { ggml_backend_buffer_free(dummy_buf); }
    for (int i = 0; i < n_buf_slots_; i++) { free(buffers[i]); }
}

bool llama_ssd_manager::is_expert_weight(const char * name) {
    const char * p = strstr(name, "ffn_");
    if (!p) return false;
    const char * exps = strstr(p, "_exps");
    if (!exps) return false;
    if (*(exps + 5) == '_') return false;
    if (strstr(p, "shexp")) return false;
    if (strstr(p, "gate_inp")) return false;
    return true;
}

static int parse_layer_idx(const char * name) {
    return (strncmp(name, "blk.", 4) == 0) ? atoi(name + 4) : -1;
}

void llama_ssd_manager::pre_alloc_scan(struct ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!is_expert_weight(ggml_get_name(t))) continue;
        int il = parse_layer_idx(ggml_get_name(t));
        if (il < 0) continue;
        while ((int)layers.size() <= il) {
            ssd_layer_info li; li.layer_idx = (int)layers.size(); layers.push_back(li);
        }
        int ti = (int)layers[il].tensors.size();
        ssd_tensor_info info; info.tensor = t; info.n_expert = (int)t->ne[2];
        layers[il].tensors.push_back(std::move(info));
        tensor_map[ggml_get_name(t)] = {il, ti};
    }
}

void llama_ssd_manager::finalize_layout(ggml_backend_buffer_type_t buft) {
    size_t max_layer_bytes = 0;
    for (auto & layer : layers) {
        size_t offset = 0;
        for (auto & ti : layer.tensors) {
            ti.tensor_offset = offset;
            offset += ggml_nbytes(ti.tensor);
        }
        layer.total_bytes = offset;
        max_layer_bytes = std::max(max_layer_bytes, offset);
    }
    if (max_layer_bytes == 0) return;

    buf_size = max_layer_bytes;
    for (int i = 0; i < n_buf_slots_; i++) {
        if (posix_memalign(&buffers[i], 4096, buf_size) != 0 || !buffers[i]) {
            throw std::runtime_error("Failed to allocate SSD buffer");
        }
    }
    for (int b = 0; b < n_buf_slots_; b++) { buf_states[b].layer_idx = -1; }

    last_selected.resize(layers.size());

    dummy_buf = ggml_backend_buft_alloc_buffer(buft, 0);
    if (dummy_buf) ggml_backend_buffer_set_usage(dummy_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (auto & layer : layers) {
        for (auto & ti : layer.tensors) {
            ti.tensor->data = (uint8_t *)buffers[0] + ti.tensor_offset;
            if (dummy_buf) ti.tensor->buffer = dummy_buf;
        }
    }

    size_t n_tensors = 0;
    for (auto & l : layers) n_tensors += l.tensors.size();
    LLAMA_LOG_INFO("%s: SSD offloading: %zu layers, %zu tensors, buffer %.1f MB x%d\n",
        __func__, layers.size(), n_tensors, (double)buf_size / (1024.0*1024.0), n_buf_slots_);
}

void llama_ssd_manager::register_tensor(ggml_tensor * tensor, uint16_t file_idx, size_t file_offset, int layer_idx) {
    GGML_ASSERT(!initialized);
    auto it = tensor_map.find(ggml_get_name(tensor));
    if (it == tensor_map.end()) return;
    auto & ti = layers[it->second.layer_idx].tensors[it->second.tensor_idx];
    GGML_ASSERT(ti.tensor == tensor);
    size_t stride = tensor->nb[2];
    ti.experts.clear();
    for (int e = 0; e < ti.n_expert; e++) {
        ti.experts.push_back({file_idx, file_offset + e * stride, stride});
    }
}

void llama_ssd_manager::init(const std::vector<std::string> & file_paths) {
    GGML_ASSERT(!initialized);
    use_direct_io = true;
    for (const auto & path : file_paths) {
        int dio_fd = -1, reg_fd = -1;
#ifdef __linux__
        dio_fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (dio_fd < 0) use_direct_io = false;
#else
        use_direct_io = false;
#endif
        reg_fd = open(path.c_str(), O_RDONLY);
        if (reg_fd < 0) throw std::runtime_error("Failed to open: " + path);
        dio_fds.push_back(dio_fd);
        reg_fds.push_back(reg_fd);
    }
    if (use_direct_io) {
        for (const auto & layer : layers)
            for (const auto & ti : layer.tensors)
                for (const auto & s : ti.experts)
                    if (s.slice_size % 4096 != 0) { use_direct_io = false; break; }
        if (!use_direct_io) {
            for (int fd : dio_fds) if (fd >= 0) close(fd);
            dio_fds.assign(dio_fds.size(), -1);
        }
    }
    // Create io_uring for batch expert reads (queue depth 256 for RAID0)
    io = std::make_unique<llama_io_uring>(256);

    initialized = true;

    io_thread = std::thread(&llama_ssd_manager::io_thread_func, this);

    LLAMA_LOG_INFO("%s: SSD offloading initialized, %zu layers, buffer %.1f MB x%d, async=%s, direct_io=%s\n",
        __func__, layers.size(), (double)buf_size/(1024.0*1024.0), n_buf_slots_,
        io->is_async() ? "io_uring" : "sync",
        use_direct_io ? "on" : "off");
}

int llama_ssd_manager::get_read_fd(uint16_t file_idx) const {
    if (use_direct_io && file_idx < dio_fds.size() && dio_fds[file_idx] >= 0)
        return dio_fds[file_idx];
    return reg_fds[file_idx];
}

void llama_ssd_manager::load_expert_sync(int layer_idx, int tensor_idx, int expert_idx, int buf_idx) {
    const auto & ti = layers[layer_idx].tensors[tensor_idx];
    const auto & slice = ti.experts[expert_idx];
    uint8_t * dest = (uint8_t *)buffers[buf_idx] + ti.tensor_offset + expert_idx * slice.slice_size;

    if (use_direct_io && dio_fds[slice.file_idx] >= 0) {
        size_t misalign = slice.file_offset % dio_alignment;
        off_t  aligned_off = (off_t)(slice.file_offset - misalign);
        if (misalign == 0) {
            pread(dio_fds[slice.file_idx], dest, slice.slice_size, aligned_off);
        } else {
            size_t aligned_sz = ((slice.slice_size + misalign + dio_alignment - 1) / dio_alignment) * dio_alignment;
            thread_local void * bounce = nullptr;
            thread_local size_t bounce_cap = 0;
            if (bounce_cap < aligned_sz) { free(bounce); posix_memalign(&bounce, dio_alignment, aligned_sz); bounce_cap = aligned_sz; }
            ssize_t ret = pread(dio_fds[slice.file_idx], bounce, aligned_sz, aligned_off);
            if (ret >= (ssize_t)(slice.slice_size + misalign))
                memcpy(dest, (uint8_t *)bounce + misalign, slice.slice_size);
        }
    } else {
        pread(reg_fds[slice.file_idx], dest, slice.slice_size, (off_t)slice.file_offset);
    }
}

void llama_ssd_manager::load_layer_predicted(int il, int buf_idx) {
    auto predicted = predict_experts(il);
    if (predicted.empty()) return;

    const auto & layer = layers[il];
    auto & bs = buf_states[buf_idx];
    bs.layer_idx = il;
    bs.expert_loaded.resize(layer.tensors.size());
    for (size_t t = 0; t < layer.tensors.size(); t++)
        bs.expert_loaded[t].assign(layer.tensors[t].n_expert, false);

    // Batch-submit all expert slice reads via io_uring.
    // One flush() syscall submits all reads; the kernel dispatches them
    // across RAID0 devices in parallel. Much more efficient than N pread syscalls.
    struct work_item { int tensor_idx; int expert_idx; };
    std::vector<work_item> work;

    for (int expert_idx : predicted) {
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (expert_idx < 0 || expert_idx >= layer.tensors[t].n_expert) continue;
            const auto & ti = layer.tensors[t];
            const auto & slice = ti.experts[expert_idx];
            uint8_t * dest = (uint8_t *)buffers[buf_idx] + ti.tensor_offset + expert_idx * slice.slice_size;

            // Use O_DIRECT fd for aligned offsets, regular fd otherwise
            int fd;
            if (use_direct_io && dio_fds[slice.file_idx] >= 0 && slice.file_offset % dio_alignment == 0) {
                fd = dio_fds[slice.file_idx];
            } else {
                fd = reg_fds[slice.file_idx];
            }

            io->submit_read(fd, dest, slice.slice_size, (off_t)slice.file_offset);
            work.push_back({(int)t, expert_idx});
        }
    }

    if (!work.empty()) {
        io->flush();    // single syscall — all reads now in kernel queue
        io->wait_all(); // block until all complete
    }

    // Update buf_states and stats (single-threaded, no races)
    for (const auto & w : work) {
        bs.expert_loaded[w.tensor_idx][w.expert_idx] = true;
        stats_.n_loads++;
        stats_.bytes_loaded += layer.tensors[w.tensor_idx].experts[w.expert_idx].slice_size;
    }
}

// Single background I/O thread: loads layers sequentially into ring buffer.
// Only one thread ever writes to buffers/buf_states — no concurrent access races.
void llama_ssd_manager::io_thread_func() {
    while (true) {
        int gen;

        // Wait for begin_token() to signal work
        {
            std::unique_lock<std::mutex> lock(io_mutex);
            io_cv.wait(lock, [this] {
                return io_thread_stop || io_cursor < io_target;
            });
            if (io_thread_stop) return;
            gen = io_generation;
        }

        // Load layers one by one until done or token changes
        while (true) {
            int layer_to_load, buf_idx;

            {
                std::unique_lock<std::mutex> lock(io_mutex);

                // Abort if token changed
                if (gen != io_generation || io_thread_stop) break;

                // Done with all layers?
                if (io_cursor >= io_target) break;

                // Wait for slot to become available
                int slot = io_cursor % n_buf_slots_;
                while (io_ready[slot] >= 0 && io_ready[slot] >= compute_cursor) {
                    if (gen != io_generation || io_thread_stop) break;
                    io_cv.wait(lock);
                }
                if (gen != io_generation || io_thread_stop) break;

                layer_to_load = io_cursor;
                buf_idx = layer_to_load % n_buf_slots_;
                io_ready[buf_idx] = -1;
                io_cursor++;
            }

            // Load outside lock (this is the slow part)
            load_layer_predicted(layer_to_load, buf_idx);

            // Mark ready (only if still current token)
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                if (gen == io_generation) {
                    io_ready[buf_idx] = layer_to_load;
                }
            }
            compute_cv.notify_all();
        }
    }
}

void llama_ssd_manager::begin_token() {
    {
        std::lock_guard<std::mutex> lock(io_mutex);
        io_generation++;
        io_cursor = 0;
        compute_cursor = 0;
        io_target = (int)layers.size();
        for (int i = 0; i < n_buf_slots_; i++) io_ready[i] = -1;
    }
    io_cv.notify_all();
}

void llama_ssd_manager::ensure_ready(int il, const int32_t * selected_experts, int n_selected) {
    if (il < 0 || il >= (int)layers.size()) return;

    int use_buf = il % n_buf_slots_;
    int64_t t0 = ggml_time_us();

    {
        std::unique_lock<std::mutex> lock(io_mutex);
        compute_cursor = il;  // layers before il are done, their slots are free
        io_cv.notify_all();   // wake I/O thread if blocked on slot

        compute_cv.wait(lock, [this, il, use_buf] {
            return io_ready[use_buf] == il;
        });
    }
    stats_.t_wait_us += (double)(ggml_time_us() - t0);

    // Handle mispredictions
    const auto & layer = layers[il];
    auto & bs = buf_states[use_buf];
    for (int s = 0; s < n_selected; s++) {
        int eidx = selected_experts[s];
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (eidx < 0 || eidx >= layer.tensors[t].n_expert) continue;
            if (bs.expert_loaded[t][eidx]) { stats_.n_prefetch_hits++; continue; }
            stats_.n_prefetch_misses++;
            int64_t t1 = ggml_time_us();
            load_expert_sync(il, (int)t, eidx, use_buf);
            stats_.t_io_us += (double)(ggml_time_us() - t1);
            stats_.n_loads++;
            stats_.bytes_loaded += layer.tensors[t].experts[eidx].slice_size;
            bs.expert_loaded[t][eidx] = true;
        }
    }
}

void llama_ssd_manager::activate_layer(int il) {
    if (il < 0 || il >= (int)layers.size()) return;
    int buf_idx = il % n_buf_slots_;
    for (const auto & ti : layers[il].tensors)
        ti.tensor->data = (uint8_t *)buffers[buf_idx] + ti.tensor_offset;
}

std::vector<int> llama_ssd_manager::predict_experts(int il) const {
    if (il < 0 || il >= (int)last_selected.size()) return {};
    return last_selected[il].empty() ? std::vector<int>{0, 1} : last_selected[il];
}

void llama_ssd_manager::update_prediction(int il, const int32_t * sel, int n) {
    if (il >= 0 && il < (int)last_selected.size())
        last_selected[il].assign(sel, sel + n);
}

void llama_ssd_manager::preload_all_layers() {}

void llama_ssd_manager::scan_graph_for_topk(struct ggml_cgraph * gf) {
    for (auto & l : layers) l.topk_tensor = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        auto * n = ggml_graph_node(gf, i);
        if (strncmp(ggml_get_name(n), "ffn_moe_topk-", 13) == 0) {
            int il = atoi(ggml_get_name(n) + 13);
            if (il >= 0 && il < (int)layers.size()) layers[il].topk_tensor = n;
        }
    }
}

bool llama_ssd_manager::update_predictions_from_graph() {
    bool mismatch = false;
    for (int il = 0; il < (int)layers.size(); il++) {
        auto * tk = layers[il].topk_tensor;
        if (!tk || !tk->data) continue;
        const int32_t * actual = (const int32_t *)tk->data;
        int n = (int)tk->ne[0];
        auto & pred = last_selected[il];
        if ((int)pred.size() != n) { mismatch = true; }
        else for (int i = 0; i < n && !mismatch; i++) {
            bool found = false;
            for (int j = 0; j < n; j++) if (actual[i] == pred[j]) { found = true; break; }
            if (!found) mismatch = true;
        }
        update_prediction(il, actual, n);
    }
    return mismatch;
}
