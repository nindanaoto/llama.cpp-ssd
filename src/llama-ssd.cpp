#include "llama-ssd.h"
#include "llama-iouring.h"
#include "llama-impl.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <unistd.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#endif

static size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static bool is_aligned_io(const void * ptr, size_t size, size_t offset, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0 && (size % alignment) == 0 && (offset % alignment) == 0;
}

static std::vector<std::string> split_csv(const std::string & value) {
    std::vector<std::string> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

static std::string path_join(const std::string & dir, const std::string & name) {
    if (dir.empty() || dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

static std::string path_basename(const std::string & path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string default_stripe_name_for_model(const std::string & path) {
    std::string name = path_basename(path);
    const std::string suffix = ".gguf";
    if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        name.resize(name.size() - suffix.size());
    }
    return name;
}

struct ssd_file_stat {
    size_t size = 0;
    int64_t mtime = 0;
};

static ssd_file_stat stat_file_for_manifest(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("failed to stat file: " + path);
    }
    return { (size_t) st.st_size, (int64_t) st.st_mtime };
}

llama_ssd_manager::llama_ssd_manager(
        int n_buf_slots,
        std::string stripe_dirs,
        std::string stripe_name,
        size_t stripe_chunk_size,
        size_t read_chunk_size,
        bool allow_direct_io,
        int predict_history,
        int prefetch_window,
        int cpu_layers,
        size_t cache_size)
    : n_buf_slots_(n_buf_slots),
      predict_history_(std::max(predict_history, 0)),
      prefetch_window_(std::max(prefetch_window, 1)),
      cpu_layers_(std::min(std::max(cpu_layers, 0), 8)),
      cache_size_limit_(cache_size),
      buffers(n_buf_slots, nullptr), buf_states(n_buf_slots),
      allow_direct_io(allow_direct_io),
      stripe_dirs_arg(std::move(stripe_dirs)),
      stripe_name_arg(std::move(stripe_name)),
      stripe_chunk_size(stripe_chunk_size),
      read_chunk_size(read_chunk_size),
      io_ready(n_buf_slots, -1) {
    GGML_ASSERT(n_buf_slots_ >= 2);
    if (this->read_chunk_size != 0 && this->read_chunk_size % dio_alignment != 0) {
        throw std::runtime_error("SSD read chunk size must be 0 or a multiple of 4096");
    }
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

    if (initialized && stats_.n_loads > 0) {
        uint64_t unique_loads = 0;
        uint64_t repeated_loads = 0;
        uint64_t bytes_unique = 0;
        uint64_t bytes_reloaded = 0;
        uint32_t max_reloads = 0;
        int max_layer = -1;
        int max_tensor = -1;
        int max_expert = -1;

        for (size_t il = 0; il < load_counts.size(); ++il) {
            for (size_t it = 0; it < load_counts[il].size(); ++it) {
                const size_t slice_size = layers[il].tensors[it].experts.empty() ? 0 : layers[il].tensors[it].experts[0].slice_size;
                for (size_t ie = 0; ie < load_counts[il][it].size(); ++ie) {
                    const uint32_t n = load_counts[il][it][ie];
                    if (n == 0) continue;
                    unique_loads++;
                    bytes_unique += slice_size;
                    if (n > 1) {
                        repeated_loads += n - 1;
                        bytes_reloaded += (uint64_t)(n - 1) * slice_size;
                    }
                    if (n > max_reloads) {
                        max_reloads = n;
                        max_layer = (int)il;
                        max_tensor = (int)it;
                        max_expert = (int)ie;
                    }
                }
            }
        }

        const uint64_t total = stats_.n_prefetch_hits + stats_.n_prefetch_misses;
        const double hit_rate = total > 0 ? 100.0 * stats_.n_prefetch_hits / total : 0.0;
        const double prefetch_bw_mbps = stats_.t_preload_us > 0 ? stats_.bytes_prefetched / stats_.t_preload_us : 0.0;
        const double miss_bw_mbps = stats_.t_io_us > 0 ? stats_.bytes_missed / stats_.t_io_us : 0.0;
        fprintf(stderr,
            "llama_ssd_manager: %lu loads, %.1f MB read, hit rate %.1f%% (%lu/%lu), "
            "prefetch %.1f MB in %.1f ms (%.0f MB/s), miss %.1f MB in %.1f ms (%.0f MB/s), wait %.1f ms\n",
            (unsigned long) stats_.n_loads,
            (double) stats_.bytes_loaded / (1024.0 * 1024.0),
            hit_rate, (unsigned long) stats_.n_prefetch_hits, (unsigned long) total,
            (double) stats_.bytes_prefetched / (1024.0 * 1024.0),
            stats_.t_preload_us / 1000.0, prefetch_bw_mbps,
            (double) stats_.bytes_missed / (1024.0 * 1024.0),
            stats_.t_io_us / 1000.0, miss_bw_mbps,
            stats_.t_wait_us / 1000.0);
        fprintf(stderr,
            "llama_ssd_manager: reuse: %lu unique slices, %lu reloads, %.1f MB unique, %.1f MB reloaded, hottest layer/tensor/expert=%d/%d/%d loaded %u times\n",
            (unsigned long) unique_loads,
            (unsigned long) repeated_loads,
            (double) bytes_unique / (1024.0 * 1024.0),
            (double) bytes_reloaded / (1024.0 * 1024.0),
            max_layer, max_tensor, max_expert, max_reloads);
        fprintf(stderr,
            "llama_ssd_manager: exact io wall: first %.1f ms/%lu batches, down %.1f ms/%lu batches, read requests %lu, max batch reads %lu\n",
            stats_.t_miss_wall_us / 1000.0,
            (unsigned long) stats_.n_miss_batches,
            stats_.t_down_wall_us / 1000.0,
            (unsigned long) stats_.n_down_batches,
            (unsigned long) stats_.n_read_reqs,
            (unsigned long) stats_.max_batch_reads);
        if (cache_size_limit_ > 0) {
            fprintf(stderr,
                "llama_ssd_manager: cache: %.1f/%.1f MB used, %lu hits, %.1f MB served from RAM\n",
                (double) cache_size_used_ / (1024.0 * 1024.0),
                (double) cache_size_limit_ / (1024.0 * 1024.0),
                (unsigned long) stats_.n_cache_hits,
                (double) stats_.bytes_cache_hits / (1024.0 * 1024.0));
        }
    }

    for (int fd : dio_fds) { if (fd >= 0) close(fd); }
    for (int fd : reg_fds) { if (fd >= 0) close(fd); }
    for (int fd : stripe_dio_fds) { if (fd >= 0) close(fd); }
    for (int fd : stripe_reg_fds) { if (fd >= 0) close(fd); }
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

bool llama_ssd_manager::should_offload_weight(const char * name) const {
    if (!is_expert_weight(name)) return false;
    const int il = parse_layer_idx(name);
    return il >= 0 && il >= cpu_layers_;
}

void llama_ssd_manager::pre_alloc_scan(struct ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!should_offload_weight(ggml_get_name(t))) continue;
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
            offset = align_up_size(offset, dio_alignment);
            ti.tensor_offset = offset;
            offset += ggml_nbytes(ti.tensor);
        }
        layer.total_bytes = align_up_size(offset, dio_alignment);
        max_layer_bytes = std::max(max_layer_bytes, layer.total_bytes);
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
    selected_history.resize(layers.size());
    load_counts.resize(layers.size());
    pending_miss_io.resize(layers.size());
    pending_down_io.resize(layers.size());
    pending_miss_active.assign(layers.size(), false);
    pending_down_active.assign(layers.size(), false);
    cache_entries.resize(layers.size());
    for (size_t il = 0; il < layers.size(); ++il) {
        load_counts[il].resize(layers[il].tensors.size());
        cache_entries[il].resize(layers[il].tensors.size());
        for (size_t it = 0; it < layers[il].tensors.size(); ++it) {
            load_counts[il][it].assign(layers[il].tensors[it].n_expert, 0);
            cache_entries[il][it].resize(layers[il].tensors[it].n_expert);
        }
    }

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
    (void) layer_idx;
    auto it = tensor_map.find(ggml_get_name(tensor));
    if (it == tensor_map.end()) return;
    auto & ti = layers[it->second.layer_idx].tensors[it->second.tensor_idx];
    GGML_ASSERT(ti.tensor == tensor);
    size_t stride = tensor->nb[2];
    ti.experts.clear();
    for (int e = 0; e < ti.n_expert; e++) {
        ti.experts.push_back({file_idx, file_offset + e * stride, stride, {}});
    }
}

void llama_ssd_manager::init(const std::vector<std::string> & file_paths) {
    GGML_ASSERT(!initialized);
    use_direct_io = false;
    for (const auto & path : file_paths) {
        int dio_fd = -1, reg_fd = -1;
#ifdef __linux__
        if (allow_direct_io) {
            dio_fd = open(path.c_str(), O_RDONLY | O_DIRECT);
            if (dio_fd >= 0) use_direct_io = true;
        }
#else
        (void) path;
#endif
        reg_fd = open(path.c_str(), O_RDONLY);
        if (reg_fd < 0) throw std::runtime_error("Failed to open: " + path);
        dio_fds.push_back(dio_fd);
        reg_fds.push_back(reg_fd);
    }

    if (!stripe_dirs_arg.empty()) {
        load_stripe_manifest(file_paths);
    }

    // Create io_uring for batch expert reads. The prefetch ring scales with the
    // number of staging slots because each slot can now hold one in-flight layer.
    const int io_qd_base = read_chunk_size > 0 ? 4096 : 256;
    const int prefetch_qd = std::max(io_qd_base, n_buf_slots_ * 256);
    io = std::make_unique<llama_io_uring>(prefetch_qd);
    io_miss = std::make_unique<llama_io_uring>(io_qd_base);

    initialized = true;

    io_thread = std::thread(&llama_ssd_manager::io_thread_func, this);

    LLAMA_LOG_INFO("%s: SSD offloading initialized, %zu layers, buffer %.1f MB x%d, async=%s, direct_io=%s, striped=%s, predict_history=%d, prefetch_window=%d, cpu_layers=%d, read_chunk=%.1f KiB, cache=%.1f MB\n",
        __func__, layers.size(), (double)buf_size/(1024.0*1024.0), n_buf_slots_,
        io->is_async() ? "io_uring" : "sync",
        use_direct_io ? "on" : "off",
        use_striped_io ? "on" : "off",
        predict_history_, prefetch_window_, cpu_layers_,
        (double)read_chunk_size / 1024.0,
        (double)cache_size_limit_ / (1024.0 * 1024.0));
}

int llama_ssd_manager::get_read_fd(uint16_t file_idx) const {
    if (use_direct_io && file_idx < dio_fds.size() && dio_fds[file_idx] >= 0)
        return dio_fds[file_idx];
    return reg_fds[file_idx];
}

int llama_ssd_manager::get_stripe_fd(uint16_t file_idx) const {
    if (file_idx < stripe_dio_fds.size() && stripe_dio_fds[file_idx] >= 0) {
        return stripe_dio_fds[file_idx];
    }
    return stripe_reg_fds[file_idx];
}

bool llama_ssd_manager::try_cache_hit(int layer_idx, int tensor_idx, int expert_idx, uint8_t * dest, size_t size) {
    if (cache_size_limit_ == 0) return false;
    if (layer_idx < 0 || layer_idx >= (int)cache_entries.size()) return false;
    if (tensor_idx < 0 || tensor_idx >= (int)cache_entries[layer_idx].size()) return false;
    if (expert_idx < 0 || expert_idx >= (int)cache_entries[layer_idx][tensor_idx].size()) return false;

    std::shared_ptr<std::vector<uint8_t>> data;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto & entry = cache_entries[layer_idx][tensor_idx][expert_idx];
        if (!entry.valid || entry.size != size || !entry.data) return false;

        data = entry.data;
        cache_lru.splice(cache_lru.begin(), cache_lru, entry.lru_it);
        stats_.n_cache_hits++;
        stats_.bytes_cache_hits += size;
    }

    std::memcpy(dest, data->data(), size);
    return true;
}

void llama_ssd_manager::cache_insert(int layer_idx, int tensor_idx, int expert_idx, const uint8_t * src, size_t size) {
    if (cache_size_limit_ == 0 || size == 0 || size > cache_size_limit_) return;
    if (layer_idx < 0 || layer_idx >= (int)cache_entries.size()) return;
    if (tensor_idx < 0 || tensor_idx >= (int)cache_entries[layer_idx].size()) return;
    if (expert_idx < 0 || expert_idx >= (int)cache_entries[layer_idx][tensor_idx].size()) return;

    auto data = std::make_shared<std::vector<uint8_t>>(size);
    std::memcpy(data->data(), src, size);

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto & entry = cache_entries[layer_idx][tensor_idx][expert_idx];
    if (entry.valid) {
        cache_lru.splice(cache_lru.begin(), cache_lru, entry.lru_it);
        return;
    }

    while (cache_size_used_ + size > cache_size_limit_ && !cache_lru.empty()) {
        const cache_key victim = cache_lru.back();
        cache_lru.pop_back();
        auto & victim_entry = cache_entries[victim.layer_idx][victim.tensor_idx][victim.expert_idx];
        if (!victim_entry.valid) continue;
        cache_size_used_ -= victim_entry.size;
        victim_entry.data.reset();
        victim_entry.size = 0;
        victim_entry.valid = false;
    }
    if (cache_size_used_ + size > cache_size_limit_) return;

    entry.data = std::move(data);
    entry.size = size;
    entry.valid = true;
    cache_lru.push_front({layer_idx, tensor_idx, expert_idx});
    entry.lru_it = cache_lru.begin();
    cache_size_used_ += size;
}

void llama_ssd_manager::load_stripe_manifest(const std::vector<std::string> & file_paths) {
    stripe_dirs = split_csv(stripe_dirs_arg);
    if (stripe_dirs.empty()) {
        throw std::runtime_error("--ssd-stripe-dirs was provided but no directories were parsed");
    }
    if (stripe_chunk_size == 0 || stripe_chunk_size % dio_alignment != 0) {
        throw std::runtime_error("--ssd-stripe-chunk-size must be a non-zero multiple of 4096");
    }

    const std::string stripe_name = stripe_name_arg.empty() ? default_stripe_name_for_model(file_paths.front()) : stripe_name_arg;
    const std::string manifest_path = path_join(stripe_dirs.front(), stripe_name + ".manifest.json");

    std::ifstream fin(manifest_path);
    if (!fin) {
        throw std::runtime_error("failed to open SSD stripe manifest: " + manifest_path);
    }

    nlohmann::ordered_json manifest = nlohmann::ordered_json::parse(fin);
    if (manifest.value("version", 0) != 1) {
        throw std::runtime_error("unsupported SSD stripe manifest version: " + manifest_path);
    }
    if (manifest.value("chunk_size", (size_t)0) != stripe_chunk_size) {
        throw std::runtime_error("SSD stripe manifest chunk size does not match --ssd-stripe-chunk-size");
    }
    if (manifest.value("stripe_count", (size_t)0) != stripe_dirs.size()) {
        throw std::runtime_error("SSD stripe manifest stripe count does not match --ssd-stripe-dirs");
    }

    const auto & model_files = manifest.at("model_files");
    if (model_files.size() != file_paths.size()) {
        throw std::runtime_error("SSD stripe manifest model file count does not match loaded GGUF splits");
    }
    for (size_t i = 0; i < file_paths.size(); ++i) {
        const auto st = stat_file_for_manifest(file_paths[i]);
        const auto & mf = model_files.at(i);
        if (mf.value("basename", std::string()) != path_basename(file_paths[i]) ||
                mf.value("size", (size_t)0) != st.size ||
                mf.value("mtime", (int64_t)0) != st.mtime) {
            throw std::runtime_error("SSD stripe manifest is incompatible with loaded model file: " + file_paths[i]);
        }
    }

    const auto & stripe_files = manifest.at("stripe_files");
    if (stripe_files.size() != stripe_dirs.size()) {
        throw std::runtime_error("SSD stripe manifest stripe file count does not match --ssd-stripe-dirs");
    }
    for (size_t i = 0; i < stripe_dirs.size(); ++i) {
        const std::string path = path_join(stripe_dirs[i], stripe_name + ".stripe");
        const auto st = stat_file_for_manifest(path);
        if (stripe_files.at(i).value("size", (size_t)0) != st.size) {
            throw std::runtime_error("SSD stripe data file size does not match manifest: " + path);
        }

        int dio_fd = -1;
#ifdef __linux__
        if (allow_direct_io) {
            dio_fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        }
#endif
        int reg_fd = open(path.c_str(), O_RDONLY);
        if (reg_fd < 0) {
            throw std::runtime_error("failed to open SSD stripe data file: " + path);
        }
        stripe_paths.push_back(path);
        stripe_dio_fds.push_back(dio_fd);
        stripe_reg_fds.push_back(reg_fd);
    }

    const auto & tensors = manifest.at("tensors");
    for (auto it = tensors.begin(); it != tensors.end(); ++it) {
        const std::string tensor_name = it.key();
        auto loc_it = tensor_map.find(tensor_name);
        if (loc_it == tensor_map.end()) {
            continue;
        }

        auto & ti = layers[loc_it->second.layer_idx].tensors[loc_it->second.tensor_idx];
        const auto & jt = it.value();
        if (jt.value("n_expert", 0) != ti.n_expert) {
            throw std::runtime_error("SSD stripe manifest expert count mismatch for tensor: " + tensor_name);
        }
        if (jt.value("slice_size", (size_t)0) != ti.experts.front().slice_size) {
            throw std::runtime_error("SSD stripe manifest slice size mismatch for tensor: " + tensor_name);
        }

        const auto & experts = jt.at("experts");
        if (experts.size() != (size_t)ti.n_expert) {
            throw std::runtime_error("SSD stripe manifest expert table size mismatch for tensor: " + tensor_name);
        }
        for (int e = 0; e < ti.n_expert; ++e) {
            ti.experts[e].stripe_chunks.clear();
            for (const auto & jc : experts.at(e)) {
                ssd_read_chunk chunk {
                    (uint16_t) jc.at("stripe").get<int>(),
                    jc.at("file_offset").get<size_t>(),
                    jc.at("dst_offset").get<size_t>(),
                    jc.at("size").get<size_t>(),
                };
                if (chunk.file_idx >= stripe_dirs.size()) {
                    throw std::runtime_error("SSD stripe manifest references an invalid stripe index for tensor: " + tensor_name);
                }
                if (chunk.dst_offset + chunk.size > ti.experts[e].slice_size) {
                    throw std::runtime_error("SSD stripe manifest chunk exceeds expert slice for tensor: " + tensor_name);
                }
                ti.experts[e].stripe_chunks.push_back(chunk);
            }
        }
    }

    for (const auto & layer : layers) {
        for (const auto & ti : layer.tensors) {
            for (const auto & expert : ti.experts) {
                if (expert.stripe_chunks.empty()) {
                    throw std::runtime_error("SSD stripe manifest does not cover all registered expert tensors");
                }
            }
        }
    }

    use_striped_io = true;
    LLAMA_LOG_INFO("%s: SSD stripe sidecar loaded: %s (%zu stripes, chunk %.1f MiB)\n",
        __func__, manifest_path.c_str(), stripe_dirs.size(), (double)stripe_chunk_size / (1024.0 * 1024.0));
}

void llama_ssd_manager::load_expert_sync(int layer_idx, int tensor_idx, int expert_idx, int buf_idx) {
    const auto & ti = layers[layer_idx].tensors[tensor_idx];
    const auto & slice = ti.experts[expert_idx];
    uint8_t * dest = (uint8_t *)buffers[buf_idx] + ti.tensor_offset + expert_idx * slice.slice_size;

    if (!slice.stripe_chunks.empty()) {
        for (const auto & chunk : slice.stripe_chunks) {
            uint8_t * chunk_dest = dest + chunk.dst_offset;
            ssize_t ret = pread(stripe_reg_fds[chunk.file_idx], chunk_dest, chunk.size, (off_t)chunk.file_offset);
            if (ret != (ssize_t)chunk.size) {
                GGML_ABORT("%s: SSD stripe read failed: expected %zu bytes, got %zd\n", __func__, chunk.size, ret);
            }
        }
        return;
    }

    if (use_direct_io && dio_fds[slice.file_idx] >= 0 &&
            is_aligned_io(dest, slice.slice_size, slice.file_offset, dio_alignment)) {
        ssize_t ret = pread(dio_fds[slice.file_idx], dest, slice.slice_size, (off_t)slice.file_offset);
        if (ret != (ssize_t)slice.slice_size) {
            GGML_ABORT("%s: SSD direct read failed: expected %zu bytes, got %zd\n", __func__, slice.slice_size, ret);
        }
    } else {
        ssize_t ret = pread(reg_fds[slice.file_idx], dest, slice.slice_size, (off_t)slice.file_offset);
        if (ret != (ssize_t)slice.slice_size) {
            GGML_ABORT("%s: SSD read failed: expected %zu bytes, got %zd\n", __func__, slice.slice_size, ret);
        }
    }
}

bool llama_ssd_manager::is_down_tensor(const ssd_tensor_info & info) {
    const char * name = ggml_get_name(info.tensor);
    return strstr(name, "ffn_down") != nullptr && strstr(name, "_exps") != nullptr;
}

bool llama_ssd_manager::tensor_matches_stage(const ssd_tensor_info & info, tensor_stage stage) {
    if (stage == tensor_stage::all) {
        return true;
    }
    const bool is_down = is_down_tensor(info);
    return stage == tensor_stage::down ? is_down : !is_down;
}

llama_ssd_manager::pending_layer_io llama_ssd_manager::submit_experts_batch(
        int il,
        const std::vector<int> & experts,
        int buf_idx,
        llama_io_uring & engine,
        tensor_stage stage) {
    const auto & layer = layers[il];
    auto & bs = buf_states[buf_idx];
    pending_layer_io pending;
    pending.layer_idx = il;
    pending.buf_idx = buf_idx;
    pending.submitted_at_us = ggml_time_us();

    std::vector<int> unique_experts = experts;
    std::sort(unique_experts.begin(), unique_experts.end());
    unique_experts.erase(std::unique(unique_experts.begin(), unique_experts.end()), unique_experts.end());

    auto submit_one_read = [&](int reg_fd, int dio_fd, uint8_t * dest, size_t size, size_t offset) {
        int fd = reg_fd;
        if (dio_fd >= 0 && is_aligned_io(dest, size, offset, dio_alignment)) {
            fd = dio_fd;
        }
        pending.reads.push_back({engine.submit_read(fd, dest, size, (off_t)offset), size});
    };
    auto submit_read = [&](int reg_fd, int dio_fd, uint8_t * dest, size_t size, size_t offset) {
        if (read_chunk_size == 0 || size <= read_chunk_size) {
            submit_one_read(reg_fd, dio_fd, dest, size, offset);
            return;
        }

        size_t pos = 0;
        while (pos < size) {
            const size_t n = std::min(read_chunk_size, size - pos);
            submit_one_read(reg_fd, dio_fd, dest + pos, n, offset + pos);
            pos += n;
        }
    };

    for (int expert_idx : unique_experts) {
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (expert_idx < 0 || expert_idx >= layer.tensors[t].n_expert) continue;
            if (!tensor_matches_stage(layer.tensors[t], stage)) continue;
            if (!bs.expert_loaded.empty() && bs.expert_loaded[t][expert_idx]) continue;

            const auto & ti = layer.tensors[t];
            const auto & slice = ti.experts[expert_idx];
            uint8_t * dest = (uint8_t *)buffers[buf_idx] + ti.tensor_offset + expert_idx * slice.slice_size;

            if (try_cache_hit(il, (int)t, expert_idx, dest, slice.slice_size)) {
                bs.expert_loaded[t][expert_idx] = true;
                continue;
            }

            if (!slice.stripe_chunks.empty()) {
                for (const auto & chunk : slice.stripe_chunks) {
                    submit_read(
                        stripe_reg_fds[chunk.file_idx],
                        stripe_dio_fds[chunk.file_idx],
                        dest + chunk.dst_offset,
                        chunk.size,
                        chunk.file_offset);
                }
            } else {
                submit_read(
                    reg_fds[slice.file_idx],
                    slice.file_idx < dio_fds.size() ? dio_fds[slice.file_idx] : -1,
                    dest,
                    slice.slice_size,
                    slice.file_offset);
            }

            pending.work.push_back({(int)t, expert_idx});
        }
    }

    return pending;
}

void llama_ssd_manager::complete_experts_batch(const pending_layer_io & pending, llama_io_uring & engine, bool is_prefetch) {
    const auto & layer = layers[pending.layer_idx];
    auto & bs = buf_states[pending.buf_idx];

    for (const auto & r : pending.reads) {
        ssize_t ret = engine.wait_for(r.ticket);
        if (ret != (ssize_t)r.expected) {
            GGML_ABORT("%s: SSD read failed: expected %zu bytes, got %zd\n", __func__, r.expected, ret);
        }
    }

    for (const auto & w : pending.work) {
        bs.expert_loaded[w.tensor_idx][w.expert_idx] = true;
        if (pending.layer_idx >= 0 && pending.layer_idx < (int)load_counts.size() &&
                w.tensor_idx >= 0 && w.tensor_idx < (int)load_counts[pending.layer_idx].size() &&
                w.expert_idx >= 0 && w.expert_idx < (int)load_counts[pending.layer_idx][w.tensor_idx].size()) {
            load_counts[pending.layer_idx][w.tensor_idx][w.expert_idx]++;
        }
        stats_.n_loads++;
        const size_t loaded = layer.tensors[w.tensor_idx].experts[w.expert_idx].slice_size;
        stats_.bytes_loaded += loaded;
        const auto & ti = layer.tensors[w.tensor_idx];
        const uint8_t * cache_src = (const uint8_t *)buffers[pending.buf_idx] + ti.tensor_offset + (size_t)w.expert_idx * loaded;
        if (!is_prefetch) {
            cache_insert(pending.layer_idx, w.tensor_idx, w.expert_idx, cache_src, loaded);
        }
        if (is_prefetch) {
            stats_.n_prefetch_loads++;
            stats_.bytes_prefetched += loaded;
        } else {
            stats_.n_miss_loads++;
            stats_.bytes_missed += loaded;
        }
    }
}

void llama_ssd_manager::load_experts_batch(int il, const std::vector<int> & experts, int buf_idx, llama_io_uring & engine, tensor_stage stage) {
    auto pending = submit_experts_batch(il, experts, buf_idx, engine, stage);
    engine.flush();
    complete_experts_batch(pending, engine, false);
}

llama_ssd_manager::pending_layer_io llama_ssd_manager::submit_layer_predicted(int il, int buf_idx) {
    auto predicted = predict_experts(il);
    const auto & layer = layers[il];
    auto & bs = buf_states[buf_idx];
    bs.layer_idx = il;
    bs.expert_loaded.resize(layer.tensors.size());
    for (size_t t = 0; t < layer.tensors.size(); t++)
        bs.expert_loaded[t].assign(layer.tensors[t].n_expert, false);

    if (predicted.empty()) {
        pending_layer_io pending;
        pending.layer_idx = il;
        pending.buf_idx = buf_idx;
        return pending;
    }

    return submit_experts_batch(il, predicted, buf_idx, *io, tensor_stage::all);
}

// Single background I/O thread. It can keep a bounded number of layers in
// flight, but defaults to one layer to preserve the low-latency behavior.
// Only one thread ever writes to buffers/buf_states -- no concurrent access races.
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

        std::deque<pending_layer_io> pending_layers;
        std::vector<bool> slot_pending(n_buf_slots_, false);
        const size_t max_inflight = (size_t)std::min(prefetch_window_, n_buf_slots_);

        while (true) {
            bool should_exit = false;

            {
                std::unique_lock<std::mutex> lock(io_mutex);

                if (gen != io_generation || io_thread_stop) break;

                while (io_cursor < io_target && pending_layers.size() < max_inflight) {
                    int slot = io_cursor % n_buf_slots_;
                    if (slot_pending[slot]) {
                        break;
                    }
                    if (io_ready[slot] >= 0 && io_ready[slot] >= compute_cursor) {
                        break;
                    }

                    const int layer_to_load = io_cursor++;
                    io_ready[slot] = -1;
                    slot_pending[slot] = true;

                    lock.unlock();
                    pending_layers.push_back(submit_layer_predicted(layer_to_load, slot));
                    lock.lock();

                    if (gen != io_generation || io_thread_stop) break;
                }

                io->flush();

                if (io_cursor >= io_target && pending_layers.empty()) {
                    should_exit = true;
                } else if (pending_layers.empty()) {
                    io_cv.wait(lock);
                    continue;
                }
            }

            if (should_exit) {
                break;
            }

            pending_layer_io done = std::move(pending_layers.front());
            pending_layers.pop_front();

            const int64_t t0 = ggml_time_us();
            complete_experts_batch(done, *io, true);
            stats_.t_preload_us += (double)(ggml_time_us() - t0);

            {
                std::lock_guard<std::mutex> lock(io_mutex);
                slot_pending[done.buf_idx] = false;
                if (gen == io_generation) {
                    io_ready[done.buf_idx] = done.layer_idx;
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

void llama_ssd_manager::request_ready(int il, const int32_t * selected_experts, int n_selected) {
    if (il < 0 || il >= (int)layers.size()) return;
    if (layers[il].tensors.empty()) return;

    if (il < (int)pending_miss_active.size() && pending_miss_active[il]) {
        finish_ready(il);
    }
    if (il < (int)pending_down_active.size() && pending_down_active[il]) {
        finish_down_ready(il);
    }

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
    std::vector<int> missing_experts;
    for (int s = 0; s < n_selected; s++) {
        int eidx = selected_experts[s];
        for (size_t t = 0; t < layer.tensors.size(); t++) {
            if (eidx < 0 || eidx >= layer.tensors[t].n_expert) continue;
            if (bs.expert_loaded[t][eidx]) { stats_.n_prefetch_hits++; continue; }
            stats_.n_prefetch_misses++;
            missing_experts.push_back(eidx);
        }
    }
    if (!missing_experts.empty()) {
        pending_layer_io pending = submit_experts_batch(il, missing_experts, use_buf, *io_miss, tensor_stage::first);
        pending_layer_io pending_down = submit_experts_batch(il, missing_experts, use_buf, *io_miss, tensor_stage::down);
        io_miss->flush();
        pending_miss_io[il] = std::move(pending);
        pending_down_io[il] = std::move(pending_down);
        pending_miss_active[il] = !pending_miss_io[il].work.empty();
        pending_down_active[il] = !pending_down_io[il].work.empty();
    } else if (il < (int)pending_miss_active.size()) {
        pending_miss_active[il] = false;
        pending_down_active[il] = false;
    }
}

void llama_ssd_manager::finish_ready(int il) {
    if (il < 0 || il >= (int)layers.size()) return;
    if (layers[il].tensors.empty()) return;

    if (il < (int)pending_miss_active.size() && pending_miss_active[il]) {
        auto & pending = pending_miss_io[il];
        int64_t t0 = ggml_time_us();
        complete_experts_batch(pending, *io_miss, false);
        const int64_t t1 = ggml_time_us();
        stats_.t_io_us += (double)(t1 - t0);
        if (pending.submitted_at_us > 0) {
            stats_.t_miss_wall_us += (double)(t1 - pending.submitted_at_us);
        }
        stats_.n_miss_batches++;
        stats_.n_read_reqs += pending.reads.size();
        stats_.max_batch_reads = std::max<uint64_t>(stats_.max_batch_reads, pending.reads.size());
        pending_miss_active[il] = false;
        pending_miss_io[il] = {};
    }

    activate_layer(il);
}

void llama_ssd_manager::finish_down_ready(int il) {
    if (il < 0 || il >= (int)layers.size()) return;
    if (layers[il].tensors.empty()) return;

    if (il < (int)pending_down_active.size() && pending_down_active[il]) {
        auto & pending = pending_down_io[il];
        int64_t t0 = ggml_time_us();
        complete_experts_batch(pending, *io_miss, false);
        const int64_t t1 = ggml_time_us();
        stats_.t_io_us += (double)(t1 - t0);
        if (pending.submitted_at_us > 0) {
            stats_.t_down_wall_us += (double)(t1 - pending.submitted_at_us);
        }
        stats_.n_down_batches++;
        stats_.n_read_reqs += pending.reads.size();
        stats_.max_batch_reads = std::max<uint64_t>(stats_.max_batch_reads, pending.reads.size());
        pending_down_active[il] = false;
        pending_down_io[il] = {};
    }

    activate_layer(il);
}

void llama_ssd_manager::ensure_ready(int il, const int32_t * selected_experts, int n_selected) {
    request_ready(il, selected_experts, n_selected);
    finish_ready(il);
    finish_down_ready(il);
}

void llama_ssd_manager::activate_layer(int il) {
    if (il < 0 || il >= (int)layers.size()) return;
    if (layers[il].tensors.empty()) return;
    int buf_idx = il % n_buf_slots_;
    for (const auto & ti : layers[il].tensors)
        ti.tensor->data = (uint8_t *)buffers[buf_idx] + ti.tensor_offset;
}

std::vector<int> llama_ssd_manager::predict_experts(int il) const {
    if (il < 0 || il >= (int)last_selected.size()) return {};
    if (predict_history_ == 0) return {};
    if (selected_history[il].empty()) {
        return last_selected[il].empty() ? std::vector<int>{0, 1} : last_selected[il];
    }

    std::vector<int> result;
    const auto & history = selected_history[il];
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        for (int expert : *it) {
            if (std::find(result.begin(), result.end(), expert) == result.end()) {
                result.push_back(expert);
            }
        }
    }
    return result;
}

void llama_ssd_manager::update_prediction(int il, const int32_t * sel, int n) {
    if (il < 0 || il >= (int)last_selected.size()) return;

    last_selected[il].assign(sel, sel + n);

    auto & history = selected_history[il];
    history.emplace_back(sel, sel + n);
    while ((int)history.size() > predict_history_) {
        history.erase(history.begin());
    }
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
