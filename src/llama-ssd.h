#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class llama_io_uring;

struct ssd_read_chunk {
    uint16_t file_idx;       // stripe file index
    size_t   file_offset;    // byte offset in stripe file
    size_t   dst_offset;     // byte offset within one expert slice
    size_t   size;           // bytes to read
};

// Metadata for a single expert slice within a 3D expert tensor
struct ssd_expert_slice {
    uint16_t file_idx;       // which GGUF file
    size_t   file_offset;    // absolute byte offset in file for this expert's data
    size_t   slice_size;     // bytes for one expert = tensor->nb[2]
    std::vector<ssd_read_chunk> stripe_chunks;
};

// Metadata for one MoE expert tensor (e.g., ffn_gate_up_exps for a given layer)
struct ssd_tensor_info {
    ggml_tensor * tensor = nullptr;
    int           n_expert = 0;
    size_t        tensor_offset = 0;
    std::vector<ssd_expert_slice> experts; // [n_expert]
};

// Metadata for one MoE layer
struct ssd_layer_info {
    int layer_idx = -1;
    std::vector<ssd_tensor_info> tensors;
    size_t total_bytes = 0;
    ggml_tensor * topk_tensor = nullptr;
};

// State of one buffer slot
struct ssd_buf_state {
    int layer_idx = -1;
    std::vector<std::vector<bool>> expert_loaded; // [tensor_idx][expert_idx]
};

class llama_ssd_manager {
public:
    explicit llama_ssd_manager(
            int n_buf_slots = 2,
            std::string stripe_dirs = {},
            std::string stripe_name = {},
            size_t stripe_chunk_size = 1024 * 1024,
            size_t read_chunk_size = 0,
            bool allow_direct_io = true,
            int predict_history = 1,
            int prefetch_window = 1,
            int cpu_layers = 0,
            size_t cache_size = 0);
    ~llama_ssd_manager();

    void register_tensor(ggml_tensor * tensor, uint16_t file_idx, size_t file_offset, int layer_idx);
    void pre_alloc_scan(struct ggml_context * ctx);
    void finalize_layout(ggml_backend_buffer_type_t buft);
    void init(const std::vector<std::string> & file_paths);

    static bool is_expert_weight(const char * name);
    bool should_offload_weight(const char * name) const;

    // === Inference time ===

    // Signal start of a new token. Wakes the I/O thread to prefetch from layer 0.
    void begin_token();

    // Queue exact selected-expert reads for layer il after routing is known.
    void request_ready(int il, const int32_t * selected_experts, int n_selected);

    // Wait for any queued exact reads for layer il and point tensors at the active slot.
    void finish_ready(int il);
    void finish_down_ready(int il);

    // Wait for layer il's experts to be loaded, handle mispredictions.
    void ensure_ready(int il, const int32_t * selected_experts, int n_selected);

    // Point all expert tensors for layer il to the active buffer's data.
    void activate_layer(int il);

    void update_prediction(int il, const int32_t * selected_experts, int n_selected);
    std::vector<int> predict_experts(int il) const;

    int n_layers() const { return (int)layers.size(); }
    int n_buf_slots() const { return n_buf_slots_; }

    // Legacy API (unused)
    void preload_all_layers();
    void scan_graph_for_topk(struct ggml_cgraph * gf);
    bool update_predictions_from_graph();

    struct stats {
        uint64_t n_prefetch_hits = 0;
        uint64_t n_prefetch_misses = 0;
        uint64_t n_loads = 0;
        uint64_t n_prefetch_loads = 0;
        uint64_t n_miss_loads = 0;
        uint64_t n_cache_hits = 0;
        uint64_t n_read_reqs = 0;
        uint64_t n_miss_batches = 0;
        uint64_t n_down_batches = 0;
        uint64_t max_batch_reads = 0;
        uint64_t bytes_loaded = 0;
        uint64_t bytes_prefetched = 0;
        uint64_t bytes_missed = 0;
        uint64_t bytes_cache_hits = 0;
        double   t_io_us = 0;
        double   t_wait_us = 0;
        double   t_preload_us = 0;
        double   t_miss_wall_us = 0;
        double   t_down_wall_us = 0;
    };
    stats get_stats() const { return stats_; }
    void  reset_stats() { stats_ = {}; }

private:
    struct pending_work_item {
        int tensor_idx;
        int expert_idx;
    };

    struct pending_read_item {
        uint64_t ticket;
        size_t expected;
    };

    struct pending_layer_io {
        int layer_idx = -1;
        int buf_idx = -1;
        int64_t submitted_at_us = 0;
        std::vector<pending_work_item> work;
        std::vector<pending_read_item> reads;
    };

    struct cache_key {
        int layer_idx;
        int tensor_idx;
        int expert_idx;
    };

    struct cache_entry {
        bool valid = false;
        size_t size = 0;
        std::shared_ptr<std::vector<uint8_t>> data;
        std::list<cache_key>::iterator lru_it;
    };

    void load_expert_sync(int layer_idx, int tensor_idx, int expert_idx, int buf_idx);
    int get_read_fd(uint16_t file_idx) const;
    enum class tensor_stage {
        all,
        first,
        down,
    };

    static bool is_down_tensor(const ssd_tensor_info & info);
    static bool tensor_matches_stage(const ssd_tensor_info & info, tensor_stage stage);
    void load_experts_batch(int layer_idx, const std::vector<int> & experts, int buf_idx, llama_io_uring & engine, tensor_stage stage);
    pending_layer_io submit_experts_batch(int layer_idx, const std::vector<int> & experts, int buf_idx, llama_io_uring & engine, tensor_stage stage);
    void complete_experts_batch(const pending_layer_io & pending, llama_io_uring & engine, bool is_prefetch);
    void load_stripe_manifest(const std::vector<std::string> & file_paths);
    int get_stripe_fd(uint16_t file_idx) const;
    void io_thread_func();
    pending_layer_io submit_layer_predicted(int il, int buf_idx);
    bool try_cache_hit(int layer_idx, int tensor_idx, int expert_idx, uint8_t * dest, size_t size);
    void cache_insert(int layer_idx, int tensor_idx, int expert_idx, const uint8_t * src, size_t size);

    std::vector<ssd_layer_info> layers;

    int n_buf_slots_;
    int predict_history_;
    int prefetch_window_;
    int cpu_layers_;
    size_t cache_size_limit_ = 0;
    size_t cache_size_used_ = 0;
    std::vector<void *> buffers;
    size_t buf_size = 0;
    std::vector<ssd_buf_state> buf_states;
    std::vector<std::vector<int>> last_selected;
    std::vector<std::vector<std::vector<int>>> selected_history;
    std::vector<std::vector<std::vector<uint32_t>>> load_counts;
    std::vector<pending_layer_io> pending_miss_io;
    std::vector<pending_layer_io> pending_down_io;
    std::vector<bool> pending_miss_active;
    std::vector<bool> pending_down_active;
    std::vector<std::vector<std::vector<cache_entry>>> cache_entries;
    std::list<cache_key> cache_lru;
    std::mutex cache_mutex;

    ggml_backend_buffer_t dummy_buf = nullptr;

    std::vector<int> dio_fds;
    std::vector<int> reg_fds;
    size_t dio_alignment = 4096;
    bool use_direct_io = false;
    bool allow_direct_io = true;
    bool use_striped_io = false;

    std::string stripe_dirs_arg;
    std::string stripe_name_arg;
    size_t stripe_chunk_size = 1024 * 1024;
    size_t read_chunk_size = 0;
    std::vector<std::string> stripe_dirs;
    std::vector<std::string> stripe_paths;
    std::vector<int> stripe_dio_fds;
    std::vector<int> stripe_reg_fds;

    struct tensor_loc { int layer_idx; int tensor_idx; };
    std::unordered_map<std::string, tensor_loc> tensor_map;

    bool initialized = false;

    // Async I/O engine for batch-submitting expert reads
    std::unique_ptr<llama_io_uring> io;
    std::unique_ptr<llama_io_uring> io_miss;

    // === Single background I/O thread ===
    std::thread io_thread;
    std::mutex  io_mutex;
    std::condition_variable io_cv;      // I/O thread waits on this
    std::condition_variable compute_cv; // compute thread waits on this

    int io_cursor = 0;       // next layer to load
    int compute_cursor = 0;  // layers before this are done computing
    int io_target = 0;       // total layers to load this token
    std::vector<int> io_ready; // per-slot: which layer is loaded (-1 = empty)
    int io_generation = 0;   // incremented per token

    bool io_thread_stop = false;

    stats stats_;
};
