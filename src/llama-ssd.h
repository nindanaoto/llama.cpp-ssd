#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class llama_io_uring;

// Metadata for a single expert slice within a 3D expert tensor
struct ssd_expert_slice {
    uint16_t file_idx;       // which GGUF file
    size_t   file_offset;    // absolute byte offset in file for this expert's data
    size_t   slice_size;     // bytes for one expert = tensor->nb[2]
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
    explicit llama_ssd_manager(int n_buf_slots = 2);
    ~llama_ssd_manager();

    void register_tensor(ggml_tensor * tensor, uint16_t file_idx, size_t file_offset, int layer_idx);
    void pre_alloc_scan(struct ggml_context * ctx);
    void finalize_layout(ggml_backend_buffer_type_t buft);
    void init(const std::vector<std::string> & file_paths);

    static bool is_expert_weight(const char * name);

    // === Inference time ===

    // Signal start of a new token. Wakes the I/O thread to prefetch from layer 0.
    void begin_token();

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
        uint64_t bytes_loaded = 0;
        double   t_io_us = 0;
        double   t_wait_us = 0;
        double   t_preload_us = 0;
    };
    stats get_stats() const { return stats_; }
    void  reset_stats() { stats_ = {}; }

private:
    void load_expert_sync(int layer_idx, int tensor_idx, int expert_idx, int buf_idx);
    int get_read_fd(uint16_t file_idx) const;
    void io_thread_func();
    void load_layer_predicted(int il, int buf_idx);

    std::vector<ssd_layer_info> layers;

    int n_buf_slots_;
    std::vector<void *> buffers;
    size_t buf_size = 0;
    std::vector<ssd_buf_state> buf_states;
    std::vector<std::vector<int>> last_selected;

    ggml_backend_buffer_t dummy_buf = nullptr;

    std::vector<int> dio_fds;
    std::vector<int> reg_fds;
    size_t dio_alignment = 4096;
    bool use_direct_io = false;

    struct tensor_loc { int layer_idx; int tensor_idx; };
    std::unordered_map<std::string, tensor_loc> tensor_map;

    bool initialized = false;

    // Async I/O engine for batch-submitting expert reads
    std::unique_ptr<llama_io_uring> io;

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
