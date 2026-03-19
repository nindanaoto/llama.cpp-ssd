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
#include <unordered_set>
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
    size_t        tensor_offset = 0; // offset of this tensor's region within the layer buffer
    std::vector<ssd_expert_slice> experts; // [n_expert]
};

// Metadata for one MoE layer
struct ssd_layer_info {
    int layer_idx = -1;
    std::vector<ssd_tensor_info> tensors; // typically 2-3 (gate_up_exps + down_exps, or gate + up + down)
    size_t total_bytes = 0;               // sum of ggml_nbytes for all expert tensors in this layer
    ggml_tensor * topk_tensor = nullptr;  // ffn_moe_topk output for reading actual selections post-compute
};

// State of one buffer slot
struct ssd_buf_state {
    int layer_idx = -1;                   // which layer's data is here (-1 = none)
    std::vector<std::vector<bool>> expert_loaded; // [tensor_idx][expert_idx]
};

class llama_ssd_manager {
public:
    // n_buf_slots: number of circular buffer slots (default 2).
    // More slots = deeper I/O pipeline = better RAID0 bandwidth utilization,
    // but each slot costs max_layer_bytes of RAM.
    // n_io_threads: number of parallel I/O threads per layer (default 1).
    // For RAID0, set to number of devices (e.g., 4) to saturate bandwidth.
    explicit llama_ssd_manager(int n_buf_slots = 2, int n_io_threads = 1);
    ~llama_ssd_manager();

    // Model load time: register an expert tensor for SSD offloading
    // Called from load_all_data() when a tensor is identified as an SSD-offloaded expert tensor
    void register_tensor(ggml_tensor * tensor, uint16_t file_idx, size_t file_offset, int layer_idx);

    // Phase 1 (before buffer allocation): scan contexts, identify SSD tensors,
    // allocate double buffers, and set tensor->data so the allocator skips them.
    // This prevents RAM from being allocated for expert tensors.
    void pre_alloc_scan(struct ggml_context * ctx);

    // Phase 2 (after pre_alloc_scan for all contexts): finalize buffer layout.
    // buft: buffer type used for CPU tensors (needed for dummy buffer)
    void finalize_layout(ggml_backend_buffer_type_t buft);

    // After all tensors registered (during load_all_data), open file handles for reading
    void init(const std::vector<std::string> & file_paths);

    // Check if a tensor name matches the pattern for SSD-offloaded expert tensors
    static bool is_expert_weight(const char * name);

    // === Inference time ===

    // Start the background I/O thread for a new token.
    // Begins prefetching from layer 0 using predictions.
    void begin_token();

    // Signal that compute has reached layer il with the given expert selections.
    // The I/O thread will prioritize loading these experts if not already loaded.
    // Blocks until the requested experts are available in the buffer.
    void ensure_ready(int il, const int32_t * selected_experts, int n_selected);

    // Point all expert tensors for layer il to the active buffer's data.
    void activate_layer(int il);

    // Update prediction state after seeing actual expert selection
    void update_prediction(int il, const int32_t * selected_experts, int n_selected);

    // Get predicted expert indices for layer il (based on last-used heuristic)
    std::vector<int> predict_experts(int il) const;

    // Get number of registered layers
    int n_layers() const { return (int)layers.size(); }

    // Get number of buffer slots (I/O pipeline depth)
    int n_buf_slots() const { return n_buf_slots_; }

    // Get which layer the pending prefetch targets (-1 if none)
    int get_prefetch_target_layer() const { return prefetch_target_layer; }

    // === Preload-all mode: load ALL layers' predicted experts in one burst ===

    // Preload predicted experts for ALL layers via io_uring, wait for completion,
    // and set all tensor->data pointers. No callback needed during graph compute.
    void preload_all_layers();

    // After graph build: scan graph for topk tensors per layer
    void scan_graph_for_topk(struct ggml_cgraph * gf);

    // After graph compute: read actual expert selections, update predictions,
    // and return true if any misprediction was detected (caller should recompute)
    bool update_predictions_from_graph();

    // Statistics
    struct stats {
        uint64_t n_prefetch_hits = 0;
        uint64_t n_prefetch_misses = 0;
        uint64_t n_loads = 0;
        uint64_t bytes_loaded = 0;
        double   t_io_us = 0;          // total I/O time (sync loads only)
        double   t_wait_us = 0;        // total time compute waited for I/O
        double   t_preload_us = 0;     // total time in preload_all_layers
    };
    stats get_stats() const { return stats_; }
    void  reset_stats() { stats_ = {}; }

private:
    // Load a specific expert slice synchronously (blocking)
    void load_expert_sync(int layer_idx, int tensor_idx, int expert_idx, int buf_idx);

    // Get the file descriptor to use for reads (O_DIRECT if available, else regular)
    int get_read_fd(uint16_t file_idx) const;

    // I/O worker thread function (each worker independently picks layers)
    void io_worker_func();

    // Load all predicted experts for a layer into a buffer slot (called by worker)
    void load_layer_predicted(int il, int buf_idx);

    std::vector<ssd_layer_info> layers;

    // Circular buffer: N slots, each holding one layer's expert data.
    // Layer il maps to slot (il % n_buf_slots_).
    int n_buf_slots_;
    int n_io_threads_;
    std::vector<void *> buffers;
    size_t buf_size = 0; // size of each slot

    std::vector<ssd_buf_state> buf_states;

    // Prediction state: last-used experts per layer
    std::vector<std::vector<int>> last_selected; // [layer_idx] -> expert indices

    // Track which buffer the last prefetch targeted and for which layer
    int prefetch_target_buf = -1;
    int prefetch_target_layer = -1;

    // Dummy buffer for SSD tensors (so backend scheduler doesn't try to allocate them)
    ggml_backend_buffer_t dummy_buf = nullptr;

    // File descriptors for expert data reading
    // Direct I/O fds (O_DIRECT) for aligned reads, regular fds as fallback
    std::vector<int> dio_fds;     // O_DIRECT file descriptors (-1 if not available)
    std::vector<int> reg_fds;     // regular file descriptors (fallback)
    size_t dio_alignment = 4096;  // O_DIRECT alignment requirement
    bool use_direct_io = false;   // whether O_DIRECT is active

    // Mapping from tensor name to (layer_idx, tensor_idx) for fast lookup
    struct tensor_loc {
        int layer_idx;
        int tensor_idx;
    };
    std::unordered_map<std::string, tensor_loc> tensor_map;

    // Registered but not yet initialized
    bool initialized = false;

    // === Background I/O thread state ===

    std::vector<std::thread> io_workers; // n_io_threads_ persistent workers
    std::mutex  io_mutex;
    std::condition_variable io_cv;      // I/O workers wait on this
    std::condition_variable compute_cv; // compute thread waits on this

    // Next layer to assign to an I/O worker (shared cursor, protected by io_mutex).
    int io_cursor = 0;

    // Which layer compute needs next. Set by begin_token / ensure_ready.
    // Protected by io_mutex.
    int compute_cursor = 0;

    // Total layers to prefetch for current token (set by begin_token).
    int io_target = 0;

    // Per-slot readiness: io_ready[slot] = layer that is fully loaded in that slot (-1 = empty)
    std::vector<int> io_ready;

    bool io_thread_running = false;
    bool io_thread_stop = false;

    // Number of workers actively loading (outside the lock, doing pread).
    // begin_token() waits for this to reach 0 before resetting state.
    int io_active_workers = 0;

    // Generation counter: incremented by begin_token(). Stale completions
    // from previous tokens are discarded by checking this counter.
    int io_generation = 0;

    stats stats_;
};
