#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
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
};

// State of one double-buffer slot
struct ssd_buf_state {
    int layer_idx = -1;                   // which layer's data is here (-1 = none)
    std::vector<std::vector<bool>> expert_loaded; // [tensor_idx][expert_idx]
};

class llama_ssd_manager {
public:
    llama_ssd_manager();
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

    // Ensure that the selected experts for layer il are loaded in the active buffer.
    // Blocks until all needed expert slices are available.
    void ensure_ready(int il, const int32_t * selected_experts, int n_selected);

    // Point all expert tensors for layer il to the active buffer's data.
    void activate_layer(int il);

    // Start speculative prefetch for layer il into the next buffer.
    void prefetch_start(int il);

    // Get predicted expert indices for layer il (based on last-used heuristic)
    std::vector<int> predict_experts(int il) const;

    // Update prediction state after seeing actual expert selection
    void update_prediction(int il, const int32_t * selected_experts, int n_selected);

    // Get number of registered layers
    int n_layers() const { return (int)layers.size(); }

    // Statistics
    struct stats {
        uint64_t n_prefetch_hits = 0;
        uint64_t n_prefetch_misses = 0;
        uint64_t n_loads = 0;
        uint64_t bytes_loaded = 0;
        double   t_io_us = 0;          // total I/O time (sync loads only)
        double   t_wait_us = 0;        // total time waiting for async completions
    };
    stats get_stats() const { return stats_; }
    void  reset_stats() { stats_ = {}; }

private:
    // Load a specific expert slice synchronously (blocking)
    void load_expert_sync(int layer_idx, int tensor_idx, int expert_idx, int buf_idx);

    // Submit async read for a specific expert slice, returns ticket
    uint64_t load_expert_async(int layer_idx, int tensor_idx, int expert_idx, int buf_idx);

    // Get the file descriptor to use for reads (O_DIRECT if available, else regular)
    int get_read_fd(uint16_t file_idx) const;

    std::vector<ssd_layer_info> layers;

    // Double buffers
    void * buffers[2] = {nullptr, nullptr};
    size_t buf_size = 0;
    int    active_buf = 0; // which buffer is currently used for compute (0 or 1)

    ssd_buf_state buf_states[2];

    // Prediction state: last-used experts per layer
    std::vector<std::vector<int>> last_selected; // [layer_idx] -> expert indices

    // Track which buffer the last prefetch targeted and for which layer
    int prefetch_target_buf = -1;
    int prefetch_target_layer = -1;

    // Dummy buffer for SSD tensors (so backend scheduler doesn't try to allocate them)
    ggml_backend_buffer_t dummy_buf = nullptr;

    // Async I/O engine
    std::unique_ptr<llama_io_uring> io;

    // Track in-flight async reads: ticket -> {layer_idx, tensor_idx, expert_idx, buf_idx}
    struct async_read_info {
        int layer_idx, tensor_idx, expert_idx, buf_idx;
    };
    std::unordered_map<uint64_t, async_read_info> in_flight;

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

    stats stats_;
};
