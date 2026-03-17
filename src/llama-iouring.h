#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <vector>

// Async I/O wrapper using Linux io_uring.
// Falls back to synchronous pread when io_uring is not available.
class llama_io_uring {
public:
    // queue_depth: max number of in-flight I/O requests
    explicit llama_io_uring(int queue_depth = 64);
    ~llama_io_uring();

    // Submit an async read. Returns a ticket ID for later completion check.
    uint64_t submit_read(int fd, void * dest, size_t size, off_t offset);

    // Block until a specific ticket completes. Returns bytes read.
    ssize_t wait_for(uint64_t ticket);

    // Block until all outstanding reads complete.
    void wait_all();

    // Non-blocking: reap any completed reads. Returns number reaped.
    int reap_completed(std::vector<uint64_t> & completed);

    // Number of outstanding (submitted but not completed) requests
    int pending() const;

    // Whether io_uring is actually being used (vs sync fallback)
    bool is_async() const;

private:
    struct impl;
    impl * pimpl;
};
