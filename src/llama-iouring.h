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

    // Queue an async read (does NOT submit to kernel yet). Returns a ticket ID.
    // Call flush() after queuing all reads to submit them in a single syscall.
    uint64_t submit_read(int fd, void * dest, size_t size, off_t offset);

    // Submit all queued SQEs to the kernel in one syscall.
    // This is the key to saturating RAID0 bandwidth: batch many reads, submit once.
    void flush();

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
