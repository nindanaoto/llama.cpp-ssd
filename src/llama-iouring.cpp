#include "llama-iouring.h"
#include "llama-impl.h"

#include <cassert>
#include <cstring>
#include <unordered_map>

#ifdef LLAMA_IO_URING
#include <liburing.h>
#include <unistd.h>

struct llama_io_uring::impl {
    struct io_uring ring;
    bool ring_initialized = false;
    uint64_t next_ticket = 1;
    int n_pending = 0;

    // Track completion results: ticket -> bytes_read (-1 if not yet complete)
    std::unordered_map<uint64_t, ssize_t> results;

    impl(int queue_depth) {
        int ret = io_uring_queue_init(queue_depth, &ring, 0);
        if (ret < 0) {
            LLAMA_LOG_WARN("%s: io_uring_queue_init failed (%s), falling back to sync I/O\n",
                __func__, strerror(-ret));
            return;
        }
        ring_initialized = true;
        LLAMA_LOG_INFO("%s: io_uring initialized, queue depth = %d\n", __func__, queue_depth);
    }

    ~impl() {
        if (ring_initialized) {
            io_uring_queue_exit(&ring);
        }
    }

    uint64_t submit_read(int fd, void * dest, size_t size, off_t offset) {
        if (!ring_initialized) {
            // Sync fallback
            ssize_t ret = pread(fd, dest, size, offset);
            uint64_t ticket = next_ticket++;
            results[ticket] = ret;
            return ticket;
        }

        struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            // SQ full — drain some completions first
            drain_cqe(1);
            sqe = io_uring_get_sqe(&ring);
            if (!sqe) {
                // Still no room — do sync fallback for this one
                ssize_t ret = pread(fd, dest, size, offset);
                uint64_t ticket = next_ticket++;
                results[ticket] = ret;
                return ticket;
            }
        }

        uint64_t ticket = next_ticket++;
        io_uring_prep_read(sqe, fd, dest, (unsigned)size, offset);
        io_uring_sqe_set_data64(sqe, ticket);

        int ret = io_uring_submit(&ring);
        if (ret < 0) {
            // Submit failed — sync fallback
            ssize_t bytes = pread(fd, dest, size, offset);
            results[ticket] = bytes;
            return ticket;
        }

        n_pending++;
        results[ticket] = -1; // not yet complete
        return ticket;
    }

    ssize_t wait_for(uint64_t ticket) {
        // Check if already completed
        auto it = results.find(ticket);
        if (it != results.end() && it->second >= 0) {
            ssize_t ret = it->second;
            results.erase(it);
            return ret;
        }

        if (!ring_initialized) {
            // Sync fallback — should already be in results
            if (it != results.end()) {
                ssize_t ret = it->second;
                results.erase(it);
                return ret;
            }
            return -1;
        }

        // Wait for completions until our ticket is done
        while (true) {
            struct io_uring_cqe * cqe;
            int ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) {
                LLAMA_LOG_ERROR("%s: io_uring_wait_cqe failed: %s\n", __func__, strerror(-ret));
                return -1;
            }

            uint64_t completed_ticket = io_uring_cqe_get_data64(cqe);
            ssize_t bytes = cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            n_pending--;

            results[completed_ticket] = bytes;

            if (completed_ticket == ticket) {
                results.erase(completed_ticket);
                return bytes;
            }
        }
    }

    void wait_all() {
        if (!ring_initialized) return;

        while (n_pending > 0) {
            struct io_uring_cqe * cqe;
            int ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) {
                LLAMA_LOG_ERROR("%s: io_uring_wait_cqe failed: %s\n", __func__, strerror(-ret));
                return;
            }

            uint64_t ticket = io_uring_cqe_get_data64(cqe);
            results[ticket] = cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            n_pending--;
        }
    }

    int reap_completed(std::vector<uint64_t> & completed) {
        if (!ring_initialized) return 0;

        int reaped = 0;
        struct io_uring_cqe * cqe;
        while (io_uring_peek_cqe(&ring, &cqe) == 0) {
            uint64_t ticket = io_uring_cqe_get_data64(cqe);
            results[ticket] = cqe->res;
            completed.push_back(ticket);
            io_uring_cqe_seen(&ring, cqe);
            n_pending--;
            reaped++;
        }
        return reaped;
    }

    void drain_cqe(int min_count) {
        struct io_uring_cqe * cqe;
        for (int i = 0; i < min_count && n_pending > 0; i++) {
            int ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) break;
            uint64_t ticket = io_uring_cqe_get_data64(cqe);
            results[ticket] = cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            n_pending--;
        }
    }
};

#else // !LLAMA_IO_URING

#include <unistd.h>

// Synchronous fallback when io_uring is not available
struct llama_io_uring::impl {
    uint64_t next_ticket = 1;
    std::unordered_map<uint64_t, ssize_t> results;

    impl(int /*queue_depth*/) {
        LLAMA_LOG_WARN("%s: io_uring not available on this platform, using sync I/O\n", __func__);
    }
    ~impl() = default;

    uint64_t submit_read(int fd, void * dest, size_t size, off_t offset) {
        ssize_t ret = pread(fd, dest, size, offset);
        uint64_t ticket = next_ticket++;
        results[ticket] = ret;
        return ticket;
    }

    ssize_t wait_for(uint64_t ticket) {
        auto it = results.find(ticket);
        if (it == results.end()) return -1;
        ssize_t ret = it->second;
        results.erase(it);
        return ret;
    }

    void wait_all() {}

    int reap_completed(std::vector<uint64_t> & completed) {
        (void)completed;
        return 0;
    }

    int n_pending = 0;
    bool ring_initialized = false;
};

#endif // LLAMA_IO_URING

// Public interface delegates to impl

llama_io_uring::llama_io_uring(int queue_depth) : pimpl(new impl(queue_depth)) {}

llama_io_uring::~llama_io_uring() { delete pimpl; }

uint64_t llama_io_uring::submit_read(int fd, void * dest, size_t size, off_t offset) {
    return pimpl->submit_read(fd, dest, size, offset);
}

ssize_t llama_io_uring::wait_for(uint64_t ticket) {
    return pimpl->wait_for(ticket);
}

void llama_io_uring::wait_all() {
    pimpl->wait_all();
}

int llama_io_uring::reap_completed(std::vector<uint64_t> & completed) {
    return pimpl->reap_completed(completed);
}

int llama_io_uring::pending() const {
    return pimpl->n_pending;
}

bool llama_io_uring::is_async() const {
    return pimpl->ring_initialized;
}
