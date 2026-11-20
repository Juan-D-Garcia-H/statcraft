// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file parallel.hpp
 * @brief Minimal @c parallelFor built on @c std::thread (no dependencies).
 *
 * Properties:
 *  - dynamic scheduling through an atomic counter, which balances workloads
 *    with subjects of unequal cost;
 *  - the result does not depend on the number of threads provided each index
 *    writes only to its own slot (per-subject samplers keep their own RNG);
 *  - the first exception thrown by a worker is rethrown on the calling thread.
 *
 * @warning The code executed inside the loop (user model/prediction code) must
 *          be thread-safe.
 */

#pragma once
#include "numcast.hpp"
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace mcmckl {

/**
 * @brief Returns the number of hardware threads.
 * @return @c std::thread::hardware_concurrency(), or 1 when it reports 0.
 */
inline int hardwareThreads() {
    const unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 1 : static_cast<int>(h);
}

/**
 * @brief Executes @p f(0) ... @p f(n-1) using up to @p nthreads threads.
 *
 * @tparam F Callable accepting an integer index.
 * @param n Number of indices to process.
 * @param nthreads Number of worker threads. Values <= 0 select all hardware
 *        threads; 1 runs serially.
 * @param f Function invoked once per index.
 * @throws Any exception thrown by @p f; the first one is propagated after all
 *         workers have been joined.
 */
template <class F>
inline void parallelFor(int n, int nthreads, F&& f) {
    if (nthreads <= 0) nthreads = hardwareThreads();
    nthreads = std::min(nthreads, n);
    if (nthreads <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    std::atomic<int> next{0};
    std::exception_ptr err;
    std::mutex mtx;
    auto worker = [&]() {
        for (;;) {
            const int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) return;
            try { f(i); }
            catch (...) {
                std::lock_guard<std::mutex> lk(mtx);
                if (!err) err = std::current_exception();
                next.store(n);          // Abort the remaining work.
                return;
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(uz(nthreads - 1));
    for (int t = 1; t < nthreads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    if (err) std::rethrow_exception(err);
}

} // namespace mcmckl
