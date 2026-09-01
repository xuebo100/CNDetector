#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace pdms
{

namespace detail
{
/// Process-wide thread cap set by setMaxThreads (0 = auto/hardware).
inline std::atomic<int> gMaxThreads{0};

/// Extra worker threads currently alive across all parallelFor regions.
/// Nested regions consult this budget, so an inner loop running inside an
/// already-saturated outer loop degrades to serial instead of oversubscribing.
inline std::atomic<int> gExtraThreads{0};

inline int acquireExtraThreads(int want, int cap)
{
    if (want <= 0 || cap <= 0)
    {
        return 0;
    }
    const int previous = gExtraThreads.fetch_add(want);
    const int allowed = std::clamp(cap - previous, 0, want);
    if (allowed < want)
    {
        gExtraThreads.fetch_sub(want - allowed);
    }
    return allowed;
}

inline void releaseExtraThreads(int count)
{
    if (count > 0)
    {
        gExtraThreads.fetch_sub(count);
    }
}
}  // namespace detail

/// Set the process-wide parallelism cap (0 restores the hardware default).
inline void setMaxThreads(int count)
{
    detail::gMaxThreads.store(count <= 0 ? 0 : count);
}

/// Effective parallelism cap: min(configured, hardware), at least 1.
inline int maxThreads()
{
    const int hardware
        = static_cast<int>(std::thread::hardware_concurrency());
    const int base = hardware > 0 ? hardware : 1;
    const int cap = detail::gMaxThreads.load();
    return cap > 0 ? std::min(cap, base) : base;
}

/**
 * Runs fn(index, workerIdx) for every index in [begin, end).
 *
 * - workerIdx is always < maxThreads(), so per-worker scratch pools sized to
 *   maxThreads() slots are safe.
 * - Indices are handed out dynamically; callers MUST write results to
 *   index-addressed slots (results[i]) so the outcome is independent of
 *   scheduling and thread count.
 * - Worker threads are budgeted globally; nested calls fall back to running
 *   serially on the calling thread when the budget is exhausted.
 * - The first exception thrown by fn is rethrown on the calling thread after
 *   all workers finish; remaining indices are skipped on error.
 */
template <typename Fn>
void parallelFor(int begin, int end, Fn &&fn)
{
    const int total = end - begin;
    if (total <= 0)
    {
        return;
    }

    const int wanted = std::min(total, maxThreads()) - 1;
    const int extra = detail::acquireExtraThreads(wanted, maxThreads() - 1);

    if (extra == 0)
    {
        for (int i = begin; i < end; ++i)
        {
            fn(i, 0);
        }
        return;
    }

    std::atomic<int> next{begin};
    std::atomic<bool> failed{false};
    std::exception_ptr firstError;
    std::mutex errorMutex;

    auto work = [&](int workerIdx)
    {
        while (!failed.load(std::memory_order_relaxed))
        {
            const int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= end)
            {
                break;
            }
            try
            {
                fn(i, workerIdx);
            }
            catch (...)
            {
                {
                    const std::lock_guard<std::mutex> lock(errorMutex);
                    if (!firstError)
                    {
                        firstError = std::current_exception();
                    }
                }
                failed.store(true, std::memory_order_relaxed);
                break;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(extra));
    for (int w = 1; w <= extra; ++w)
    {
        threads.emplace_back(work, w);
    }
    work(0);
    for (auto &thread : threads)
    {
        thread.join();
    }
    detail::releaseExtraThreads(extra);

    if (firstError)
    {
        std::rethrow_exception(firstError);
    }
}

}  // namespace pdms

#endif  // PARALLEL_FOR_H
