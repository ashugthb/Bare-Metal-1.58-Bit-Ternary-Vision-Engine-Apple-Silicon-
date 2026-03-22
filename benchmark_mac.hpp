#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// macOS-focused helpers: ONNX file size, process RSS, CPU% from getrusage.

/** @return file size in bytes, or -1 on error */
int64_t getFileSizeBytes(const std::string& path);

/** Resident / physical footprint for current process (bytes). 0 if unavailable. */
std::size_t getProcessResidentBytes();

unsigned getLogicalCpuCount();

/**
 * Samples process CPU time vs wall time between pollCpuPercent() calls.
 * Return value can exceed 100 on multi-core (same idea as Activity Monitor % CPU).
 */
class BenchmarkSampler {
public:
    void reset();
    double pollCpuPercent();

private:
    long utime_sec_ = 0;
    long utime_usec_ = 0;
    long stime_sec_ = 0;
    long stime_usec_ = 0;
    std::int64_t last_wall_ns_ = 0;
    bool initialized_ = false;
};

/** Ring buffer of inference times (ms) for avg / percentiles */
class RollingLatencyMs {
public:
    explicit RollingLatencyMs(std::size_t capacity = 120);

    void push(double ms_ms);
    bool empty() const { return count_ == 0; }
    std::size_t size() const { return count_; }

    double mean() const;
    /** p in [0, 100], e.g. 50 = median, 95 = p95 */
    double percentile(double p) const;

private:
    std::vector<double> buf_;
    std::size_t head_ = 0;
    std::size_t count_ = 0;
};
