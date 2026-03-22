#include "benchmark_mac.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <mach/task_info.h>
#include <sys/sysctl.h>
#endif

#include <sys/resource.h>
#include <sys/stat.h>

int64_t getFileSizeBytes(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return -1;
    }
    return static_cast<int64_t>(st.st_size);
}

std::size_t getProcessResidentBytes() {
#if defined(__APPLE__) && defined(__MACH__)
    struct task_vm_info vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO,
                                 reinterpret_cast<task_info_t>(&vm_info), &count);
    if (kr == KERN_SUCCESS) {
        return static_cast<std::size_t>(vm_info.phys_footprint);
    }
    struct task_basic_info_64 ti;
    count = TASK_BASIC_INFO_64_COUNT;
    kr = task_info(mach_task_self(), TASK_BASIC_INFO_64,
                   reinterpret_cast<task_info_t>(&ti), &count);
    if (kr == KERN_SUCCESS) {
        return static_cast<std::size_t>(ti.resident_size);
    }
#endif
    return 0;
}

unsigned getLogicalCpuCount() {
#if defined(__APPLE__) && defined(__MACH__)
    int ncpu = 0;
    std::size_t len = sizeof(ncpu);
    if (sysctlbyname("hw.logicalcpu", &ncpu, &len, nullptr, 0) == 0 && ncpu > 0) {
        return static_cast<unsigned>(ncpu);
    }
#endif
    return 1;
}

void BenchmarkSampler::reset() {
    struct rusage ru;
    std::memset(&ru, 0, sizeof(ru));
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        utime_sec_ = utime_usec_ = stime_sec_ = stime_usec_ = 0;
    } else {
        utime_sec_ = ru.ru_utime.tv_sec;
        utime_usec_ = ru.ru_utime.tv_usec;
        stime_sec_ = ru.ru_stime.tv_sec;
        stime_usec_ = ru.ru_stime.tv_usec;
    }
    last_wall_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    initialized_ = true;
}

double BenchmarkSampler::pollCpuPercent() {
    struct rusage ru;
    std::memset(&ru, 0, sizeof(ru));
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return 0.0;
    }

    const long nu_sec = ru.ru_utime.tv_sec;
    const long nu_usec = ru.ru_utime.tv_usec;
    const long ns_sec = ru.ru_stime.tv_sec;
    const long ns_usec = ru.ru_stime.tv_usec;

    const std::int64_t wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count();

    if (!initialized_) {
        utime_sec_ = nu_sec;
        utime_usec_ = nu_usec;
        stime_sec_ = ns_sec;
        stime_usec_ = ns_usec;
        last_wall_ns_ = wall_ns;
        initialized_ = true;
        return 0.0;
    }

    const double prev_cpu_s =
        static_cast<double>(utime_sec_) + static_cast<double>(utime_usec_) * 1e-6 +
        static_cast<double>(stime_sec_) + static_cast<double>(stime_usec_) * 1e-6;
    const double now_cpu_s =
        static_cast<double>(nu_sec) + static_cast<double>(nu_usec) * 1e-6 +
        static_cast<double>(ns_sec) + static_cast<double>(ns_usec) * 1e-6;
    const double wall_s = static_cast<double>(wall_ns - last_wall_ns_) * 1e-9;

    utime_sec_ = nu_sec;
    utime_usec_ = nu_usec;
    stime_sec_ = ns_sec;
    stime_usec_ = ns_usec;
    last_wall_ns_ = wall_ns;

    if (wall_s <= 1e-9) {
        return 0.0;
    }
    return (now_cpu_s - prev_cpu_s) / wall_s * 100.0;
}

RollingLatencyMs::RollingLatencyMs(std::size_t capacity) : buf_(capacity, 0.0) {}

void RollingLatencyMs::push(double ms_ms) {
    if (buf_.empty()) {
        return;
    }
    buf_[head_] = ms_ms;
    head_ = (head_ + 1) % buf_.size();
    if (count_ < buf_.size()) {
        ++count_;
    }
}

double RollingLatencyMs::mean() const {
    if (count_ == 0) {
        return 0.0;
    }
    double sum = 0.0;
    const std::size_t cap = buf_.size();
    if (count_ < cap) {
        for (std::size_t i = 0; i < count_; ++i) {
            sum += buf_[i];
        }
        return sum / static_cast<double>(count_);
    }
    for (double v : buf_) {
        sum += v;
    }
    return sum / static_cast<double>(cap);
}

double RollingLatencyMs::percentile(double p) const {
    if (count_ == 0 || p < 0.0 || p > 100.0) {
        return 0.0;
    }
    std::vector<double> tmp;
    const std::size_t cap = buf_.size();
    if (count_ < cap) {
        tmp.assign(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(count_));
    } else {
        tmp = buf_;
    }
    std::sort(tmp.begin(), tmp.end());
    const double idx = (p / 100.0) * (static_cast<double>(tmp.size() - 1));
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) {
        return tmp[lo];
    }
    const double t = idx - static_cast<double>(lo);
    return tmp[lo] * (1.0 - t) + tmp[hi] * t;
}
