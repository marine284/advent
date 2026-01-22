#pragma once
#include <cstdint>
#include <src/defs.hpp>

namespace bench {

#if defined(_MSC_VER)

INLINE uint64_t
start() {
    _ReadWriteBarrier();
    _mm_lfence();
    _ReadWriteBarrier();
    uint64_t tsc = __rdtsc();
    _ReadWriteBarrier();
    _mm_lfence();
    _ReadWriteBarrier();
    return tsc;
}

INLINE uint64_t
stop() {
    _ReadWriteBarrier();
    unsigned aux;
    uint64_t tsc = __rdtscp(&aux);
    _ReadWriteBarrier();
    _mm_lfence();
    _ReadWriteBarrier();
    return tsc;
}

#else // GCC/Clang

INLINE uint64_t
start() {
    uint32_t low, high;
    asm volatile(
        "lfence\n\t"
        "rdtsc\n\t"
        "lfence"
        : "=a"(low), "=d"(high)
        :
        : "memory");
    return (uint64_t(high) << 32) | low;
}

INLINE uint64_t
stop() {
    uint32_t low, high;
    asm volatile(
        "rdtscp\n\t"
        "lfence"
        : "=a"(low), "=d"(high)
        :
        : "rcx", "memory");
    return (uint64_t(high) << 32) | low;
}

#endif

template <typename T>
INLINE void
DoNotOptimizeAway(T const &value) {
#if defined(_MSC_VER)
    volatile T *ptr = const_cast<volatile T *>(&value);
    (void)*ptr;
    _ReadWriteBarrier();
#else
    asm volatile("" : : "g"(value) : "memory");
#endif
}

INLINE void
ClobberMemory() {
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#else
    asm volatile("" : : : "memory");
#endif
}

} // namespace bench
