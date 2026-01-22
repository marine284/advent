#include "src/defs.hpp"

#include "src/bench.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring> // memcpy
#include <immintrin.h>

#include <utility> // day10

#define MCA_BEGIN(x) __asm__ volatile("# LLVM-MCA-BEGIN " #x)
#define MCA_END(x) __asm__ volatile("# LLVM-MCA-END " #x)

struct result {
    i64 P1, P2;
};

namespace _1 {

alignas(4096) static const u16 digit_lut[UINT16_MAX + 1] = {
#include "digit_lut.inc"
};

static const __m256i v_newline = _mm256_set1_epi8('\n');
static const __m256i v_R = _mm256_set1_epi16('R');
static const __m256i v_zero = _mm256_setzero_si256();
static const __m256i v_100 = _mm256_set1_epi16(100);
static const __m256i v_div100_mul = _mm256_set1_epi16(2622);
static const __m256i v_bcast_mask = _mm256_set_epi8(15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, //
                                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
static const __m256i v_w15_bcast = _mm256_set_epi8(15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, //
                                                   15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14);

#define PARSE_BATCH(BATCH_START, C0, C1, C2, C3, LRS, PARSED)                     \
    do {                                                                          \
        u32 m0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(C0, v_newline));          \
        u32 m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(C1, v_newline));          \
        const u8 *_lp = BATCH_START;                                              \
        int i = 0;                                                                \
        u64 mask = (u64)m0 | ((u64)m1 << 32);                                     \
        /* Extract from first 64 bytes */                                         \
        while (mask && i < 16) {                                                  \
            int tz = __builtin_ctzll(mask);                                       \
            LRS[i] = *_lp;                                                        \
            PARSED[i] = digit_lut[*(const u16 *)(BATCH_START + tz - 2)];          \
            _lp = BATCH_START + tz + 1;                                           \
            mask = _blsr_u64(mask);                                               \
            i++;                                                                  \
        }                                                                         \
        /* Extract from second 64 bytes if needed */                              \
        if (i < 16) {                                                             \
            u32 m2 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(C2, v_newline));      \
            u32 m3 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(C3, v_newline));      \
            mask = (u64)m2 | ((u64)m3 << 32);                                     \
            do {                                                                  \
                int tz = __builtin_ctzll(mask);                                   \
                LRS[i] = *_lp;                                                    \
                PARSED[i] = digit_lut[*(const u16 *)(BATCH_START + 64 + tz - 2)]; \
                _lp = BATCH_START + 64 + tz + 1;                                  \
                mask = _blsr_u64(mask);                                           \
                i++;                                                              \
            } while (mask && i < 16);                                             \
        }                                                                         \
        BATCH_START = _lp;                                                        \
    } while (0)

// L/R logic: R -> +n, L -> -(100-n)
#define CALC_LR(V_LRS, V_N)                              \
    do {                                                 \
        __m256i v_is_r = _mm256_cmpeq_epi16(V_LRS, v_R); \
        __m256i v_neg = _mm256_sub_epi16(v_100, V_N);    \
        V_N = _mm256_blendv_epi8(v_neg, V_N, v_is_r);    \
    } while (0)

// Prefix sum using Kogge-Stone algorithm
// Intra-lane only; cross-lane handled separately
#define CALC_PREFIX(V_N)                                                      \
    do {                                                                      \
        V_N = _mm256_add_epi16(V_N, _mm256_bslli_epi128(V_N, 2));             \
        V_N = _mm256_add_epi16(V_N, _mm256_bslli_epi128(V_N, 4));             \
        V_N = _mm256_add_epi16(V_N, _mm256_bslli_epi128(V_N, 8));             \
        __m256i v_hi = _mm256_permute2x128_si256(V_N, V_N, 0x00);             \
        V_N = _mm256_add_epi16(V_N, _mm256_shuffle_epi8(v_hi, v_bcast_mask)); \
    } while (0)

// Count positions divisible by 100, update offset for next batch
#define CALC_COUNT(V_N, CNT, V_OFF)                                                \
    do {                                                                           \
        __m256i v_q = _mm256_srli_epi16(_mm256_mulhi_epu16(V_N, v_div100_mul), 2); \
        __m256i v_q100 = _mm256_mullo_epi16(v_q, v_100);                           \
        __m256i v_rem = _mm256_sub_epi16(V_N, v_q100);                             \
        u32 zero_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi16(v_rem, v_zero));   \
        CNT += __builtin_popcount(zero_mask) >> 1;                                 \
        /* vperm2i128 $0x11 (3c) vs vpermq $0xFF (6c): force it; clang             \
           reassociates to vpshufb+vpermq otherwise, worse latency on the          \
           loop-carried dep. Mask is {15,14,...}: word15 sits at bytes 14-15 of    \
           the high lane.   */                                                     \
                                                                                   \
        /*__m256i _off;                                                            \
         __asm__(                                                                  \
             "vperm2i128 $0x11, %1, %1, %0\n\t"                                    \
             "vpshufb %2, %0, %0\n\t"                                              \
             : "=v"(_off)                                                          \
             : "v"(v_rem), "v"(v_w15_bcast));                                      \
         V_OFF = _off; */                                                          \
                                                                                   \
        /* no domain crossing! */                                                  \
        __m256i tmp = _mm256_permute4x64_epi64(v_rem, 0xFF); /* bcast qword3 */    \
        V_OFF = _mm256_shuffle_epi8(tmp, v_w15_bcast);       /* bcast word15 */    \
    } while (0)

#define LOAD_0_TO_64(PTR, C0, C1)                             \
    do {                                                      \
        C0 = _mm256_loadu_si256((const __m256i *)(PTR + 0));  \
        C1 = _mm256_loadu_si256((const __m256i *)(PTR + 32)); \
    } while (0)

#define LOAD_64_TO_128(PTR, C0, C1)                           \
    do {                                                      \
        C0 = _mm256_loadu_si256((const __m256i *)(PTR + 64)); \
        C1 = _mm256_loadu_si256((const __m256i *)(PTR + 96)); \
    } while (0)

#define LOAD_128(PTR, C0, C1, C2, C3)                         \
    do {                                                      \
        C0 = _mm256_loadu_si256((const __m256i *)(PTR + 0));  \
        C1 = _mm256_loadu_si256((const __m256i *)(PTR + 32)); \
        C2 = _mm256_loadu_si256((const __m256i *)(PTR + 64)); \
        C3 = _mm256_loadu_si256((const __m256i *)(PTR + 96)); \
    } while (0)

result
day() {
    static const u8 input[] = {
#embed "1.txt"
    };

    const u8 *__restrict__ ptr = (const u8 *)input;
    const u8 *end = ptr + sizeof(input);

    i32 cnt = 0;
    __m256i v_off = _mm256_set1_epi16(50);

    __m256i c0_0, c1_0, c2_0, c3_0;
    __m256i c0_1, c1_1, c2_1, c3_1;

    // BATCH 0: Only preload the first half to stay under 16 YMM registers
    LOAD_0_TO_64(ptr, c0_0, c1_0);

    // Main loop: process two batches (32 lines) per iteration
    while (ptr + 256 <= end) [[likely]] {
        alignas(16) u8 lrs0[16], lrs1[16];
        alignas(32) u16 parsed0[16], parsed1[16];

        // BATCH 0: Load second half
        LOAD_64_TO_128(ptr, c2_0, c3_0);
        PARSE_BATCH(ptr, c0_0, c1_0, c2_0, c3_0, lrs0, parsed0);

        // BATCH 1: Load and extract
        LOAD_128(ptr, c0_1, c1_1, c2_1, c3_1);
        PARSE_BATCH(ptr, c0_1, c1_1, c2_1, c3_1, lrs1, parsed1);

        // Load both batches into vectors
        __m256i v_lrs0 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i *)lrs0));
        __m256i v_n0 = _mm256_load_si256((const __m256i *)parsed0);
        __m256i v_lrs1 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i *)lrs1));
        __m256i v_n1 = _mm256_load_si256((const __m256i *)parsed1);

        // BATCH 0: Preload the first half
        LOAD_0_TO_64(ptr, c0_0, c1_0);

        // Process both batches
        CALC_LR(v_lrs0, v_n0);
        CALC_LR(v_lrs1, v_n1);

        CALC_PREFIX(v_n0);
        v_n0 = _mm256_add_epi16(v_n0, v_off);
        CALC_COUNT(v_n0, cnt, v_off);

        CALC_PREFIX(v_n1);
        v_n1 = _mm256_add_epi16(v_n1, v_off);
        CALC_COUNT(v_n1, cnt, v_off);
    }

    // Tail*
    while (ptr + 128 <= end) {
        LOAD_128(ptr, c0_0, c1_0, c2_0, c3_0);

        alignas(16) u8 lrs[16];
        alignas(32) u16 parsed[16];

        PARSE_BATCH(ptr, c0_0, c1_0, c2_0, c3_0, lrs, parsed);

        __m256i v_lrs = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i *)lrs));
        __m256i v_n = _mm256_load_si256((const __m256i *)parsed);

        CALC_LR(v_lrs, v_n);
        CALC_PREFIX(v_n);
        v_n = _mm256_add_epi16(v_n, v_off);
        CALC_COUNT(v_n, cnt, v_off);
    }

    return { (i64)cnt, 0 };
}

} // namespace _1

namespace _2 {

// Magic numbers for division by (d + 1)
constexpr u64
compute_m(u64 d) {
    return ((__uint128_t)1 << 64) / d + 1;
}

// 64-bit multiply-high for fast division
INLINE u64
fastdiv(u64 n, u64 m) {
    return ((__uint128_t)n * m) >> 64;
}

struct DivData {
    u64 divisor_minus_one;
    u64 magic_multiplier;
};

result
day() {
    static const u8 input[] = {
#embed "2.txt"
    };

    // clang-format off

    // Tables only need to cover up to length 10
    // Index corresponds to length (e.g., DIV[4] handles 4-digit numbers)
    static DivData DIV[] = {
        {0, 0}, {0, 0},                                      // 0, 1
        {10, compute_m(11)}, {0, 0},                         // 2, 3
        {100, compute_m(101)}, {0, 0},                       // 4, 5
        {1000, compute_m(1001)}, {0, 0},                     // 6, 7
        {10000, compute_m(10001)}, {0, 0},                   // 8, 9
        {100000, compute_m(100001)}
    };

    // Next smallest valid number for odd lengths (bumps up to next even length)
    // e.g., NEXT[3] (range 100-999) -> 1010 (Start of len 4)
    static u64 NEXT[] = {
        0, 11,
        0, 1010,
        0, 100100,
        0, 10001000,
        0, 1000010000ULL,
        0,
    };

    // Prev largest valid number for odd lengths (bumps down to prev even length)
    // e.g., PREV[3] (range 100-999) -> 99 (End of len 2)
    static u64 PREV[] = {
        0, 0, 0, 99,
        0, 9999,
        0, 999999,
        0, 99999999,
        0,
    };

    // clang-format on

    const u8 *ptr = input;
    const u8 *end = ptr + sizeof(input);

    i64 tot = 0;

    while (ptr < end) {
        // Parse start (fused value + length)
        const u8 *s_start = ptr;
        u64 s = *ptr - '0';
        while (*++ptr >= '0') {
            s = s * 10 + (*ptr - '0');
        }
        int slen = ptr - s_start;

        ptr++; // skip '-'

        // Parse end (fused value + length)
        const u8 *e_start = ptr;
        u64 e = *ptr - '0';
        while (*++ptr >= '0') {
            e = e * 10 + (*ptr - '0');
        }
        int elen = ptr - e_start;

        ptr++; // skip ',' - oob read on the very last num but that's okay

        // Adjust odd lengths
        if (slen % 2 != 0) {
            s = NEXT[slen];
            slen++;
        }
        if (elen % 2 != 0) {
            e = PREV[elen];
            elen--;
        }

        if (slen > elen) {
            continue;
        }

        // Constant-time arithmetic sum
        const DivData &dd = DIV[slen];

        // fastdiv(s + d, m) is effectively ceil(s / divisor) because we stored d =
        // divisor - 1 If first_mul > last_mul, then n becomes 0 thanks to unsigned
        // overflow: (last_mul - first-mul + 1) -> (small - big + 1) -> 0
        u64 first_mul = fastdiv(s + dd.divisor_minus_one, dd.magic_multiplier);
        u64 last_mul = fastdiv(e, dd.magic_multiplier);

        u64 step_size = dd.divisor_minus_one + 1;
        u64 term_count = last_mul - first_mul + 1; // Number of terms

        // Sum of arithmetic progression
        u64 term = 2 * first_mul + term_count - 1;
        tot += (step_size * term_count * term) / 2;
    }

    return { tot, 0 };
}

} // namespace _2

namespace _3 {

static const __m256i v_digit[10] = {
    _mm256_set1_epi8('0'), _mm256_set1_epi8('1'), _mm256_set1_epi8('2'), _mm256_set1_epi8('3'), _mm256_set1_epi8('4'), _mm256_set1_epi8('5'), _mm256_set1_epi8('6'), _mm256_set1_epi8('7'), _mm256_set1_epi8('8'), _mm256_set1_epi8('9'),
};

result
day() {
    static const u8 input[] = {
#embed "3.txt"
    };

    const u8 *ptr = input;
    const u8 *end = ptr + sizeof(input);

    i64 tot = 0;

    while (ptr + 100 <= end) {
        // Load the 64 bytes at ptr+0 as two 32-byte vectors
        __m256i b1_0 = _mm256_loadu_si256((const __m256i *)(ptr));
        __m256i b1_1 = _mm256_loadu_si256((const __m256i *)(ptr + 32));

        // Load the 64 bytes at ptr+36 as two 32-byte vectors
        __m256i b2_0 = _mm256_loadu_si256((const __m256i *)(ptr + 36));
        __m256i b2_1 = _mm256_loadu_si256((const __m256i *)(ptr + 68));

        u64 mask1 = ~0ULL;
        // mask2 excludes bits 0..27 (the overlap) and bit 63 (the 100th char/last)
        u64 mask2 = (~0ULL << 28) ^ (1ULL << 63);

        // The compiler pulled a 16-byte vector load just to grab ptr[99] (lol).
        // Force it to issue a simple scalar movzbl: u8 last = ptr[99];
        u8 last = *(volatile const u8 *)(ptr + 99);
        u64 acc = 0;

        for (int i = 9; i >= 0; --i) {
            __m256i vi = v_digit[i];
            //__m256i vi = _mm256_set1_epi8(i + '0');

            u32 m1_0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b1_0, vi));
            u32 m1_1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b1_1, vi));
            u64 m1 = ((u64)m1_1 << 32) | m1_0;
            m1 &= mask1;

            u32 m2_0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b2_0, vi));
            u32 m2_1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b2_1, vi));
            u64 m2 = ((u64)m2_1 << 32) | m2_0;
            m2 &= mask2;

            int c1 = __builtin_popcountll(m1);
            int c2 = __builtin_popcountll(m2);

            if (c1 + c2 >= 2) {
                acc = std::max(acc, (u64)i);
                u64 l = std::max((u64)(last - '0'), (u64)i);
                tot += 10 * acc + l;
                goto inner_break;
            }

            if (c1 + c2 == 1) {
                if (acc != 0) {
                    tot += 10 * acc + i;
                    goto inner_break;
                }

                if (last >= i + '0') {
                    tot += 10 * i + (last - '0');
                    goto inner_break;
                }

                acc = i;

                if (c1 == 1) {
                    mask1 = ~0ULL << __builtin_ctzll(m1);
                    mask2 = ~0ULL << 28;
                } else {
                    mask1 = 0;
                    mask2 = ~0ULL << __builtin_ctzll(m2);
                }
            }
        }

    inner_break:
        // 100 digits + 1 newline
        ptr += 101;
    }

    return { tot, 0 };
}

} // namespace _3

namespace _4 {

struct u8x64 {
    __m256i lo, hi;

    INLINE u8x64
    operator+(const u8x64 &o) const {
        return { _mm256_add_epi8(lo, o.lo), _mm256_add_epi8(hi, o.hi) };
    }

    INLINE u8x64
    operator&(const u8x64 &o) const {
        return { _mm256_and_si256(lo, o.lo), _mm256_and_si256(hi, o.hi) };
    }
};

static INLINE u8x64
u8x64_load(const u8 *p) {
    return { _mm256_loadu_si256((const __m256i *)p), _mm256_loadu_si256((const __m256i *)(p + 32)) };
}

static INLINE u8x64
u8x64_splat(u8 v) {
    return { _mm256_set1_epi8(v), _mm256_set1_epi8(v) };
}

static INLINE u64
simd_eq_zero_mask(const u8x64 &a) {
    __m256i z = _mm256_setzero_si256();
    u32 mlo = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a.lo, z));
    u32 mhi = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a.hi, z));
    return ((u64)mhi << 32) | mlo;
}

static INLINE u64
simd_ge_mask(const u8x64 &a, u8 threshold) {
    // _mm256_cmpgt_epi8 is signed, but max sum is ~18 so it naturally won't
    // overflow
    __m256i t = _mm256_set1_epi8(threshold - 1);
    u32 mlo = (u32)_mm256_movemask_epi8(_mm256_cmpgt_epi8(a.lo, t));
    u32 mhi = (u32)_mm256_movemask_epi8(_mm256_cmpgt_epi8(a.hi, t));
    return ((u64)mhi << 32) | mlo;
}

// scalar flush since it runs only once
static INLINE u8x64
shift_right_1(const u8x64 &v, u8 fill) {
    alignas(32) u8 buf[64];
    _mm256_store_si256((__m256i *)buf, v.lo);
    _mm256_store_si256((__m256i *)(buf + 32), v.hi);
    for (int i = 63; i >= 1; i--)
        buf[i] = buf[i - 1];
    buf[0] = fill;
    return u8x64_load(buf);
}

static INLINE u8x64
shift_right_2(const u8x64 &v, u8 fill) {
    alignas(32) u8 buf[64];
    _mm256_store_si256((__m256i *)buf, v.lo);
    _mm256_store_si256((__m256i *)(buf + 32), v.hi);
    for (int i = 63; i >= 2; i--)
        buf[i] = buf[i - 2];
    buf[0] = fill;
    buf[1] = fill;
    return u8x64_load(buf);
}

result
day() {
    static const u8 input[] = {
#embed "4.txt"
    };

    const u8 *ptr = input;
    const u8 *end = ptr + sizeof(input);

    i64 tot = 0;

    __m256i v_nl = _mm256_loadu_si256((const __m256i *)(ptr + 128));
    __m256i nl = _mm256_set1_epi8('\n');
    u32 mask_nl = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v_nl, nl));
    size_t len = 128 + __builtin_ctz(mask_nl);

    u64 mask1 = (~0ULL << 2) >> 1; // 0x7FFFFFFFFFFFFFFE
    u64 mask2 = mask1;
    u64 mask3 = ~0ULL << (64 - (len - 123));

    u8 bit = 0b10;
    u8x64 mask_bit = u8x64_splat(bit);
    u8x64 outside_lane = u8x64_splat(3 * bit);

    u8x64 lane11 = outside_lane;
    u8x64 lane12 = outside_lane;
    u8x64 lane13 = outside_lane;

    u8x64 base21 = u8x64_load(ptr) & mask_bit;
    u8x64 lane21c = shift_right_1(base21, bit);
    u8x64 lane21r = shift_right_2(base21, bit);
    u8x64 lane21l = base21;
    u8x64 lane21 = lane21c + lane21r + lane21l;

    u8x64 lane22c = u8x64_load(ptr + 61) & mask_bit;
    u8x64 lane22r = u8x64_load(ptr + 60) & mask_bit;
    u8x64 lane22l = u8x64_load(ptr + 62) & mask_bit;
    u8x64 lane22 = lane22c + lane22r + lane22l;

    u8x64 lane23c = u8x64_load(ptr + len + 1 - 64) & mask_bit;
    u8x64 lane23r = u8x64_load(ptr + len + 1 - 64 - 1) & mask_bit;
    u8x64 lane23l = u8x64_load(ptr + len + 1 - 64 + 1) & mask_bit;
    u8x64 lane23 = lane23c + lane23r + lane23l;

    ptr += len + 1;

    auto count = [&](const u8x64 &l31, const u8x64 &l32,
                     const u8x64 &l33) { // inline pls
        u8x64 sum1 = lane11 + lane21 + l31;
        u8x64 sum2 = lane12 + lane22 + l32;
        u8x64 sum3 = lane13 + lane23 + l33;

        tot += __builtin_popcountll(simd_eq_zero_mask(lane21c) & simd_ge_mask(sum1, 10) & mask1);
        tot += __builtin_popcountll(simd_eq_zero_mask(lane22c) & simd_ge_mask(sum2, 10) & mask2);
        tot += __builtin_popcountll(simd_eq_zero_mask(lane23c) & simd_ge_mask(sum3, 10) & mask3);
    };

    while (true) {
        u8x64 lane31c = u8x64_load(ptr - 1) & mask_bit;
        u8x64 lane31r = u8x64_load(ptr - 2) & mask_bit;
        u8x64 lane31l = u8x64_load(ptr) & mask_bit;
        u8x64 lane31 = lane31c + lane31r + lane31l;

        u8x64 lane32c = u8x64_load(ptr + 61) & mask_bit;
        u8x64 lane32r = u8x64_load(ptr + 60) & mask_bit;
        u8x64 lane32l = u8x64_load(ptr + 62) & mask_bit;
        u8x64 lane32 = lane32c + lane32r + lane32l;

        u8x64 lane33c = u8x64_load(ptr + len + 1 - 64) & mask_bit;
        u8x64 lane33r = u8x64_load(ptr + len + 1 - 64 - 1) & mask_bit;
        u8x64 lane33l = u8x64_load(ptr + len + 1 - 64 + 1) & mask_bit;
        u8x64 lane33 = lane33c + lane33r + lane33l;

        count(lane31, lane32, lane33);

        lane11 = lane21;
        lane12 = lane22;
        lane13 = lane23;
        lane21 = lane31;
        lane22 = lane32;
        lane23 = lane33;
        lane21c = lane31c;
        lane22c = lane32c;
        lane23c = lane33c;

        ptr += len + 1;
        if (ptr >= end) {
            break;
        }
    }

    u8x64 lane31 = outside_lane;
    u8x64 lane32 = outside_lane;
    u8x64 lane33 = outside_lane;
    count(lane31, lane32, lane33);

    return { tot, 0 };
}

} // namespace _4

// namespace _9 {
// // Offset calculation: '0' * sum_of_place_values
// //   4-digit:  48 * (1000+100+10+1)   = 53328
// //   5-digit:  48 * (10000+...+10+1)  = 533328
// static INLINE u32
// parse_n5(const u8 *&ptr, u8 sep) {
//     u32 s0 = ptr[0];
//     u32 s1 = ptr[1];
//     u32 s2 = ptr[2];
//     u32 s3 = ptr[3];
//     u32 s4 = ptr[4];

//     if (ptr[4] == sep) {
//         ptr += 5;
//         return (s0 * 1000 + s1 * 100 + s2 * 10 + s3) - 53328;
//     }
//     ptr += 6;
//     return (s0 * 10000 + s1 * 1000 + s2 * 100 + s3 * 10 + s4) - 533328;
// }

// // AVX2-compatible 64-bit max (znver3 lacks AVX-512)
// static INLINE __m256i
// mm256_max_epi64_avx2(__m256i a, __m256i b) {
//     __m256i cmp = _mm256_cmpgt_epi64(a, b);
//     return _mm256_blendv_epi8(b, a, cmp);
// }

// // O(n^2)
// result
// day() {
//     MCA_BEGIN(day09);
//     static const u8 input[] = {
// #embed "9.txt"
//     };

//     const u8 *ptr = input;
//     const u8 *end = ptr + sizeof(input);

//     // 1. PHASE ONE: Parse everything upfront
//     alignas(32) i32 xs[512];
//     alignas(32) i32 ys[512];
//     i32 n = 0;

//     while (ptr != end) [[likely]] {
//         xs[n] = parse_n5(ptr, ',');
//         ys[n] = parse_n5(ptr, '\n');
//         n++;
//     }

//     // Pad the end of the arrays with the last element.
//     // This ensures our SIMD loop can safely round up and read past `n`
//     // without hitting uninitialized memory (computing against itself yields
//     area=1). _mm256_storeu_si256((__m256i *)&xs[n], _mm256_set1_epi32(xs[n -
//     1])); _mm256_storeu_si256((__m256i *)&ys[n], _mm256_set1_epi32(ys[n -
//     1]));

//     // 2. PHASE TWO: Compute areas
//     __m256i max_vec = _mm256_setzero_si256();
//     const __m256i one = _mm256_set1_epi32(1);

//     i32 i = 1;
//     // Process TWO new points at a time.
//     for (; i + 1 < n; i += 2) {
//         __m256i xv0 = _mm256_set1_epi32(xs[i]), yv0 =
//         _mm256_set1_epi32(ys[i]);
//         __m256i xv1 = _mm256_set1_epi32(xs[i + 1]), yv1 =
//         _mm256_set1_epi32(ys[i + 1]);

//         __m256i m0 = _mm256_setzero_si256();
//         __m256i m1 = _mm256_setzero_si256();

//         // r seamlessly covers the rounded upper-bound for both i and i+1
//         const i32 r = (i + 8) & ~7;
//         for (i32 j = 0; j < r; j += 8) {
//             __m256i x2 = _mm256_load_si256((const __m256i *)&xs[j]);
//             __m256i y2 = _mm256_load_si256((const __m256i *)&ys[j]);

//             // --- Point 0 ---
//             __m256i dx0 = _mm256_abs_epi32(_mm256_sub_epi32(xv0, x2));
//             __m256i dy0 = _mm256_abs_epi32(_mm256_sub_epi32(yv0, y2));
//             __m256i dx0p = _mm256_add_epi32(dx0, one);
//             __m256i dy0p = _mm256_add_epi32(dy0, one);

//             __m256i ae0 = _mm256_mul_epu32(dx0p, dy0p);
//             __m256i ao0 =
//                 _mm256_mul_epu32(_mm256_srli_epi64(dx0p, 32),
//                 _mm256_srli_epi64(dy0p, 32));

//             // Shortened reduction tree
//             m0 = mm256_max_epi64_avx2(m0, mm256_max_epi64_avx2(ae0, ao0));

//             // --- Point 1 ---
//             __m256i dx1 = _mm256_abs_epi32(_mm256_sub_epi32(xv1, x2));
//             __m256i dy1 = _mm256_abs_epi32(_mm256_sub_epi32(yv1, y2));
//             __m256i dx1p = _mm256_add_epi32(dx1, one);
//             __m256i dy1p = _mm256_add_epi32(dy1, one);

//             __m256i ae1 = _mm256_mul_epu32(dx1p, dy1p);
//             __m256i ao1 =
//                 _mm256_mul_epu32(_mm256_srli_epi64(dx1p, 32),
//                 _mm256_srli_epi64(dy1p, 32));

//             m1 = mm256_max_epi64_avx2(m1, mm256_max_epi64_avx2(ae1, ao1));
//         }
//         // Accumulate both independent chains back into the global max
//         max_vec = mm256_max_epi64_avx2(max_vec, mm256_max_epi64_avx2(m0,
//         m1));
//     }

//     // Tail handling for odd `n` amounts
//     for (; i < n; i++) {
//         __m256i xv = _mm256_set1_epi32(xs[i]), yv = _mm256_set1_epi32(ys[i]);
//         __m256i m0 = _mm256_setzero_si256();

//         const i32 r = (i + 7) & ~7;
//         for (i32 j = 0; j < r; j += 8) {
//             __m256i x2 = _mm256_load_si256((const __m256i *)&xs[j]);
//             __m256i y2 = _mm256_load_si256((const __m256i *)&ys[j]);

//             __m256i dx = _mm256_abs_epi32(_mm256_sub_epi32(xv, x2));
//             __m256i dy = _mm256_abs_epi32(_mm256_sub_epi32(yv, y2));
//             __m256i dxp = _mm256_add_epi32(dx, one);
//             __m256i dyp = _mm256_add_epi32(dy, one);

//             __m256i ae = _mm256_mul_epu32(dxp, dyp);
//             __m256i ao = _mm256_mul_epu32(_mm256_srli_epi64(dxp, 32),
//             _mm256_srli_epi64(dyp, 32)); m0 = mm256_max_epi64_avx2(m0,
//             mm256_max_epi64_avx2(ae, ao));
//         }
//         max_vec = mm256_max_epi64_avx2(max_vec, m0);
//     }

//     // Horizontal fold
//     __m128i lo = _mm256_castsi256_si128(max_vec);
//     __m128i hi = _mm256_extracti128_si256(max_vec, 1);
//     __m128i m = _mm_blendv_epi8(lo, hi, _mm_cmpgt_epi64(hi, lo));
//     __m128i m2 = _mm_unpackhi_epi64(m, m);
//     __m128i r_m = _mm_blendv_epi8(m, m2, _mm_cmpgt_epi64(m2, m));
//     i64 result = _mm_cvtsi128_si64(r_m);
//     MCA_END(day09);

//     return { result, 0 };
// }
// } // namespace _9

namespace _9 {
// Offset calculation: '0' * sum_of_place_values
//   4-digit:  48 * (1000+100+10+1)   = 53328
//   5-digit:  48 * (10000+...+10+1)  = 533328
static INLINE u32
parse_n5(const u8 *&ptr, u8 sep) {
    u32 s0 = ptr[0], s1 = ptr[1], s2 = ptr[2], s3 = ptr[3], s4 = ptr[4];
    if (s4 == sep) {
        ptr += 5;
        return (s0 * 1000 + s1 * 100 + s2 * 10 + s3) - 53328;
    }
    ptr += 6;
    return (s0 * 10000 + s1 * 1000 + s2 * 100 + s3 * 10 + s4) - 533328;
}

// AVX2-compatible 64-bit max
static INLINE __m256i
mm256_max_epi64(__m256i a, __m256i b) {
    __m256i cmp = _mm256_cmpgt_epi64(a, b);
    return _mm256_blendv_epi8(b, a, cmp);
}

// 128-bit counterpart for horizontal max
static INLINE __m128i
mm128_max_epi64(__m128i a, __m128i b) {
    __m128i cmp = _mm_cmpgt_epi64(a, b);
    return _mm_blendv_epi8(b, a, cmp);
}

// Pareto Front
// O(|BL|*|TR| + |TL|*|BR|) instead of O(n^2), which would be 125k comparisons
// Maximum rectangles always span from one extreme corner to its opposite.
result
day() {
    static const u8 input[] = {
#embed "9.txt"
    };
    const u8 *ptr = input;
    const u8 *end = ptr + sizeof(input);

    // 1. Parse & pack
    alignas(32) u64 pts[512];
    i32 n = 0;
    while (ptr != end) [[likely]] {
        u64 x = parse_n5(ptr, ',');
        u64 y = parse_n5(ptr, '\n');
        pts[n++] = (y << 32) | x;
    }

    // 2. Radix Sort by X coordinate (lower 32 bits)
    // 17-bit split: 8 bits + 9 bits for two-pass sort
    // Only sorts the lower 32 bits (which is our X coordinate)
    alignas(32) u64 tmp[512];
    u16 lo_c[256] = { 0 }, hi_c[392] = { 0 };

    // Count occurrences in each bucket
    for (i32 i = 0; i < n; i++) {
        u32 x = (u32)pts[i];
        lo_c[x & 0xFF]++;
        hi_c[x >> 8]++;
    }

    // Prefix sums to convert counts to positions
    u32 sum_lo = 0;
    for (i32 i = 0; i < 256; i++) {
        u32 count = lo_c[i];
        lo_c[i] = sum_lo;
        sum_lo += count;
    }

    u32 sum_hi = 0;
    for (i32 i = 0; i < 392; i++) {
        u32 count = hi_c[i];
        hi_c[i] = sum_hi;
        sum_hi += count;
    }

    // Scatters: pts -> tmp by low byte, tmp -> pts by high byte
    for (i32 i = 0; i < n; i++) {
        u32 x = (u32)pts[i];
        tmp[lo_c[x & 0xFF]++] = pts[i];
    }
    for (i32 i = 0; i < n; i++) {
        u32 x = (u32)tmp[i];
        pts[hi_c[x >> 8]++] = tmp[i];
    }

    // 3. Extract Frontiers Directly to SoA
    alignas(32) i32 bl_x[128], bl_y[128];
    alignas(32) i32 tl_x[128], tl_y[128];
    i32 n_bl = 0, n_tl = 0;
    i32 min_y_bl = INT32_MAX, max_y_tl = INT32_MIN;

    // Left-to-right scan: bottom-left (min Y) & top-left (max Y)
    for (i32 i = 0; i < n; i++) {
        i32 x = (i32)pts[i];
        i32 y = (i32)(pts[i] >> 32);
        if (y < min_y_bl) {
            bl_x[n_bl] = x;
            bl_y[n_bl] = y;
            n_bl++;
            min_y_bl = y;
        }
        if (y > max_y_tl) {
            tl_x[n_tl] = x;
            tl_y[n_tl] = y;
            n_tl++;
            max_y_tl = y;
        }
    }

    alignas(32) i32 tr_x[128], tr_y[128];
    alignas(32) i32 br_x[128], br_y[128];
    i32 n_tr = 0, n_br = 0;
    i32 min_y_br = INT32_MAX, max_y_tr = INT32_MIN;

    // Right-to-left scan: top-right (max Y) & bottom-right (min Y)
    for (i32 i = n - 1; i >= 0; i--) {
        i32 x = (i32)pts[i];
        i32 y = (i32)(pts[i] >> 32);
        if (y > max_y_tr) {
            tr_x[n_tr] = x;
            tr_y[n_tr] = y;
            n_tr++;
            max_y_tr = y;
        }
        if (y < min_y_br) {
            br_x[n_br] = x;
            br_y[n_br] = y;
            n_br++;
            min_y_br = y;
        }
    }

    // Pad inner looping sets for simd
    auto pad_soa = [](i32 orig_n, i32 &padded_n, i32 *xs, i32 *ys) {
        padded_n = (orig_n + 7) & ~7;
        for (i32 i = orig_n; i < padded_n; i++) {
            xs[i] = xs[orig_n - 1];
            ys[i] = ys[orig_n - 1];
        }
    };

    i32 p_tr, p_br;
    pad_soa(n_tr, p_tr, tr_x, tr_y);
    pad_soa(n_br, p_br, br_x, br_y);

    // 4. Comparisons
    // Split into even and odd to slice the dependency chain depth in half
    __m256i max_even = _mm256_setzero_si256();
    __m256i max_odd = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);

    // Compute 64-bit areas and update running maximum vector
    auto update_max_area = [&](__m256i x1, __m256i y1, i32 count, const i32 *xs, const i32 *ys) {
        for (i32 j = 0; j < count; j += 8) {
            __m256i x2 = _mm256_load_si256((const __m256i *)&xs[j]);
            __m256i y2 = _mm256_load_si256((const __m256i *)&ys[j]);
            __m256i dx = _mm256_abs_epi32(_mm256_sub_epi32(x1, x2));
            __m256i dy = _mm256_abs_epi32(_mm256_sub_epi32(y1, y2));
            __m256i dx_p1 = _mm256_add_epi32(dx, one);
            __m256i dy_p1 = _mm256_add_epi32(dy, one);

            // Avoid overflow (coords up to 99999 -> 10B area)
            __m256i area_e = _mm256_mul_epu32(dx_p1, dy_p1);
            __m256i dx_o = _mm256_srli_epi64(dx_p1, 32);
            __m256i dy_o = _mm256_srli_epi64(dy_p1, 32);
            __m256i area_o = _mm256_mul_epu32(dx_o, dy_o);

            max_even = mm256_max_epi64(max_even, area_e);
            max_odd = mm256_max_epi64(max_odd, area_o);
        }
    };

    // Compare Bottom-Left corners against Top-Right corners
    for (i32 i = 0; i < n_bl; i++) {
        __m256i x1 = _mm256_set1_epi32(bl_x[i]);
        __m256i y1 = _mm256_set1_epi32(bl_y[i]);
        update_max_area(x1, y1, p_tr, tr_x, tr_y);
    }

    // Compare Top-Left corners against Bottom-Right corners
    for (i32 i = 0; i < n_tl; i++) {
        __m256i x1 = _mm256_set1_epi32(tl_x[i]);
        __m256i y1 = _mm256_set1_epi32(tl_y[i]);
        update_max_area(x1, y1, p_br, br_x, br_y);
    }

    // Combine even/odd split vectors, and horizontally reduce
    __m256i max_vec = mm256_max_epi64(max_even, max_odd);

    __m128i lo_vec = _mm256_castsi256_si128(max_vec);
    __m128i hi_vec = _mm256_extracti128_si256(max_vec, 1);
    __m128i m = mm128_max_epi64(lo_vec, hi_vec);
    __m128i m2 = _mm_unpackhi_epi64(m, m);
    __m128i r = mm128_max_epi64(m, m2);

    return { _mm_cvtsi128_si64(r), 0 };
}

} // namespace _9

// Each input line: [light-state] (button0) (button1) ... {part2-ignored}
// lights and each button are bitmasks over up to 16 positions; pressing a
// button XORs its mask. Algorithm: enumerate all 2^N button subsets via a
// doubling pass.
//   queue[k] = lights ^ XOR(buttons[i] for each set bit i in k)
// Answer per line = min popcount(k) such that queue[k] == 0.
// N<=13 in this input, so max queue size is 8192 u16s = 16 KB.
namespace _10 {

static INLINE __m256i
popcount_epu16(__m256i v) {
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i lut = _mm256_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    __m256i lo = _mm256_and_si256(v, mask4);
    // srli_epi16 shifts the full 16-bit lane, bleeding bits from the high byte
    // into the low byte. For u16 0xABCD: shifted = 0x0ABC; bytes {0xBC, 0x0A};
    // after & 0x0F0F: {0x0C, 0x0A}. 0x0C is the high nibble of the original low
    // byte — correct by coincidence of the bleed.
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask4);
    __m256i pclo = _mm256_shuffle_epi8(lut, lo);
    __m256i pchi = _mm256_shuffle_epi8(lut, hi);
    __m256i sum = _mm256_add_epi8(pclo, pchi);
    return _mm256_maddubs_epi16(sum, _mm256_set1_epi8(1));
}

static INLINE u16
hmin_epu16(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m = _mm_min_epu16(lo, hi);
    m = _mm_min_epu16(m, _mm_srli_si128(m, 8));
    m = _mm_min_epu16(m, _mm_srli_si128(m, 4));
    m = _mm_min_epu16(m, _mm_srli_si128(m, 2));
    return (u16)_mm_extract_epi16(m, 0);
}

alignas(32) static u16 queue[1 << 16];

result
day() {
    alignas(32) static const u8 input[] = {
#embed "10.txt"
        , '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
    };

    const u8 *ptr = input;
    const u8 *end_ptr = input + sizeof(input) - 8;
    i64 tot = 0;

    while (true) {
        __m128i blk = _mm_loadu_si128((const __m128i *)(ptr + 1));
        u16 lights = (u16)_mm_movemask_epi8(_mm_cmpeq_epi8(blk, _mm_set1_epi8('#')));
        u32 bracket = (u32)_mm_movemask_epi8(_mm_cmpeq_epi8(blk, _mm_set1_epi8(']')));
        u32 blen = (u32)__builtin_ctz(bracket);
        ptr += 1 + blen + 1;
        ptr += 1;

        u16 buttons[16];
        int buttons_len = 16;
        for (int i = 0; i < 16; i++) {
            if (*ptr != '(') {
                buttons_len = i;
                break;
            }
            ptr++;
            u16 btn = 0;
            for (int k = 0; k < 10; k++) {
                if (*ptr == ' ') {
                    break;
                }
                btn |= (u16)(1u << (*ptr - '0'));
                ptr += 2;
            }
            buttons[i] = btn;
            ptr++;
        }

        queue[0] = lights;
        u32 steps = UINT32_MAX;
        __m256i steps_acc = _mm256_set1_epi16((i16)UINT16_MAX);

        const __m256i vff = _mm256_set1_epi16((i16)UINT16_MAX);
        const __m256i voff = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        // Lambda with template parameter for compile-time constant
        auto process = [&]<int I>() {
            if constexpr (I < 4) {
                for (int j = 0; j < (1 << I); j++) {
                    int k = (1 << I) + j;
                    u16 nl = queue[j] ^ buttons[I];
                    if (nl == 0) {
                        u32 pc = (u32)__builtin_popcount(k);
                        steps = steps < pc ? steps : pc;
                    }
                    queue[k] = nl;
                }
            } else {
                const int nblk = (1 << I) / 16;
                const __m256i vb = _mm256_set1_epi16((short)buttons[I]);
                for (int j = 0; j < nblk; j++) {
                    int k = (1 << I) + 16 * j;
                    __m256i vq = _mm256_loadu_si256((const __m256i *)(queue + 16 * j));
                    __m256i vnl = _mm256_xor_si256(vq, vb);
                    __m256i vmsk = _mm256_cmpeq_epi16(vnl, _mm256_setzero_si256());
                    if (_mm256_movemask_epi8(vmsk)) {
                        __m256i vidx = _mm256_add_epi16(_mm256_set1_epi16((short)k), voff);
                        __m256i vsel = _mm256_blendv_epi8(vff, vidx, vmsk);
                        steps_acc = _mm256_min_epu16(steps_acc, popcount_epu16(vsel));
                    }
                    _mm256_storeu_si256((__m256i *)(queue + k), vnl);
                }
            }
        };

        // Short-circuit fold: evaluates process<0>(), process<1>(), ... in order,
        // stopping as soon as buttons_len is reached.
        [&]<int... I>(std::integer_sequence<int, I...>) {
            (void)(... && (buttons_len > I && (process.template operator()<I>(), true)));
        }(std::make_integer_sequence<int, 16>{});

        tot += (i64)std::min(steps, (u32)hmin_epu16(steps_acc));

        if (ptr >= end_ptr - 40) {
            break;
        }
        const __m256i vnl = _mm256_set1_epi8('\n');
        __m256i a = _mm256_loadu_si256((const __m256i *)ptr);
        __m256i b = _mm256_loadu_si256((const __m256i *)(ptr + 32));
        u32 mlo = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, vnl));
        u32 mhi = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, vnl));
        u64 nl_mask = (u64)mlo | ((u64)mhi << 32);
        ptr += __builtin_ctzll(nl_mask) + 1;
    }

    return { tot, 0 };
}

} // namespace _10

namespace _11 {

// Each node name is 3 ASCII bytes.  Packed into a little-endian u32 (with a
// trailing space or colon), pext extracts bits [4:0] of bytes 0, 1, 2 into a
// contiguous 15-bit index.
static constexpr u32 NODE_MASK = (0b11111u << 0) | (0b11111u << 8) | (0b11111u << 16);
static INLINE u32
parse_node(u32 n) {
    return _pext_u32(n, NODE_MASK);
}

// `gen` before `sum` would save nothing; `sum` first keeps the hot field at
// offset 0 and lets the gen check emit a single fused load-compare:
//   cmpl %eax, 4(%rdi,%r14,8)   ; compare current_gen against CACHE[m].gen
// with no intermediate register, saving one uop on the critical path.
struct CacheEntry {
    i32 sum;
    u32 gen;
};
static CacheEntry CACHE[32 * 32 * 27];

// 16-byte Frame size is load-bearing: the compiler folds the stack pointer
// arithmetic into a single-cycle shift rather than aslower lea-chain or
// multiply. node is u64 (not u32) to prevent a redundant movl zero-extension
// when popping the node id into a 64-bit CACHE index.  {u64, u32, i32} = 16
// bytes. idx is now an index into CHILDREN[] (not a byte offset into input).
struct Frame {
    u64 node;
    u32 idx;
    i32 sum;
};

// This input has 80 reachable nodes from "you", so max DFS depth < 80.
// 128 gives headroom for other inputs without wasting much stack.
static Frame stk[128];

// Flat child-id list built at parse time.  Each node's children are stored
// contiguously and terminated by a 0 sentinel.  Node IDs are 15-bit
// (max 27647), so 0 is always a safe sentinel value.
//
// The sentinel approach eliminates pext from the hot DFS loop:
//   old: movl(lat5) -> pext(lat3) -> load CACHE[m](lat5) = 13-cycle chain
//   new: movzwl(lat5) -> load CACHE[m](lat5)             = 10-cycle chain
//
// It also unifies the cascade into the main loop — hitting a sentinel is
// equivalent to "this node's last child was just processed", so no separate
// cascade while-loop is needed.
//
// 591 lines × avg ~3.4 children + 591 sentinels ≈ 2600 entries; 4096 is safe.
static u16 CHILDREN[4096];
// Per-node start index into CHILDREN[].
// Stored as (colon_offset + 2) so the DFS loop reads ptr[OFFSETS[n]] == first
// child directly without an extra add each iteration.
static u32 CHILD_START[32 * 32 * 27];
static u32 children_len = 0;

// Generation counter: replaces per-call memset of CACHE (~110KB, ~6900 cyc)
// with a single increment.  A stale entry is one whose .gen != current_gen.
// NOTE: wraps at 2^32 calls; at that point a zeroed entry (gen==0) would
// incorrectly match current_gen==0.  Non-issue for AoC, but worth knowing.
static u32 global_gen = 0;

static bool parsed = false;

result
day() {
    // alignas(32): guarantees the 64-byte stride in the parse loop is always
    // 32-byte aligned, making vmovdqa valid for both lo and hi loads.
    alignas(32) static const u8 input[] = {
#embed "11.txt"
    };
    const u8 *ptr = input;
    u32 you_raw, out_raw;
    memcpy(&you_raw, "you ", 4);
    memcpy(&out_raw, "out ", 4);
    u32 you = parse_node(you_raw);
    u32 out = parse_node(out_raw);

    // -- Parse (runs once) ------------------------------------------------
    // Two ymm loads per 64-byte chunk find all ':' characters; tzcnt+blsr
    // extracts each position.  For each colon, we also walk its children and
    // store pre-extracted u16 IDs into CHILDREN[], terminated by 0.
    // pext is used here (cold path), but never in the hot DFS loop.
    if (!parsed) {
        const u64 len = sizeof(input);
        const __m256i colon = _mm256_set1_epi8(':');
        auto process_colon = [&](u64 pos) __attribute__((always_inline)) {
            u32 node_raw;
            memcpy(&node_raw, ptr + pos - 3, 4);
            u32 id = _pext_u32(node_raw, NODE_MASK);
            CHILD_START[id] = children_len;
            u32 child_pos = (u32)pos + 2; // first child byte offset in input
            while (true) {
                u32 child_raw;
                memcpy(&child_raw, ptr + child_pos, 4);
                CHILDREN[children_len++] = (u16)_pext_u32(child_raw, NODE_MASK);
                // top byte of child_raw is the separator after the child name:
                // '\n' (0x0A) means last child, ' ' (0x20) means more follow.
                if ((child_raw >> 24) == '\n') {
                    break;
                }
                child_pos += 4;
            }
            CHILDREN[children_len++] = 0; // sentinel
        };
        auto process_mask = [&](u64 mask, u64 base) __attribute__((always_inline)) {
            while (mask) {
                process_colon(base + (u32)__builtin_ctzll(mask));
                mask = _blsr_u64(mask);
            }
        };
        u64 offset = 0, aligned_end = len & ~(u64)63;
        for (; offset < aligned_end; offset += 64) {
            __m256i lo = _mm256_load_si256((const __m256i *)(ptr + offset));
            __m256i hi = _mm256_load_si256((const __m256i *)(ptr + offset + 32));
            u32 mlo = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, colon));
            u32 mhi = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, colon));
            process_mask((u64)mlo | ((u64)mhi << 32), offset);
        }
        if (offset < len) {
            // alignas(32) required: _mm256_load_si256 faults on unaligned addresses
            alignas(32) u8 buf[64] = {};
            memcpy(buf, ptr + offset, len - offset);
            __m256i lo = _mm256_load_si256((const __m256i *)buf);
            __m256i hi = _mm256_load_si256((const __m256i *)(buf + 32));
            u32 mlo = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, colon));
            u32 mhi = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, colon));
            u64 mask = ((u64)mlo | ((u64)mhi << 32)) & ((u64(1) << (len - offset)) - 1);
            process_mask(mask, offset);
        }
        parsed = true;
    }

    // -- DFS setup --------------------------------------------------------
    u32 current_gen = ++global_gen;
    CACHE[out] = { -1, current_gen }; // "out" is the only leaf; has no ':' line

    u32 depth = 0;
    u64 cur_node = you;
    u32 cur_idx = CHILD_START[you];
    i32 cur_sum = 0;

    // -- Iterative post-order DFS -----------------------------------------
    //
    // Hot path per CHILDREN[] entry:
    //   movzwl (lat5) -> cmpl/addl on CACHE[m] (lat5) = 10-cycle dep chain
    //   (was: movl(5) + pext(3) + cmpl(5) = 13 cycles)
    //
    // Three cases each iteration:
    //   m != 0, cached:   fold cur_sum += CACHE[m].sum, continue
    //   m != 0, uncached: push frame, descend into m
    //   m == 0 (sentinel): cur_node fully summed; store to CACHE, pop frame
    //
    // The sentinel unifies the cascade: multiple consecutive last-children
    // cascade naturally — each sentinel fires, pops one frame, and the next
    // iteration may immediately hit another sentinel for the parent.
    //
    // ptr is not accessed in this loop at all (only CHILDREN[] and CACHE[]).
    while (true) {
        u16 m = CHILDREN[cur_idx++];

        if (!m) [[unlikely]] {
            // Sentinel: cur_node's child list is exhausted.
            CACHE[cur_node] = { cur_sum, current_gen };
            if (depth == 0) {
                return { -(i64)cur_sum, 0 };
            }
            depth--;
            i32 res = cur_sum;
            cur_node = stk[depth].node;
            cur_idx = stk[depth].idx;
            cur_sum = stk[depth].sum + res;
            continue;
        }

        if (CACHE[m].gen != current_gen) [[unlikely]] {
            // Cold path: child unresolved - push frame and descend
            stk[depth++] = { cur_node, cur_idx, cur_sum };
            cur_node = m;
            cur_idx = CHILD_START[m];
            cur_sum = 0;
            continue;
        }

        // Hot path: child resolved - fold sum
        cur_sum += CACHE[m].sum;
    }
}

} // namespace _11

namespace _12 {

// clang-format off

// --- Day 12 constants --------------------------------------------------------
//
// Data-block coefficients for vpmaddubsw.
// Line format: "WW x HH: d1 d2 d3 d4 d5 d6\n"  (25 bytes)
// We load 16 bytes starting at ptr+7 into each 128-bit lane.
// The bytes at positions [0..15] encode d1..d5 as "TU " triples and
// the tens digit of d6 at position 15.
//
//  Pair   Bytes  Raw value after -'0'         Coeff  Contribution
//   0     0,1    d1_T, d1_U                   90, 9  -> 9*d1
//   1     2,3    ' '-'0'=240, d2_T            0, 90  -> 90*d2_T = 9*d2 (partial)
//   2     4,5    d2_U, ' '-'0'=240            9, 0   -> 9*d2_U  (partial)
//   3     6,7    d3_T, d3_U                   90, 9  -> 9*d3
//   4     8,9    ' '-'0'=240, d4_T            0, 90  -> 9*d4 (partial)
//   5    10,11   d4_U, ' '-'0'=240            9, 0   -> 9*d4_U (partial)
//   6    12,13   d5_T, d5_U                   90, 9  -> 9*d5
//   7    14,15   ' '-'0'=240, d6_T            0, 90  -> 90*d6_T
//
// hsum = 9*(d1+d2+d3+d4+d5) + 90*tens(d6)
static const __m256i v12_coeffs = _mm256_broadcastsi128_si256(_mm_set_epi8(
    90, 0, 9,   // bytes 15..13
    90, 0, 9,
    90, 0, 9,
    90, 0, 9,
    90, 0, 9,
    90          // byte 0
));

static const __m256i v12_ascii0 = _mm256_set1_epi8('0');
static const __m256i v12_ones = _mm256_set1_epi16(1);
// Skipping ascii subtraction on data blocks eliminates four vpaddb instructions
// per 8-line iteration. The resulting inflation of (sum of coefficients) * '0'
// = 585 * 48 = 28080 is absorbed by adding v12_28080 to wh_packed before comparing.
static const __m256i v12_28080 = _mm256_set1_epi32(28080);

// Header-field shuffle.
// One 32-byte load at ptr+0 covers both line headers:
//   Low  lane (bytes  0..15): line 0 header -- Wt0@0, Wu0@1, Ht0@3, Hu0@4
//   High lane (bytes 16..31): line 1 starts at byte 25, i.e. high-lane offset 9
//                              -- Wt1@9, Wu1@10, Ht1@12, Hu1@13
// vpshufb rearranges each lane to [Wt, Wu, Ht, Hu, 0, ...].
//
// _mm256_set_epi8 arguments run from byte 31 (first arg) down to byte 0 (last):
static const __m256i v12_hdr_shuf = _mm256_set_epi8(
    -1, -1, -1, -1,  // high lane bytes 15..12
    -1, -1, -1, -1,
    -1, -1, -1, -1,
    13, 12, 10,  9,  // high lane bytes  3..0: Hu1, Ht1, Wu1, Wt1

    -1, -1, -1, -1,  // low  lane bytes 15..12
    -1, -1, -1, -1,
    -1, -1, -1, -1,
     4,  3,  1,  0   // low  lane bytes  3..0: Hu0, Ht0, Wu0, Wt0
);

// vpmaddubsw coefficients for W/H: pairs (Wt,Wu) and (Ht,Hu) weighted [10,1].
static const __m256i v12_wh_coeff = _mm256_set_epi8(
    0,0,0,0, 0,0,0,0, 0,0,0,0, 1,10, 1,10,   // high lane: Hu,Ht coeff=1,10; Wu,Wt coeff=1,10
    0,0,0,0, 0,0,0,0, 0,0,0,0, 1,10, 1,10    // low  lane: same
);

// clang-format on

// TODO: vphaddd -> vpaddd
result
day() {
    static const u8 input[] = {
#embed "12.txt"
    };

    const u8 *ptr = input + 96; // skip the 6 shape-definitions header
    __m256i v_fail = _mm256_setzero_si256();

#pragma clang loop unroll_count(8)
    // Process 8 lines (4 pairs) per iteration.
    for (int _ = 0; _ < 1000; _ += 8, ptr += 200) {
        // Data loads before headers: writing to an xmm register zeros the upper
        // half of the corresponding ymm. If the compiler assigns the header load
        // to the same ymm it would later need for an xmm data load, it can't
        // use vmovdqu+vinserti128 and falls back to reconstructing the unaligned
        // load through integer registers. Loading data first avoids that conflict.
        __m128i dl1, dh1, dl2, dh2, dl3, dh3, dl4, dh4;
        memcpy(&dl1, ptr + 7, 16);
        memcpy(&dh1, ptr + 32, 16);
        memcpy(&dl2, ptr + 57, 16);
        memcpy(&dh2, ptr + 82, 16);
        memcpy(&dl3, ptr + 107, 16);
        memcpy(&dh3, ptr + 132, 16);
        memcpy(&dl4, ptr + 157, 16);
        memcpy(&dh4, ptr + 182, 16);

        __m256i t1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set_m128i(dh1, dl1), v12_coeffs), v12_ones);
        __m256i t2 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set_m128i(dh2, dl2), v12_coeffs), v12_ones);
        __m256i t3 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set_m128i(dh3, dl3), v12_coeffs), v12_ones);
        __m256i t4 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set_m128i(dh4, dl4), v12_coeffs), v12_ones);

        // hadd reduces 4 dwords per lane down to 1, interleaving even/odd lines:
        // layout after two hadds: [s0,s2,s4,s6 | s1,s3,s5,s7]
        __m256i sums = _mm256_hadd_epi32(_mm256_hadd_epi32(t1, t2), _mm256_hadd_epi32(t3, t4));

        // Headers: loaded after data to avoid ymm alias with xmm data loads.
        __m256i h1, h2, h3, h4;
        memcpy(&h1, ptr, 32);
        memcpy(&h2, ptr + 50, 32);
        memcpy(&h3, ptr + 100, 32);
        memcpy(&h4, ptr + 150, 32);

        auto wh_prod = [&](__m256i h) -> __m256i {
            h = _mm256_shuffle_epi8(h, v12_hdr_shuf);
            h = _mm256_sub_epi8(h, v12_ascii0);
            __m256i wh = _mm256_maddubs_epi16(h, v12_wh_coeff);
            return _mm256_mullo_epi16(wh, _mm256_srli_epi64(wh, 16));
        };
        __m256i wh1 = wh_prod(h1);
        __m256i wh2 = wh_prod(h2);
        __m256i wh3 = wh_prod(h3);
        __m256i wh4 = wh_prod(h4);

        // unpacklo_epi32 then unpacklo_epi64 packs word0 of each W*H result
        // into the same [0,2,4,6 | 1,3,5,7] layout as sums, so the comparison
        // is elementwise correct across all 8 lines
        __m256i unpacked_lo = _mm256_unpacklo_epi32(wh1, wh2);
        __m256i unpacked_hi = _mm256_unpacklo_epi32(wh3, wh4);
        __m256i wh_packed = _mm256_add_epi32(_mm256_unpacklo_epi64(unpacked_lo, unpacked_hi), v12_28080);

        // cmpgt: -1 where sum > W*H (fail); sub(-1) increments accumulator by 1.
        v_fail = _mm256_sub_epi32(v_fail, _mm256_cmpgt_epi32(sums, wh_packed));
    }

    // Horizontal sum of 8 dword accumulators
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v_fail), _mm256_extracti128_si256(v_fail, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return { 1000 - _mm_cvtsi128_si32(s), 0 };
}

} // namespace _12

namespace _6 {

// Precomputed AND masks for each possible token width 1-4.
// Avoids vmovd (GPR->XMM, FP45 bottleneck) + vpbroadcastd + scalar shlx by
// using a single vmovdqa/vpand load. Address is computable at the same time
// as the 4 row loads, so the mask is off the critical path.
const __m128i mask_tab[4] = {
    _mm_set1_epi32(0x000000FFu),
    _mm_set1_epi32(0x0000FFFFu),
    _mm_set1_epi32(0x00FFFFFFu),
    _mm_set1_epi32(0xFFFFFFFFu),
};

result
day() {
    alignas(32) static const u8 input[] = {
#embed "6.txt"
        // 64-byte over-read guard
        , ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
    };

    // All 5 rows are the same length
    u64 line_len = 0;
    const __m256i nl = _mm256_set1_epi8('\n');
    for (;;) {
        u32 mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((const __m256i *)(input + line_len)), nl));
        if (mask) {
            line_len += __builtin_ctz(mask);
            break;
        }
        line_len += 32;
    }

    const u8 *row1 = input;
    const u8 *row2 = input + (line_len + 1);
    const u8 *row3 = input + (line_len + 1) * 2;
    const u8 *row4 = input + (line_len + 1) * 3;
    const u8 *row_op = input + (line_len + 1) * 4;

    // Constants
    const __m128i v_mask_b4 = _mm_set1_epi32(0x10101010);
    const __m128i v_cmp0 = _mm_set1_epi32(0x00101010);
    const __m128i v_cmp1 = _mm_set1_epi32(0x00001010);
    const __m128i v_cmp2 = _mm_set1_epi32(0x00000010);
    const __m128i v_24 = _mm_set1_epi32(24);

    const __m128i v_0 = _mm_set1_epi8('0');
    const __m128i mul10 = _mm_set_epi8(1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10);
    const __m128i mul100 = _mm_set_epi16(1, 100, 1, 100, 1, 100, 1, 100);

    __m128i tot_acc = _mm_setzero_si128();
    __m128i tot_mul_acc = _mm_setzero_si128();

    // evaluate: parse and accumulate one column at position pos, width t_len
    auto evaluate = [&]<int TLen>(u32 pos) {
        // 4 independent loads.
        __m128i x1 = _mm_loadu_si32(row1 + pos);
        __m128i x2 = _mm_loadu_si32(row2 + pos);
        __m128i x3 = _mm_loadu_si32(row3 + pos);
        __m128i x4 = _mm_loadu_si32(row4 + pos);

        // The empty asm constraint stops the compiler from folding this into a
        // chain of `vpinsrd` instructions which would bottleneck the multiplier
        // ports (FP45).
        asm volatile("" : "+x"(x1), "+x"(x2), "+x"(x3), "+x"(x4));

        // Pack into one lane with two unpack levels
        __m128i lo = _mm_unpacklo_epi32(x1, x2);   // [row1, row2,    0,    0]
        __m128i hi = _mm_unpacklo_epi32(x3, x4);   // [row3, row4,    0,    0]
        __m128i lane = _mm_unpacklo_epi64(lo, hi); //[row1, row2, row3, row4]

        // Mask out bytes past token length with 0x00.
        // Table lookup replaces scalar shlx + vmovd (FP45) + vpbroadcastd.
        lane = _mm_and_si128(lane, mask_tab[TLen - 1]);

        // Compute parallel shift amounts per 32-bit element to left-justify
        // numbers. We isolate bit 4 and stack gt-comparisons to find the popcount
        // footprint effectively.
        __m128i b4 = _mm_and_si128(lane, v_mask_b4);
        __m128i c0 = _mm_cmpgt_epi32(b4, v_cmp0);
        __m128i c1 = _mm_cmpgt_epi32(b4, v_cmp1);
        __m128i c2 = _mm_cmpgt_epi32(b4, v_cmp2);

        // sum will map to[0, -1, -2, -3] based on number of digits.
        // Doing `24 + (sum << 3)` maps neatly to shifts `[24, 16, 8, 0]`.
        __m128i sum = _mm_add_epi32(c0, _mm_add_epi32(c1, c2));
        __m128i shift = _mm_add_epi32(v_24, _mm_slli_epi32(sum, 3));
        lane = _mm_sllv_epi32(lane, shift);

        // Parse: sat-sub '0', maddubs [10,1,…], madd[100,1,…]
        __m128i d = _mm_subs_epu8(lane, v_0);
        __m128i t16 = _mm_maddubs_epi16(d, mul10);
        __m128i vals = _mm_madd_epi16(t16, mul100);

        // Accumulate
        if (row_op[pos] == '+') {
            tot_acc = _mm_add_epi32(tot_acc, vals);
        } else {
            // Multiply v0*v1 and v2*v3 in SIMD (pmuludq), then scalar final mul
            __m128i m04 = _mm_mul_epu32(vals, _mm_srli_epi64(vals, 32));
            __m128i prod = _mm_mul_epu32(m04, _mm_srli_si128(m04, 8));
            tot_mul_acc = _mm_add_epi64(tot_mul_acc, prod);
        }
    };

    // TODO comment
    // Outer scan: process 64 columns per iteration using AVX2.
    // OR all 4 rows; slli_epi32(3) moves bit4 of each input byte -> bit7 (MSB).
    // movemask_epi8 extracts those MSBs: 1=digit-present, 0=all-rows-space.
    // Inverting gives 1 at each all-space (token-boundary) position.
    u32 P = 0;
    for (size_t offset = 0; offset <= line_len; offset += 64) {
        // Load first 32 bytes
        __m256i r1_lo = _mm256_loadu_si256((const __m256i *)(row1 + offset));
        __m256i r2_lo = _mm256_loadu_si256((const __m256i *)(row2 + offset));
        __m256i r3_lo = _mm256_loadu_si256((const __m256i *)(row3 + offset));
        __m256i r4_lo = _mm256_loadu_si256((const __m256i *)(row4 + offset));

        // Load next 32 bytes
        __m256i r1_hi = _mm256_loadu_si256((const __m256i *)(row1 + offset + 32));
        __m256i r2_hi = _mm256_loadu_si256((const __m256i *)(row2 + offset + 32));
        __m256i r3_hi = _mm256_loadu_si256((const __m256i *)(row3 + offset + 32));
        __m256i r4_hi = _mm256_loadu_si256((const __m256i *)(row4 + offset + 32));

        // Compute space mask for first 32 bytes
        __m256i or4_lo = _mm256_or_si256(_mm256_or_si256(r1_lo, r2_lo), _mm256_or_si256(r3_lo, r4_lo));
        u32 m_spc_lo = ~(u32)_mm256_movemask_epi8(_mm256_slli_epi32(or4_lo, 3));

        // Compute space mask for next 32 bytes
        __m256i or4_hi = _mm256_or_si256(_mm256_or_si256(r1_hi, r2_hi), _mm256_or_si256(r3_hi, r4_hi));
        u32 m_spc_hi = ~(u32)_mm256_movemask_epi8(_mm256_slli_epi32(or4_hi, 3));

        // Combine into 64-bit mask
        u64 m_spc = (u64)m_spc_lo | ((u64)m_spc_hi << 32);

        while (m_spc) {
            u32 idx = __builtin_ctzll(m_spc);
            m_spc = _blsr_u64(m_spc);
            u32 S = (u32)offset + idx;

            if (S > P) {
                u32 t_len = S - P;
                switch (t_len) {
                    case 1:
                        evaluate.template operator()<1>(P);
                        break;
                    case 2:
                        evaluate.template operator()<2>(P);
                        break;
                    case 3:
                        evaluate.template operator()<3>(P);
                        break;
                    default:
                        evaluate.template operator()<4>(P);
                        break;
                }
            }

            if (P > line_len) {
                goto done;
            }

            P = S + 1;
        }
    }
done:
    u32 acc[4];
    _mm_store_si128((__m128i *)acc, tot_acc);
    return { _mm_extract_epi64(tot_mul_acc, 0) + (i64)(acc[0] + acc[1] + acc[2] + acc[3]), 0 };
}

} // namespace _6

template <typename Fn>
void
TimeSolution(Fn &&Fn_, int Day) {
    const int Warmup = 100;
    const int Batches = 2000;
    const int BatchSize = 8;
    result Result;

    for (int I = 0; I < Warmup; ++I) {
        Result = Fn_();
        bench::DoNotOptimizeAway(Result);
    }

    u64 MinCycles = UINT64_MAX;

    for (int Batch = 0; Batch < Batches; ++Batch) {
        u64 Start = bench::start();

        for (int _ = 0; _ < BatchSize; ++_) {
            result R = Fn_();
            bench::DoNotOptimizeAway(R);
            Result = R;
        }

        u64 End = bench::stop();

        u64 Cycles = (End - Start) / BatchSize;
        MinCycles = std::min(MinCycles, Cycles);
    }

    // Assume 4.2GHz cat /proc/cpuinfo | head -30
    double ns = MinCycles / 4.2;
    printf("Day %2d: %-12ld %-12ld %6lu cyc (%lu ns)\n", Day, Result.P1, Result.P2, MinCycles, (u64)ns);
}

int
main() {
    TimeSolution(_1::day, 1);
    TimeSolution(_2::day, 2);
    TimeSolution(_3::day, 3);
    TimeSolution(_4::day, 4);
    TimeSolution(_6::day, 6);
    TimeSolution(_9::day, 9);
    TimeSolution(_10::day, 10);
    TimeSolution(_11::day, 11);
    TimeSolution(_12::day, 12);
    return 0;
}
