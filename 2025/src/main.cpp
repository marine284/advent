#include "src/defs.hpp"

#include "src/bench.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring> // memcpy
#include <immintrin.h>

#define MCA_BEGIN(x) __asm__ volatile("# LLVM-MCA-BEGIN " #x)
#define MCA_END(x) __asm__ volatile("# LLVM-MCA-END " #x)

struct result {
    i64 P1, P2;
};

namespace day01 {

alignas(4096) static const u16 digit_lut[UINT16_MAX + 1] = {
#include "digit_lut.inc"
};

static const __m256i v_newline = _mm256_set1_epi8('\n');
static const __m256i v_R = _mm256_set1_epi16('R');
static const __m256i v_zero = _mm256_setzero_si256();
static const __m256i v_100 = _mm256_set1_epi16(100);
static const __m256i v_div100_mul = _mm256_set1_epi16(2622);
static const __m256i v_bcast_mask =
    _mm256_set_epi8(15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, //
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
static const __m256i v_w15_bcast =
    _mm256_set_epi8(15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, //
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

} // namespace day01

result
Day01() {
    using namespace day01;
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

namespace day02 {

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

} // namespace day02

result
Day02() {
    using namespace day02;
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

        if (slen > elen)
            continue;

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

namespace day03 {

static const __m256i v_digit[10] = {
    _mm256_set1_epi8('0'), _mm256_set1_epi8('1'), _mm256_set1_epi8('2'), _mm256_set1_epi8('3'),
    _mm256_set1_epi8('4'), _mm256_set1_epi8('5'), _mm256_set1_epi8('6'), _mm256_set1_epi8('7'),
    _mm256_set1_epi8('8'), _mm256_set1_epi8('9'),
};

}

result
Day03() {
    using namespace day03;
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

namespace day12 {

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

} // namespace day12

// TODO: vphaddd -> vpaddd
result
Day12() {
    using namespace day12;
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

        __m256i t1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set_m128i(dh1, dl1), v12_coeffs),
                                       v12_ones);
        __m256i t2 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set_m128i(dh2, dl2), v12_coeffs),
                                       v12_ones);
        __m256i t3 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set_m128i(dh3, dl3), v12_coeffs),
                                       v12_ones);
        __m256i t4 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set_m128i(dh4, dl4), v12_coeffs),
                                       v12_ones);

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
        __m256i wh_packed =
            _mm256_add_epi32(_mm256_unpacklo_epi64(unpacked_lo, unpacked_hi), v12_28080);

        // cmpgt: -1 where sum > W*H (fail); sub(-1) increments accumulator by 1.
        v_fail = _mm256_sub_epi32(v_fail, _mm256_cmpgt_epi32(sums, wh_packed));
    }

    // Horizontal sum of 8 dword accumulators
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v_fail), _mm256_extracti128_si256(v_fail, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return { 1000 - _mm_cvtsi128_si32(s), 0 };
}

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
    printf("Day %2d: %-12ld %-12ld %6lu cyc (%lu ns)\n", Day, Result.P1, Result.P2, MinCycles,
           (u64)ns);
}

namespace day11 {

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
// arithmetic into `shlq $4, %reg` (a single-cycle shift) rather than a
// slower lea-chain or multiply.
// node is u64 (not u32) to prevent a redundant movl zero-extension when
// popping the node id into a 64-bit CACHE index.  Dropping the old _pad
// field preserves the 16-byte size: {u64, u32, i32} = 16 bytes exactly.
struct Frame {
    u64 node;
    u32 idx; // points to the *next* unprocessed child (already past the ':')
    i32 sum;
};

// This input has 80 reachable nodes from "you", so max DFS depth < 80.
// 128 gives headroom for other inputs without wasting much stack.
static Frame stk[128];

// Stored as (colon_offset + 2) so the DFS loop reads the first child
// directly at ptr[OFFSETS[n]] without an extra add each iteration.
static u32 OFFSETS[32 * 32 * 27];

// Generation counter: replaces per-call memset of CACHE (~110KB, ~6900 cyc)
// with a single increment.  A stale entry is one whose .gen != current_gen.
// NOTE: wraps at 2^32 calls; at that point a zeroed entry (gen==0) would
// incorrectly match current_gen==0.  Non-issue for AoC, but worth knowing.
static u32 global_gen = 0;

static bool parsed = false;

} // namespace day11

result
Day11() {
    alignas(32) static const u8 input[] = {
#embed "11.txt"
    };
    using namespace day11;
    const u8 *ptr = input;
    u32 you_raw, out_raw;
    memcpy(&you_raw, "you ", 4);
    memcpy(&out_raw, "out ", 4);
    u32 you = parse_node(you_raw);
    u32 out = parse_node(out_raw);

    // -- Parse (runs once) ------------------------------------------------
    // Two ymm loads per 64-byte chunk find all ':' characters; tzcnt+blsr
    // extracts each position.  Cold path - excluded from the benchmark loop.
    if (!parsed) {
        const u64 len = sizeof(input);
        const __m256i colon = _mm256_set1_epi8(':');
        auto process_mask = [&](u64 mask, u64 base) __attribute__((always_inline)) {
            while (mask) {
                u64 pos = base + (u32)__builtin_ctzll(mask);
                u32 raw;
                memcpy(&raw, ptr + pos - 3, 4);
                // +2 pre-baked so DFS reads ptr[OFFSETS[n]] == first child directly
                OFFSETS[_pext_u32(raw, NODE_MASK)] = (u32)pos + 2;
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

    u32 depth    = 0;
    u64 cur_node = you; // u64: native register width, no zero-extension into CACHE index
    u64 cur_idx  = OFFSETS[you]; // u64: folds ptr[cur_idx - 1] into -1(%base,%idx) directly
    i32 cur_sum  = 0;

    // -- Iterative post-order DFS -----------------------------------------
    //
    // Each iteration reads one child from ptr[cur_idx..cur_idx+4]:
    //   bytes 0-2  - child node name (parsed via pext)
    //   byte  3    - separator: ' ' (more siblings follow) or '\n' (last child)
    //
    // Hot path (child already cached): fold sum, continue or cascade.
    // Cold path (child not yet cached): push current frame, descend.
    //
    // Cascade invariant: when a node completes, we pop the parent frame.
    // parent.idx was advanced *before* the child was pushed, so
    // ptr[parent.idx - 1] is the separator that followed that child;
    // '\n' means the parent is also complete and we cascade further.
    while (true) {
        u32 m_raw;
        memcpy(&m_raw, ptr + cur_idx, 4);
        u32 m = parse_node(m_raw);
        // Separator sits in the top byte of the same word we already loaded,
        // no extra memory access needed.
        cur_idx += 4;

        if (CACHE[m].gen != current_gen) [[unlikely]] {
            // Cold path: child unresolved - push frame and descend
            stk[depth++] = { cur_node, (u32)cur_idx, cur_sum };
            cur_node = m;
            cur_idx  = OFFSETS[m];
            cur_sum  = 0;
            continue;
        }

        // Hot path: child resolved - fold and check separator.
        cur_sum += CACHE[m].sum;

        // Bit 29 of m_raw is bit 5 of the separator byte (the 4th byte).
        // Space 0x20: bit 5 set -> more siblings, keep going.
        // Newline 0x0A: bit 5 clear -> last child, cascade.
        // Replaces andl $0xFF000000 + cmpl with a single testl, saving one uop.
        if (m_raw & 0x20000000u) [[likely]] {
            continue; // more siblings
        }

        // Last child of cur_node: store result and cascade up the stack
        while (true) {
            CACHE[cur_node] = { cur_sum, current_gen };
            if (depth == 0) {
                return { -(i64)cur_sum, 0 };
            }
            depth--;
            i32 res  = cur_sum;
            cur_node = stk[depth].node;
            cur_idx  = stk[depth].idx; // zero-extends u32 -> u64 implicitly
            cur_sum  = stk[depth].sum + res;
            // See cascade invariant above.
            // u64 cur_idx lets this compile to cmpb $10, -1(%base,%idx) directly.
            if (ptr[cur_idx - 1] != '\n') {
                break;
            }
        }
    }
}

int
main() {
    // TimeSolution(Day01, 1);
    // TimeSolution(Day02, 2);
    // TimeSolution(Day03, 3);
    TimeSolution(Day11, 11);
    // TimeSolution(Day12, 12);
    return 0;
}
