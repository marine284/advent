// Lookup table generator for AoC 2025 Day 1
// Generates a 64KB table mapping u16 (raw ASCII bytes) -> parsed value (0-99)

#include "defs.hpp"

#include <cstdio>

int main() {
    u16 lut[UINT16_MAX + 1] = {0};

    // Fill valid combinations
    // Little-endian: low byte at lower address (tens digit)
    for (int lo = 0; lo < 256; lo++) {
        for (int hi = 0; hi < 256; hi++) {
            u16 key = lo | (hi << 8);

            // Parse low byte (tens digit)
            int d0 = lo - '0';
            if (d0 < 0 || d0 > 9) d0 = 0;

            // Parse high byte (ones digit)
            int d1 = hi - '0';
            if (d1 < 0 || d1 > 9) d1 = 0;

            // Combine: tens * 10 + ones
            lut[key] = d0 * 10 + d1;
        }
    }

    for (int i = 0; i < UINT16_MAX + 1; i++) {
        if (i % 16 == 0) printf("\n    ");
        printf("%u", lut[i]);
        if (i < UINT16_MAX) printf(",");
    }
    printf("\n");

    return 0;
}
