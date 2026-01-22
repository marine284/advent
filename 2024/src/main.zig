const std = @import("std");
const builtin = @import("builtin");
const cast = @import("anycast").cast;
const bitcast = @import("anycast").bitcast;
const Allocator = std.mem.Allocator;
const Map = std.AutoHashMap;
const Array = std.ArrayList;
const StrMap = std.StringHashMap;
const BitSet = std.DynamicBitSet;
const maxInt = std.math.maxInt;
const isDigit = std.ascii.isDigit;
const tokenizeAny = std.mem.tokenizeAny;
const tokenizeSequence = std.mem.tokenizeSequence;
const tokenizeScalar = std.mem.tokenizeScalar;
const splitAny = std.mem.splitAny;
const splitSequence = std.mem.splitSequence;
const splitScalar = std.mem.splitScalar;
const indexOf = std.mem.indexOf;
const indexOfScalar = std.mem.indexOfScalar;
const indexOfAny = std.mem.indexOfAny;
const indexOfStr = std.mem.indexOfPosLinear;
const indexOfNone = std.mem.indexOfNone;
const lastIndexOf = std.mem.lastIndexOfScalar;
const lastIndexOfAny = std.mem.lastIndexOfAny;
const lastIndexOfStr = std.mem.lastIndexOfLinear;
const readInt = std.mem.readInt;
const doNotOptimizeAway = std.mem.doNotOptimizeAway;
const print = std.debug.print;
const trim = std.mem.trim;
const sliceMin = std.mem.min;
const sliceMax = std.mem.max;
const parseInt = std.fmt.parseInt;
const parseFloat = std.fmt.parseFloat;
const assert = std.debug.assert;
const asc = std.sort.asc;
const desc = std.sort.desc;
const sort = std.sort.pdq;
const eql = std.mem.eql;
const log10_int = std.math.log10_int;
const pow = std.math.pow;
const alignForward = std.mem.alignForward;
const suggestVectorLength = std.simd.suggestVectorLength;
const __u8x8 = @Vector(8, u8);
const __u8x16 = @Vector(16, u8);
const __u8x32 = @Vector(32, u8);
const __i8x8 = @Vector(8, i8);
const __i8x16 = @Vector(16, i8);
const __i8x32 = @Vector(32, i8);
const __u16x8 = @Vector(8, u16);
const __u16x16 = @Vector(16, u16);
const __i16x8 = @Vector(16, i8);
const __i16x16 = @Vector(16, i16);
const __u32x8 = @Vector(8, u32);
const __u32x16 = @Vector(16, u32);
const __i32x8 = @Vector(8, i32);
const __i32x16 = @Vector(16, i32);
const __i32 = @Vector(suggestVectorLength(i32).?, i32);
const __u32 = @Vector(suggestVectorLength(u32).?, u32);

var arena_instance = std.heap.ArenaAllocator.init(std.heap.page_allocator);
const arena = arena_instance.allocator();

// Cycle-accurate timing (mirrors the C++ bench::start / bench::stop)

inline fn rdtsc_start() u64 {
    var lo: u32 = undefined;
    var hi: u32 = undefined;
    asm volatile ("lfence\n\trdtsc\n\tlfence"
        : [lo] "={eax}" (lo),
          [hi] "={edx}" (hi),
        :
        : .{ .memory = true });
    return @as(u64, hi) << 32 | lo;
}

inline fn rdtscp_stop() u64 {
    var lo: u32 = undefined;
    var hi: u32 = undefined;
    asm volatile ("rdtscp\n\tlfence"
        : [lo] "={eax}" (lo),
          [hi] "={edx}" (hi),
        :
        : .{ .rcx = true, .memory = true });
    return @as(u64, hi) << 32 | lo;
}

/// TSC frequency in GHz - update to match your machine's base clock.
/// Check: dmesg | grep -i "tsc: Refined"
const tsc_ghz: f64 = 4.2;

inline fn cycToNs(cyc: u64) u64 {
    return @intFromFloat(@as(f64, @floatFromInt(cyc)) / tsc_ghz);
}

var total_cycles: u64 = 0;

fn time(comptime f: anytype, day: u32) void {
    const warmup = 100;
    const batches = 2000;
    const batch_size = 8;

    const Ret = blk: {
        const R = @typeInfo(@TypeOf(f)).@"fn".return_type.?;
        break :blk if (@typeInfo(R) == .error_union) @typeInfo(R).error_union.payload else R;
    };

    var ans: Ret = undefined;

    for (0..warmup) |_| {
        ans = f() catch unreachable;
        doNotOptimizeAway(ans);
    }

    var min_cyc: u64 = maxInt(u64);
    for (0..batches) |_| {
        const start = rdtsc_start();
        for (0..batch_size) |_| {
            ans = f() catch unreachable;
            doNotOptimizeAway(ans);
        }
        const cyc = (rdtscp_stop() - start) / batch_size;
        if (cyc < min_cyc) min_cyc = cyc;
    }

    total_cycles += min_cyc;
    const fmt = if (@typeInfo(Ret) == .pointer) "Day {: >2}: {s: <17} {:>6} cyc  ({} ns)\n" else "Day {: >2}: {: <17} {:>6} cyc  ({} ns)\n";
    print(fmt, .{ day, ans, min_cyc, cycToNs(min_cyc) });
}

pub fn main() void {
    const days = .{
        .{ @"1", 1 },
        .{ @"2", 2 },
        .{ @"3", 3 },
        .{ @"4", 4 },
        .{ @"5", 5 },
        .{ @"6", 6 },
        .{ @"7", 7 },
        .{ @"8", 8 },
        .{ @"9", 9 },
        .{ @"10", 10 },
        .{ @"11", 11 },
        .{ @"12", 12 },
        .{ @"13", 13 },
        .{ @"14", 14 },
        .{ @"15", 15 },
        .{ @"17", 17 },
        .{ @"18", 18 },
        .{ @"25", 25 },
    };
    inline for (days) |day| time(day[0], day[1]);
    print("Total: {} cyc  ({} us)\n", .{ total_cycles, @as(u64, @intFromFloat(@as(f64, @floatFromInt(total_cycles)) / tsc_ghz)) / 1000 });
}

// noinline fn @"1"() !u64 {
//     var cols: [2]@Vector(1000, i32) = undefined;
//     var nums = tokenizeAny(u8, @embedFile("1.txt"), " \n");
//     for (0..1000) |row| {
//         for (&cols) |*col| col[row] = try parseInt(i32, nums.next().?, 10);
//     }
//     for (&cols) |*col| sort(i32, @ptrCast(col), {}, asc(i32));
//     return @reduce(.Add, @abs(cols[0] - cols[1]));
// }

fn ChunkRet(comptime C: usize, comptime T: type) type {
    var info = @typeInfo(T).pointer;
    if (info.size == .one and @typeInfo(info.child) == .array) {
        // *[N]T -> *[M][C]T
        const A = @typeInfo(info.child).array;
        info.child = [(A.len + C - 1) / C][C]A.child;
    } else {
        // []T -> [][C]T
        if (info.size == .one) @compileError("chunksf: expected *[N]T or []T, got *" ++ @typeName(info.child));
        info.child = [C]info.child;
    }
    return @Type(.{ .pointer = info });
}

fn chunksf(comptime C: usize, input: anytype) ChunkRet(C, @TypeOf(input)) {
    const Ret = ChunkRet(C, @TypeOf(input));
    // *[N]T -> *[M][C]T
    if (@typeInfo(Ret).pointer.size == .one) return @ptrCast(input);
    // []T -> [][C]T
    return @as(Ret.ptr, @ptrCast(input.ptr))[0 .. (input.len + C - 1) / C]; // round up - caller guarantees padding
}

const avx2 = struct {
    extern fn @"llvm.x86.avx2.pmadd.ub.sw"(@Vector(32, u8), @Vector(32, i8)) @Vector(16, i16);
    extern fn @"llvm.x86.avx2.pmadd.wd"(@Vector(16, i16), @Vector(16, i16)) @Vector(8, i32);
    extern fn @"llvm.x86.avx2.pshuf.b"(@Vector(32, u8), @Vector(32, u8)) @Vector(32, u8);
};

inline fn maddubs(a: @Vector(32, u8), b: @Vector(32, i8)) @Vector(16, i16) {
    if (builtin.zig_backend == .stage2_llvm) {
        return avx2.@"llvm.x86.avx2.pmadd.ub.sw"(a, b);
    }
    return asm ("vpmaddubsw %[b], %[a], %[ret]"
        : [ret] "=x" (-> @Vector(16, i16)),
        : [a] "x" (a),
          [b] "x" (b), // 'xm' allows memory folding but it doesn't compile in debug mode
    );
}

inline fn maddwd(a: @Vector(16, i16), b: @Vector(16, i16)) @Vector(8, i32) {
    if (builtin.zig_backend == .stage2_llvm) {
        return avx2.@"llvm.x86.avx2.pmadd.wd"(a, b);
    }
    return asm ("vpmaddwd %[b], %[a], %[ret]"
        : [ret] "=x" (-> @Vector(8, i32)),
        : [a] "x" (a),
          [b] "x" (b),
    );
}

inline fn vpshufb(a: @Vector(32, u8), b: @Vector(32, u8)) @Vector(32, u8) {
    if (builtin.zig_backend == .stage2_llvm) {
        return avx2.@"llvm.x86.avx2.pshuf.b"(a, b);
    }
    return asm ("vpshufb %[b], %[a], %[ret]"
        : [ret] "=x" (-> @Vector(32, u8)),
        : [a] "x" (a),
          [b] "x" (b),
    );
}

noinline fn @"1"() !u64 {
    const S = struct {
        noinline fn parseSimdPipelined(input: [*]const u8, left: *[1000]u32, right: *[1000]u32) void {
            const bias: __u8x32 = @splat('0');

            // grab 5 digits at offset 0 (left) and 8 (right), zero the rest
            const shuf_ctrl: __u8x32 = .{
                0, 1, 2,  3,  4,  0x80, 0x80, 0x80,
                8, 9, 10, 11, 12, 0x80, 0x80, 0x80,
                0, 1, 2,  3,  4,  0x80, 0x80, 0x80,
                8, 9, 10, 11, 12, 0x80, 0x80, 0x80,
            };

            // horizontal radix: pairs -> 2-digit chunks
            const w1: __u8x32 = .{
                10, 1, 10, 1, 1, 0, 0, 0, 10, 1, 10, 1, 1, 0, 0, 0,
                10, 1, 10, 1, 1, 0, 0, 0, 10, 1, 10, 1, 1, 0, 0, 0,
            };

            // 2-digit chunks -> 4-digit + lone digit
            const w2: __i16x16 = .{ 100, 1, 1, 0, 100, 1, 1, 0, 100, 1, 1, 0, 100, 1, 1, 0 };

            // final: 4-digit * 10 + lone digit
            const w3: __i16x16 = .{ 10, 1, 10, 1, 0, 0, 0, 0, 10, 1, 10, 1, 0, 0, 0, 0 };

            // gather low halves for w3
            const extract: [16]i32 = .{ 0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14 };

            // transpose: [L_k, R_k, _, _, L_{k+4}, R_{k+4}, _, _] -> grouped Ls/Rs
            const pack_pair: [8]i32 = .{
                0, ~@as(i32, 0), 4, ~@as(i32, 4),
                1, ~@as(i32, 1), 5, ~@as(i32, 5),
            };

            // deinterleave
            const merge_left: [8]i32 = .{
                0, 1, ~@as(i32, 0), ~@as(i32, 1),
                2, 3, ~@as(i32, 2), ~@as(i32, 3),
            };
            const merge_right: [8]i32 = .{
                4, 5, ~@as(i32, 4), ~@as(i32, 5),
                6, 7, ~@as(i32, 6), ~@as(i32, 7),
            };

            // pre-load first iteration
            var pending: [4]__u8x32 = undefined;
            inline for (0..4) |k| {
                // row k and k+4 in separate lanes
                const lo: __u8x16 = cast(*const [16]u8, input + k * 14).*;
                const hi: __u8x16 = cast(*const [16]u8, input + (k + 4) * 14).*;
                pending[k] = std.simd.join(lo, hi);
            }

            var j: usize = 0;
            var base: usize = 0;

            while (j < 1000) : ({
                j += 8;
                base += 14 * 8;
            }) {
                // grab what we pre-loaded
                const chunks = pending;

                // process all 4 chunks
                var results: [4]__i32x8 = undefined;
                inline for (0..4) |k| {
                    const arranged = vpshufb(chunks[k], shuf_ctrl);
                    const s1 = maddubs(arranged -% bias, w1);
                    const s2 = bitcast(__i16x16, maddwd(s1, w2));
                    const s3 = @shuffle(i16, s2, undefined, extract);
                    results[k] = maddwd(s3, w3);
                }

                // load next iteration while results are still cooking
                const next_base = base + 14 * 8;
                inline for (0..4) |k| {
                    const lo: __u8x16 = cast(*const [16]u8, input + next_base + k * 14).*;
                    const hi: __u8x16 = cast(*const [16]u8, input + next_base + (k + 4) * 14).*;
                    pending[k] = std.simd.join(lo, hi);
                }

                // transpose and store
                const pack_01 = @shuffle(i32, results[0], results[1], pack_pair);
                const pack_23 = @shuffle(i32, results[2], results[3], pack_pair);
                const left_vec = bitcast(__u32x8, @shuffle(i32, pack_01, pack_23, merge_left));
                const right_vec = bitcast(__u32x8, @shuffle(i32, pack_01, pack_23, merge_right));

                cast(*__u32x8, left[j..].ptr).* = left_vec;
                cast(*__u32x8, right[j..].ptr).* = right_vec;
            }
        }

        const Scratch = struct {
            var left: [1024]u32 align(64) = undefined;
            var right: [1024]u32 align(64) = undefined;
            var pos: [768]u32 = undefined; // zero-init in .bss
            var tmp: [1000]u32 = undefined;
        };
        const SS = Scratch;

        inline fn radixImpl(data: *[1000]u32) void {
            const split = 424;
            const hi = SS.pos[0..392]; // (99999 >> 8) + 1 = 391 (minimum buckets required)
            const lo = SS.pos[split..][0..256];

            // counting
            for (chunksf(8, data)) |chunk| {
                inline for (chunk) |x| {
                    lo[x & 0xFF] += 1;
                    hi[x >> 8] += 1;
                }
            }

            // prefix sums
            var sum_lo: u32 = 0;
            for (lo) |*count| {
                sum_lo += count.*;
                count.* = sum_lo - count.*;
            }
            var sum_hi: u32 = 0;
            for (hi) |*count| {
                sum_hi += count.*;
                count.* = sum_hi - count.*;
            }

            // scatters: data -> tmp by low byte
            for (data) |x| {
                const idx = cast(usize, x & 0xFF);
                SS.tmp[lo[idx]] = x;
                lo[idx] += 1;
            }
            // scatters: tmp -> data by high byte
            for (&SS.tmp) |x| {
                const idx = cast(usize, x >> 8);
                data[hi[idx]] = x;
                hi[idx] += 1;
            }
        }

        noinline fn radix(data: *[1000]u32) void {
            radixImpl(data);
            @memset(&SS.pos, 0); // tail
        }
    };
    const SS = S.SS;

    const input = @embedFile("1.txt") ++ "00000   00000\n";

    S.parseSimdPipelined(input.ptr, SS.left[0..1000], SS.right[0..1000]);

    S.radix(SS.left[0..1000]);
    S.radix(SS.right[0..1000]);

    const VL = @typeInfo(__i32).vector.len;
    const N = comptime alignForward(usize, 1000, VL);
    if (N > 1000) {
        @memset(SS.left[1000..N], 0);
        @memset(SS.right[1000..N], 0);
    }

    var acc: __u32 = @splat(0);
    inline for (chunksf(VL, SS.left[0..N]), chunksf(VL, SS.right[0..N])) |l, r| {
        acc += @abs(bitcast(__i32, l) - bitcast(__i32, r));
    }
    return @reduce(.Add, acc);
}

noinline fn @"2"() !u64 {
    const S = struct {
        inline fn readNum(input: []const u8, pos: *u32) i8 {
            const p = pos.*;
            const n1 = input[p] - '0';
            // lookahead without committing
            const nc = input[p + 1];
            if (nc >= '0' and nc <= '9') {
                @branchHint(.likely);
                pos.* = p + 2; // commit 2-digit read
                return @intCast(n1 * 10 + (nc - '0'));
            }
            pos.* = p + 1; // commit 1-digit read
            return @intCast(n1);
        }
    };
    const input = @embedFile("2.txt");
    const valid_lut = comptime blk: {
        var arr: [256]bool align(32) = @splat(false);
        for ([_]u8{ 1, 2, 3, 253, 254, 255 }) |i| arr[i] = true;
        break :blk arr;
    };
    var lines: [1001]u32 = undefined;
    lines[0] = 0;
    var num_lines: u32 = 0;
    var offset: u32 = 0;
    //comptime var offset: u32 = 0; // unroll
    const newline_vec: __u8x32 = @splat('\n');

    // newline detection pass
    while (offset + 32 <= input.len) : (offset += 32) {
        const chunk: __u8x32 = input[offset..][0..32].*;
        var mask: u32 = @bitCast(chunk == newline_vec);
        while (mask != 0) {
            const pos = @ctz(mask);
            mask &= mask - 1;
            lines[num_lines] = offset + pos + 1;
            num_lines += 1;
        }
    }
    while (offset < input.len) : (offset += 1) {
        if ((input[offset] == '\n') and (num_lines < lines.len)) {
            lines[num_lines] = offset + 1;
            num_lines += 1;
        }
    }

    lines[num_lines] = @intCast(input.len + 1); // sentinel

    var count: u32 = 0;
    for (0..num_lines) |i| {
        var pos: u32 = lines[i];
        const end: u32 = lines[i + 1] - 1;
        if (pos >= end) continue;
        const n1 = S.readNum(input, &pos);
        pos += 1;
        var n2 = S.readNum(input, &pos);
        var diff1 = n2 - n1;
        if (!valid_lut[bitcast(u8, diff1)]) continue;
        var is_valid: bool = true;
        while (pos + 1 < end) {
            pos += 1;
            const n3 = S.readNum(input, &pos);
            const diff2 = n3 - n2;
            if (!valid_lut[bitcast(u8, diff2)] or (diff1 ^ diff2) < 0) {
                is_valid = false;
                break;
            }
            n2 = n3;
            diff1 = diff2;
        }
        count += @intFromBool(is_valid);
    }
    return count;
}

noinline fn @"3"() !u64 {
    const input = @embedFile("3.txt");
    var count: u64 = 0;
    var off: usize = 0;
    while (true) { // zig fmt: off
        const pos   =   off + (indexOf(u8, input[off..], "ul(") orelse break) + "ul(".len;
        const comma =   pos + (indexOf(u8, input[pos..],   ",") orelse { off = pos; continue; });
        const paren = comma + (indexOf(u8, input[comma..], ")") orelse { off = pos; continue; });
        count += (parseInt(u64, input[pos..comma], 10)          catch  { off = pos; continue; })
               * (parseInt(u64, input[comma + 1 .. paren], 10)  catch  { off = pos; continue; });
        off = paren + 1;
    }
    // zig fmt: on
    return count;
}

noinline fn @"4"() !u64 {
    const input = @embedFile("4.txt");
    const width: usize = 140;
    const stride: usize = width + 1;
    const height: usize = input.len / stride;
    // zig fmt: off
    const step:  [4]usize = .{         1,     stride, stride + 1, stride - 1 }; // r d dr dl
    const x_min: [4]usize = .{         0,          0,          0,          3 };
    const x_max: [4]usize = .{ width - 3,      width,  width - 3,      width };
    const y_max: [4]usize = .{    height, height - 3, height - 3, height - 3 };
    // zig fmt: on
    var count: u32 = 0;
    for (0..4) |dir| {
        for (0..y_max[dir]) |y| {
            for (x_min[dir]..x_max[dir]) |x| {
                var word: u32 = 0;
                const pos = y * stride + x;
                inline for (0..4) |i| word |= @as(u32, input[pos + i * step[dir]]) << @intCast(i * 8);
                count += @intFromBool(word == 0x584D4153 or word == 0x53414D58);
            }
        }
    }
    return count;
}

noinline fn @"5"() !u64 {
    const input = @embedFile("5.txt");
    var rules: [100]u128 = @splat(0);
    var pos: usize = 0;

    while (pos < input.len) {
        if (input[pos] == '\n') {
            pos += 1;
            break;
        }
        const x = (input[pos] - '0') * 10 + (input[pos + 1] - '0');
        const y = (input[pos + 3] - '0') * 10 + (input[pos + 4] - '0');
        rules[x] |= @as(u128, 1) << @intCast(y);
        pos += 6;
    }

    var count: u32 = 0;
    while (pos < input.len) {
        var update: [24]u8 = undefined;
        var update_len: usize = 0;
        var mask: u128 = 0;
        var valid = true;

        while (pos < input.len and input[pos] != '\n') {
            const page = (input[pos] - '0') * 10 + (input[pos + 1] - '0');
            update[update_len] = page;
            update_len += 1;

            if (rules[page] & mask != 0) valid = false;
            mask |= @as(u128, 1) << @intCast(page);
            pos += @as(usize, 2) + @intFromBool(pos < input.len and input[pos + 2] == ',');
        }

        if (valid and update_len > 0) count += update[update_len / 2];
        if (pos < input.len and input[pos] == '\n') pos += 1;
    }
    return count;
}

noinline fn @"6"() !u64 {
    const input = @embedFile("6.txt");
    const stride: usize = 131;
    const height: usize = 130;
    var visited: [stride * height]u8 = @splat(1);
    var pos: usize = indexOfScalar(u8, input, '^').?;
    var count: u32 = 0;
    const dirs = [4]isize{ -@as(isize, stride), 1, stride, -1 }; // up right down left
    var dir_idx: u2 = 0;
    outer: while (true) : (dir_idx +%= 1) {
        const dir = dirs[dir_idx];
        while (true) {
            count += visited[pos];
            visited[pos] = 0;
            const next = cast(usize, cast(isize, pos) + dir);
            //if (next < 0 or next >= stride * height) break :outer; // sample
            if (input[next] == '#') break;
            if (input[next] == '\n') break :outer;
            pos = next;
        }
    }
    return count;
}

noinline fn @"7"() !u64 {
    var lines = tokenizeScalar(u8, @embedFile("7.txt"), '\n');
    var count: u64 = 0;

    while (lines.next()) |line| {
        const colon = indexOfScalar(u8, line, ':').?;
        const target = try parseInt(u64, line[0..colon], 10);

        var values: [16]u64 = undefined;
        var len: u8 = 0;
        var nums = tokenizeScalar(u8, line[colon + 2 ..], ' ');
        while (nums.next()) |num| : (len += 1) values[len] = try parseInt(u64, num, 10);

        for (0..@as(u64, 1) << @intCast(len - 1)) |combo| {
            var result = values[0];
            for (0..len - 1) |i|
                result = if ((combo >> @intCast(i)) & 1 == 1)
                    result * values[i + 1]
                else
                    result + values[i + 1];
            if (result == target) {
                count += target;
                break;
            }
        }
    }
    return count;
}

noinline fn @"8"() !u64 {
    const input = @embedFile("8.txt");
    var antinodes: [50]u64 = @splat(0);
    var frequencies: [75][4][2]u8 = @splat(@splat(@splat(0)));
    var counts: [75]u8 = @splat(0);

    for (0..50) |y| {
        for (0..50) |x| {
            const c = input[y * 51 + x];
            if (c == '.') continue;

            const freq = c - '0';
            const count = counts[freq];
            frequencies[freq][count] = .{ @intCast(x), @intCast(y) };
            counts[freq] += 1;

            for (0..count) |i| {
                const px = frequencies[freq][i][0];
                const py = frequencies[freq][i][1];
                const dx = cast(i8, x) - cast(i8, px);
                const dy = cast(i8, y) - cast(i8, py);
                const nx1 = cast(i8, x) + dx;
                const ny1 = cast(i8, y) + dy;
                const nx2 = cast(i8, px) - dx;
                const ny2 = cast(i8, py) - dy;

                if (nx1 >= 0 and nx1 < 50 and ny1 >= 0 and ny1 < 50)
                    antinodes[@intCast(ny1)] |= @as(u64, 1) << @intCast(nx1);
                if (nx2 >= 0 and nx2 < 50 and ny2 >= 0 and ny2 < 50)
                    antinodes[@intCast(ny2)] |= @as(u64, 1) << @intCast(nx2);
            }
        }
    }

    var count: u32 = 0;
    for (antinodes) |row| count += @popCount(row);
    return count;
}

noinline fn @"9"() !u64 {
    const S = struct {
        inline fn checksumContrib(id: u64, pos: u64, len: u64) u64 {
            // id * sum_{j=pos}^{pos+len-1} j (x2 here; divided by 2 later)
            return id * len * (2 * pos + len - 1);
        }
    };
    var sum: u64 = 0;
    const input = @embedFile("9.txt")[0 .. @embedFile("9.txt").len - 1];
    var pos: u64 = 0;
    var lo: u64 = 0;
    var hi: u64 = input.len / 2;
    var hi_rem: u64 = input[hi * 2] - '0';
    outer: while (true) {
        var gap: u64 = input[lo * 2 + 1] - '0';
        const lo_len = input[lo * 2] - '0';
        sum += S.checksumContrib(lo, pos, lo_len);
        pos += lo_len;
        lo += 1;
        while (gap != 0) {
            if (gap <= hi_rem) {
                @branchHint(.likely);
                sum += S.checksumContrib(hi, pos, gap);
                pos += gap;
                hi_rem -= gap;
                continue :outer;
            }
            sum += S.checksumContrib(hi, pos, hi_rem);
            pos += hi_rem;
            gap -= hi_rem;
            hi -= 1;
            if (hi <= lo) break :outer;
            hi_rem = input[hi * 2] - '0';
        }
    }
    return sum / 2;
}

noinline fn @"10"() !u64 {
    const input = @embedFile("10.txt");
    const stride: usize = 1 + indexOfScalar(u8, input, '\n').?;
    var count: u32 = 0;

    for (0..input.len) |start| {
        if (input[start] != '9') continue;

        var seen: __u16x16 = @splat(0xFFFF);
        var seen_len: u8 = 0;
        var stack: [64]u16 = undefined;
        var stack_len: u16 = 1;
        stack[0] = @intCast(start);

        while (stack_len > 0) {
            stack_len -= 1;
            const pos = stack[stack_len];
            const c = input[pos];

            if (c == '0') {
                const pos_vec: __u16x16 = @splat(pos);
                if (@reduce(.And, seen != pos_vec)) {
                    seen[seen_len] = pos;
                    seen_len += 1;
                }
                continue;
            }
            const target = c - 1;

            for ([_]isize{ 1, -1, @intCast(stride), -cast(isize, stride) }) |dir| {
                const next_pos = pos + dir;
                if (next_pos >= 0 and next_pos < input.len and input[@intCast(next_pos)] == target) {
                    stack[stack_len] = @intCast(next_pos);
                    stack_len += 1;
                }
            }
        }
        count += seen_len;
    }
    return count;
}

noinline fn @"11"() !u64 {
    const input = @embedFile("11.txt")[0 .. @embedFile("11.txt").len - 1];
    const MEMO_SIZE = 100;

    const S = struct {
        fn calc(stone: u64, blinks: u8, m: *[26][MEMO_SIZE]u32) u32 {
            if (blinks == 0) return 1;
            if (stone < MEMO_SIZE and m[blinks][stone] != 0) return m[blinks][stone];
            const result = if (stone == 0) calc(1, blinks - 1, m) else blk: {
                const digits = log10_int(stone) + 1;
                if (digits % 2 == 0) {
                    const divisor = pow(u64, 10, @intCast(digits / 2));
                    break :blk calc(stone / divisor, blinks - 1, m) + calc(stone % divisor, blinks - 1, m);
                } else {
                    break :blk calc(stone * 2024, blinks - 1, m);
                }
            };
            if (stone < MEMO_SIZE) m[blinks][stone] = result;
            return result;
        }
    };

    var memo = comptime blk: {
        @setEvalBranchQuota(maxInt(u32));
        var table: [26][MEMO_SIZE]u32 = @splat(@splat(0));
        for (0..25) |b| {
            for (0..MEMO_SIZE) |s| {
                _ = S.calc(s, b, &table);
            }
        }
        break :blk table;
    };

    var total: u32 = 0;
    var i: usize = 0;
    while (i < input.len) {
        while (input[i] == ' ') i += 1;
        var stone: u64 = 0;
        while (i < input.len and isDigit(input[i])) : (i += 1) stone = stone * 10 + (input[i] - '0');
        total += S.calc(stone, 25, &memo);
    }
    return total;
}

noinline fn @"12"() !u64 {
    const input = @embedFile("12.txt");
    const width: usize = 140;
    const stride = width + 1;
    var visited: [width * width]bool = @splat(false);
    var stack: [128][2]u16 = undefined;
    var total: u32 = 0;

    for (0..width) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            if (visited[idx]) continue;

            const plant = input[y * stride + x];
            var area: u32 = 0;
            var perimeter: u32 = 0;
            var sp: u32 = 0;
            stack[sp] = .{ @intCast(x), @intCast(y) };
            sp += 1;
            visited[idx] = true;

            while (sp > 0) {
                sp -= 1;
                const cx, const cy = stack[sp];
                area += 1;
                for ([4][2]i8{ .{ 1, 0 }, .{ 0, 1 }, .{ -1, 0 }, .{ 0, -1 } }) |d| {
                    const nx = @as(i32, cx) + d[0];
                    const ny = @as(i32, cy) + d[1];
                    if (nx < 0 or ny < 0) {
                        perimeter += 1;
                        continue;
                    }
                    const ux = cast(u16, nx);
                    const uy = cast(u16, ny);
                    if (input[uy * stride + ux] != plant) {
                        perimeter += 1;
                        continue;
                    }
                    const nidx = uy * width + ux;
                    if (!visited[nidx]) {
                        visited[nidx] = true;
                        stack[sp] = .{ ux, uy };
                        sp += 1;
                    }
                }
            }
            total += area * perimeter;
        }
    }
    return total;
}

noinline fn @"13"() !u64 {
    const input = @embedFile("13.txt");
    var sum_a: i32 = 0;
    var sum_b: i32 = 0;
    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        var nums: [6]i32 = undefined;
        for (&nums) |*num| {
            while (!isDigit(input[i])) i += 1;
            var n: i32 = 0;
            while (isDigit(input[i])) : (i += 1) n = n * 10 + (input[i] - '0');
            num.* = n;
        }
        // cramer's rule!
        const det = nums[0] * nums[3] - nums[1] * nums[2];
        if (det != 0) {
            const num_a = nums[4] * nums[3] - nums[5] * nums[2];
            const num_b = nums[0] * nums[5] - nums[1] * nums[4];
            if (@rem(num_a, det) == 0 and @rem(num_b, det) == 0) {
                sum_a += @intCast(@divExact(num_a, det));
                sum_b += @intCast(@divExact(num_b, det));
            }
        }
    }
    return @intCast(sum_a * 3 + sum_b);
}

noinline fn @"14"() !u64 {
    const input = @embedFile("14.txt");
    const width = 101;
    const height = 103;

    // four 16-bit counters
    var counts_packed: u64 = 0;

    var pos: usize = 0;
    while (pos < input.len) : (pos += 1) { // skip "\n"
        // parse px
        pos += 2; // skip "p="
        var px: u32 = input[pos] - '0';
        pos += 1;
        inline for (0..2) |_| {
            if (input[pos] == ',') break;
            px = px * 10 + (input[pos] - '0');
            pos += 1;
        }

        // parse py
        pos += 1; // skip ","
        var py: u32 = input[pos] - '0';
        pos += 1;
        inline for (0..2) |_| {
            if (input[pos] == ' ') break;
            py = py * 10 + (input[pos] - '0');
            pos += 1;
        }

        // parse vx
        pos += 3; // skip " v="
        const vx_neg = input[pos] == '-';
        if (vx_neg) pos += 1;
        var vx_abs: u32 = input[pos] - '0';
        pos += 1;
        inline for (0..2) |_| {
            if (input[pos] == ',') break;
            vx_abs = vx_abs * 10 + (input[pos] - '0');
            pos += 1;
        }
        const vx_u: u32 = if (vx_neg) (width - vx_abs) else vx_abs;

        // parse vy
        pos += 1; // skip ","
        const vy_neg = input[pos] == '-';
        if (vy_neg) pos += 1;
        var vy_abs: u32 = input[pos] - '0';
        pos += 1;
        inline for (0..2) |_| {
            if (input[pos] < '0') break;
            vy_abs = vy_abs * 10 + (input[pos] - '0');
            pos += 1;
        }
        const vy_u: u32 = if (vy_neg) (height - vy_abs) else vy_abs;

        const x: u32 = (px + 100 * vx_u) % width;
        const y: u32 = (py + 100 * vy_u) % height;

        // bump one of the counters
        if (x != 50 and y != 51) {
            const i: u32 = @as(u32, @intFromBool(x > 50)) * 2 + @intFromBool(y > 51);
            counts_packed += @as(u64, 1) << @intCast(i * 16);
        }
    }

    var count: u32 = 1;
    inline for (0..4) |i| count *= @intCast(counts_packed >> i * 16 & 0xFFFF);

    return count;
}

noinline fn @"15"() !u64 {
    const input = @embedFile("15.txt");
    var count: u32 = 0;
    const stride: usize = 51;

    var map: [2560]u8 = input[0..2560].*;
    var pos: usize = 24 * stride + 24;
    map[pos] = '.';

    const dir_lut = comptime blk: {
        var table: [256]isize = @splat(0);
        table['>'] = 1;
        table['v'] = stride;
        table['<'] = -1;
        table['^'] = -@as(isize, stride);
        break :blk table;
    };

    var ip: usize = 2561;
    while (ip < input.len) : (ip += 1) {
        if (input[ip] == '\n') continue;
        const next_pos = cast(usize, cast(isize, pos) + dir_lut[input[ip]]);
        switch (map[next_pos]) {
            '.' => pos = next_pos,
            '#' => {},
            'O' => {
                var block_pos = next_pos;
                while (map[block_pos] == 'O') {
                    block_pos = cast(usize, cast(isize, block_pos) + dir_lut[input[ip]]);
                }
                if (map[block_pos] == '.') {
                    map[next_pos] = '.';
                    map[block_pos] = 'O';
                    pos = next_pos;
                }
            },
            else => unreachable,
        }
    }
    for (0..50) |y| {
        for (0..50) |x| {
            if (map[y * stride + x] == 'O') {
                count += cast(u32, (y * 100 + x));
            }
        }
    }
    return count;
}

var day17_buf: [17]u8 = @splat(',');
noinline fn @"17"() ![]u8 {
    // lol
    //  mov    al,0x33
    //  mov    al,0x2c
    //  mov    al,0x35
    //  mov    al,0x2c
    //  mov    al,0x30
    //  mov    al,0x2c
    //  mov    al,0x31
    //  mov    al,0x2c
    //  mov    al,0x35
    //  mov    al,0x2c
    //  mov    al,0x31
    //  mov    al,0x2c
    //  mov    al,0x35
    //  mov    al,0x2c
    //  mov    al,0x31
    //  mov    al,0x2c
    //  mov    al,0x30
    var a: u32 = 60589763; // Register A
    const p3: u32 = 5; // Program[3]
    const p7: u32 = 6; // Program[7]

    var pos: usize = 0;
    while (a != 0) {
        const b = (a & 7) ^ p3;
        day17_buf[pos] = cast(u8, ((a >> @intCast(b)) ^ b ^ p7) & 7) + '0';
        a >>= 3;
        pos += 2;
    }
    return day17_buf[0 .. pos - 1];
}

noinline fn @"18"() !u64 {
    const input = @embedFile("18.txt");
    var blocked: [71][71]bool = @splat(@splat(false));
    var pos: usize = 0;

    for (0..1024) |_| {
        var x: u8, var y: u8 = .{ 0, 0 };
        while (input[pos] != ',') : (pos += 1) x = x * 10 + input[pos] - '0';
        pos += 1;
        while (input[pos] != '\n') : (pos += 1) y = y * 10 + input[pos] - '0';
        pos += 1;
        blocked[y][x] = true;
    }

    var queue: [71 * 71][2]u8 = undefined;
    var dist: [71][71]u32 = @splat(@splat(maxInt(u32)));
    var head: u32 = 0;
    var tail: u32 = 1;
    queue[0] = .{ 0, 0 };
    dist[0][0] = 0;

    while (head < tail) {
        const x, const y = queue[head];
        head += 1;
        for ([_][2]i8{ .{ 0, 1 }, .{ 1, 0 }, .{ 0, -1 }, .{ -1, 0 } }) |d| {
            const nx = cast(i8, x) + d[0];
            const ny = cast(i8, y) + d[1];
            if (nx < 0 or nx >= 71 or ny < 0 or ny >= 71) continue;
            const ux = cast(u8, nx);
            const uy = cast(u8, ny);
            if (blocked[uy][ux] or dist[uy][ux] <= dist[y][x] + 1) continue;
            dist[uy][ux] = dist[y][x] + 1;
            queue[tail] = .{ ux, uy };
            tail += 1;
        }
    }
    return dist[70][70];
}

// noinline fn @"25"() !u64 {
//     const input = @embedFile("25.txt");
//     var locks: [250]u32 = undefined;
//     var keys: [250]u32 = undefined;
//     var lock_count: u32 = 0;
//     var key_count: u32 = 0;
//     var pos: u32 = 4;

//     while (pos < input.len - 32) : (pos += 43) {
//         const chunk: __u8x32 = input[pos..][0..32].*;
//         const cmp = chunk == @as(__u8x32, @splat('#'));
//         const mask = bitcast(u32, cmp);

//         if (input[pos] == '#') {
//             locks[lock_count] = mask;
//             lock_count += 1;
//         } else {
//             keys[key_count] = mask;
//             key_count += 1;
//         }
//     }
//     var count: u32 = 0;
//     for (locks[0..lock_count]) |lock| {
//         for (keys[0..key_count]) |key| {
//             count += @intFromBool(lock & key == 0);
//         }
//     }
//     return count;
// }

noinline fn @"25"() !u64 {
    const input = @embedFile("25.txt");
    const S = struct {
        var locks: [256]u32 align(32) = @splat(~@as(u32, 0));
        var keys: [256]u32 = undefined;
    };

    var nl: usize = 0;
    var nk: usize = 0;

    var ptr = input.ptr + 4;
    const end = input.ptr + input.len - 32;

    while (@intFromPtr(ptr) <= @intFromPtr(end)) : (ptr += 43) {
        const chunk: __u8x32 = @bitCast(ptr[0..32].*);
        const mask: u32 = @bitCast(chunk == @as(__u8x32, @splat('#')));
        if (chunk[0] == '#') {
            S.locks[nl] = mask;
            nl += 1;
        } else {
            S.keys[nk] = mask;
            nk += 1;
        }
    }

    var count: __u32x8 = @splat(0);
    const zero: __u32x8 = @splat(0);
    const one: __u32x8 = @splat(1);
    const lv = @as([*]const __u32x8, @ptrCast(&S.locks))[0..32];

    inline for (.{ lv[0..11], lv[11..22], lv[22..32] }) |batch| {
        for (S.keys[0..nk]) |key| {
            const k: __u32x8 = @splat(key);
            inline for (batch) |v| count += @select(u32, (v & k) == zero, one, zero);
        }
    }

    return @reduce(.Add, count);
}

//
