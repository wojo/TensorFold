//! MTP drafts with the prompt cache (one stream): which rows the MTP head absorbs from each prompt chunk, where it must
//! stand when a prompt state is kept or a prompt goes on from one, and TF_GLM_MTP_DIGEST's hash of its state.
//!
//! The head absorbs one row behind the target. Its row i pairs the target's normed hidden row i with token i + 1, so a
//! target chunk [s, e) feeds head rows [s - 1, e - 1) with tokens [s, e): row s - 1 from the tail (the target's row
//! s - 1, held as the head's pending row since the chunk before), the rest from the chunk's rows 0 .. e - s - 2; the
//! chunk's last row becomes the tail. The prompt's first chunk has no row before it: rows [0, e - 1), tokens [1, e).
//! A chunk never reads a token past its end, so after any chunk ending at k the head stands at k - 1 rows with row
//! k - 1 pending, whatever follows: a kept state at k holds exactly that (cuda_snapshot `withMtp`), and a resume from
//! k does what the run that kept it did next. At a prompt's end the head stands at n - 1 with row n - 1 pending, which
//! the first draft round absorbs with the first token (cuda_mtp.draftMany, `prefill_last`).
//! Generic over the head (`cursor`, `slot`, `absorb`) so host tests drive a fake one.
const std = @import("std");
const snapshot = @import("cuda_snapshot.zig");
const G = @import("cuda_geometry.zig").Geometry;

/// What target chunk [start, start + rows) of `prompt` feeds the head: these tokens, the first paired with the tail
/// row when `tail` (else every one with the chunk's rows from its first).
pub const Feed = struct { tokens: []const u32, tail: bool };

/// The head's rows for one chunk, its cursor checked (`len` absorbed rows, `drafted` speculative ones of them).
pub fn plan(prompt: []const u32, start: usize, rows: usize, len: usize, drafted: usize) !Feed {
    if (rows == 0 or start + rows > prompt.len) return error.BadMtpRows;
    // the head stands one row behind the chunk's start, nothing drafted (a prompt pass rolls drafts back first)
    if (drafted != 0 or len + 1 != @max(start, 1)) return error.BadMtpCursor;
    if (start == 0) return .{ .tokens = prompt[1..rows], .tail = false };
    return .{ .tokens = prompt[start .. start + rows], .tail = true };
}

/// One target chunk into the head: `hidden` the chunk's normed rows, `last` its last row (the next tail).
pub fn feed(h: anytype, e: anytype, prompt: []const u32, start: usize, rows: usize, hidden: u64, last: u64) !void {
    const f = try plan(prompt, start, rows, h.cursor.len, h.cursor.drafted);
    try h.absorb(e, f.tokens, hidden, last, f.tail);
}

/// Whether the head stands where a prompt state of `slot` at `at` tokens leaves it: `at` - 1 rows, none drafted.
pub fn stands(h: anytype, slot: usize, at: usize) bool {
    return at > 0 and h.slot == slot and h.cursor.drafted == 0 and h.cursor.len == at - 1;
}

/// A prompt pass of `slot` from `at` (0: from the start, the head cleared; else a kept or live state it must stand at).
pub fn enter(h: anytype, slot: usize, at: usize) !void {
    if (at == 0) {
        h.cursor = .{};
        h.slot = slot;
        return;
    }
    if (!stands(h, slot, at)) return error.MtpNotAtResume;
}

/// A state is kept at `at`: the head must stand there (a drafted row or another sequence's head is a bug, not a refusal).
pub fn keepable(h: anytype, slot: usize, at: usize) !void {
    if (!stands(h, slot, at)) return error.BadMtpCursor;
}

/// The head after its section was put back from a state kept at `at`.
pub fn restored(h: anytype, slot: usize, at: usize) void {
    h.cursor = .{ .len = at -| 1 };
    h.slot = slot;
}

/// How TF_GLM_MTP_DIGEST reads device bytes to the host (the engine: a copy on its stream, then a wait).
pub const Reader = struct {
    ptr: *anyopaque,
    read: *const fn (ptr: *anyopaque, dst: []u8, src: u64) anyerror!void,
};

/// TF_GLM_MTP_DIGEST: a hash of what the head's next pass reads of its state at `len` rows: latent rows [0, len), the
/// complete pools [0, len / 4), the index rings' rows of the last min(len, ring) positions (a ring row is a position
/// modulo the ring) and the pending row. Bytes no pass reads (rows past `len`, a ring's older rows) are left out, so
/// a cold run and a resumed one digest alike whatever an earlier request left there.
pub fn digest(r: Reader, m: snapshot.MtpPlane, len: usize, scratch: []u8) !u64 {
    if (scratch.len == 0) return error.NoScratch;
    var h = std.hash.Wyhash.init(0);
    h.update(std.mem.asBytes(&@as(u64, len)));
    try hashRange(r, &h, m.latent, len * G.latent_row, scratch);
    try hashRange(r, &h, m.pooled, len / 4 * G.pool_row, scratch);
    const live = @min(len, m.ring);
    const first = len - live;
    for ([_]u64{ m.keys, m.gates }) |plane| {
        // positions first .. len in order: at most two runs of ring rows
        var p = first;
        while (p < len) {
            const slot = p % m.ring;
            const n = @min(len - p, m.ring - slot);
            try hashRange(r, &h, plane + slot * 256, n * 256, scratch);
            p += n;
        }
    }
    try hashRange(r, &h, m.pending, snapshot.mtp_tail_bytes, scratch);
    return h.final();
}

fn hashRange(r: Reader, h: *std.hash.Wyhash, src: u64, bytes: usize, scratch: []u8) !void {
    var off: usize = 0;
    while (off < bytes) {
        const n = @min(scratch.len, bytes - off);
        try r.read(r.ptr, scratch[0..n], src + off);
        h.update(scratch[0..n]);
        off += n;
    }
}

// -- tests: a head that records what it absorbs ----------------------------------------------------------------------

const Cursor = struct {
    len: usize = 0,
    drafted: usize = 0,
};

/// Each absorbed head row: its position, its token and the target row its hidden input came from.
const Row = struct { row: usize, token: u32, hidden: usize };

/// Hidden rows are target row numbers here: a chunk's `hidden` is its first row, `last` its last.
const FakeHead = struct {
    cursor: Cursor = .{},
    slot: ?usize = null,
    pending: ?usize = null,
    log: std.ArrayList(Row) = .empty,
    fn deinit(h: *FakeHead) void {
        h.log.deinit(std.testing.allocator);
    }
    fn absorb(h: *FakeHead, _: void, tokens: []const u32, hidden: u64, last: u64, tail: bool) !void {
        for (tokens, 0..) |t, i| {
            const from: usize = if (tail) (if (i == 0) h.pending orelse return error.NoTail else @as(usize, @intCast(hidden)) + i - 1) else @as(usize, @intCast(hidden)) + i;
            try h.log.append(std.testing.allocator, .{ .row = h.cursor.len + i, .token = t, .hidden = from });
        }
        h.cursor.len += tokens.len;
        h.pending = @intCast(last);
    }
};

/// A prefill pass as cuda_engine.prefillWith runs one: chunks of `limit` rows of `prompt` from `begin` to its end, each
/// fed to the head after the target ran it.
fn pass(h: *FakeHead, prompt: []const u32, begin: usize, limit: usize) !void {
    try enter(h, 0, begin);
    var s = begin;
    while (s < prompt.len) {
        const rows = @min(limit, prompt.len - s);
        try feed(h, {}, prompt, s, rows, s, s + rows - 1);
        s += rows;
    }
}

fn expectWhole(h: *const FakeHead, prompt: []const u32) !void {
    // every head row [0, n - 1) once, in order, paired with its own target row and the token after it
    try std.testing.expectEqual(prompt.len - 1, h.log.items.len);
    for (h.log.items, 0..) |r, i| {
        try std.testing.expectEqual(i, r.row);
        try std.testing.expectEqual(i, r.hidden);
        try std.testing.expectEqual(prompt[i + 1], r.token);
    }
    try std.testing.expectEqual(prompt.len - 1, h.cursor.len);
    try std.testing.expectEqual(@as(?usize, prompt.len - 1), h.pending);
}

fn testPrompt(buf: []u32, seed: u32) []u32 {
    for (buf, 0..) |*t, i| t.* = @intCast((i * 7919 + seed) % 154880);
    return buf;
}

test "MTP head: a cold prompt absorbs every row once, one behind the target, whatever the chunks" {
    var buf: [700]u32 = undefined;
    const prompt = testPrompt(&buf, 3);
    for ([_]usize{ 64, 128, 256, 2048 }) |limit| for ([_]usize{ 1, 2, 64, 65, 127, 128, 129, 700 }) |n| {
        var h: FakeHead = .{};
        defer h.deinit();
        try pass(&h, prompt[0..n], 0, limit);
        try expectWhole(&h, prompt[0..n]);
    };
}

test "MTP head: a state kept at a mark and resumed absorbs what the run that kept it did, each row once" {
    var buf: [900]u32 = undefined;
    const prompt = testPrompt(&buf, 11);
    var other_buf: [900]u32 = undefined;
    const other = testPrompt(&other_buf, 12);
    @memcpy(other[0..640], prompt[0..640]);
    for ([_]usize{ 64, 128, 2048 }) |limit| for ([_]usize{ 64, 128, 320, 640 }) |m| {
        // the lanes' pass: the prompt up to the mark (its state kept there), then the rest from it
        var a: FakeHead = .{};
        defer a.deinit();
        try pass(&a, prompt[0..m], 0, limit);
        try keepable(&a, 0, m);
        const kept = .{ a.cursor, a.pending.? };
        try pass(&a, prompt, m, limit);
        try expectWhole(&a, prompt);
        // the same rows from the state kept at m, on a head that ran something else meanwhile
        var b: FakeHead = .{};
        defer b.deinit();
        try pass(&b, other[0..50], 0, limit);
        b.log.clearRetainingCapacity();
        restored(&b, 0, m);
        try std.testing.expectEqual(kept[0].len, b.cursor.len);
        b.pending = kept[1];
        try pass(&b, prompt, m, limit);
        try std.testing.expectEqualSlices(Row, a.log.items[m - 1 ..], b.log.items);
        // another prompt with the kept prefix: its own token pairs with the kept tail row
        var c: FakeHead = .{};
        defer c.deinit();
        restored(&c, 0, m);
        c.pending = kept[1];
        try pass(&c, other, m, limit);
        try std.testing.expectEqual(Row{ .row = m - 1, .token = other[m], .hidden = m - 1 }, c.log.items[0]);
        try std.testing.expectEqual(other.len - m, c.log.items.len);
    };
}

test "MTP head: a refused save goes on from the live head; a head that moved, drafted or is another stream's is refused" {
    var buf: [300]u32 = undefined;
    const prompt = testPrompt(&buf, 5);
    var a: FakeHead = .{};
    defer a.deinit();
    try pass(&a, prompt[0..128], 0, 64);
    // the save was refused: no state, the next segment goes on from the rows the head holds (live_from)
    try pass(&a, prompt, 128, 64);
    try expectWhole(&a, prompt);
    // a head not where the live rows are cannot go on from them
    var b: FakeHead = .{};
    defer b.deinit();
    try pass(&b, prompt[0..128], 0, 64);
    b.cursor.drafted = 1;
    try std.testing.expectError(error.MtpNotAtResume, enter(&b, 0, 128));
    try std.testing.expectError(error.BadMtpCursor, keepable(&b, 0, 128));
    b.cursor.drafted = 0;
    try std.testing.expectError(error.MtpNotAtResume, enter(&b, 1, 128));
    try std.testing.expectError(error.MtpNotAtResume, enter(&b, 0, 192));
    try enter(&b, 0, 128);
    // a chunk out of the head's order is refused, not absorbed at the wrong rows
    try std.testing.expectError(error.BadMtpCursor, feed(&b, {}, prompt, 192, 64, 192, 255));
    try std.testing.expectError(error.BadMtpCursor, feed(&b, {}, prompt, 0, 64, 0, 63));
}

test "MTP digest: what the next pass reads, not rows past the head's length or a ring's older rows" {
    const gpa = std.testing.allocator;
    const cap = 512;
    const ring = 64;
    const Host = struct {
        fn read(_: *anyopaque, dst: []u8, src: u64) anyerror!void {
            const s: [*]const u8 = @ptrFromInt(src);
            @memcpy(dst, s[0..dst.len]);
        }
    };
    const latent = try gpa.alloc(u8, cap * G.latent_row);
    defer gpa.free(latent);
    const pooled = try gpa.alloc(u8, (cap / 4 + 2) * G.pool_row);
    defer gpa.free(pooled);
    const keys = try gpa.alloc(u8, ring * 256);
    defer gpa.free(keys);
    const gates = try gpa.alloc(u8, ring * 256);
    defer gpa.free(gates);
    const pending = try gpa.alloc(u8, snapshot.mtp_tail_bytes);
    defer gpa.free(pending);
    var prng = std.Random.DefaultPrng.init(9);
    for ([_][]u8{ latent, pooled, keys, gates, pending }) |b| prng.random().bytes(b);
    const m: snapshot.MtpPlane = .{ .latent = @intFromPtr(latent.ptr), .pooled = @intFromPtr(pooled.ptr), .keys = @intFromPtr(keys.ptr), .gates = @intFromPtr(gates.ptr), .ring = ring, .pending = @intFromPtr(pending.ptr) };
    var scratch: [1000]u8 = undefined;
    const r: Reader = .{ .ptr = undefined, .read = Host.read };
    for ([_]usize{ 30, 127, 200 }) |len| {
        const d = try digest(r, m, len, &scratch);
        try std.testing.expectEqual(d, try digest(r, m, len, scratch[0..7]));
        // bytes no pass reads: the latent row at len, the pool still filling, a ring row of an older position
        latent[len * G.latent_row] +%= 1;
        pooled[(len / 4) * G.pool_row] +%= 1;
        if (len < ring) keys[(ring - 1) * 256] +%= 1;
        try std.testing.expectEqual(d, try digest(r, m, len, &scratch));
        // bytes it does: a latent row, a complete pool, the ring row of the last position, the pending row
        inline for (.{ .{ latent, (len - 1) * G.latent_row }, .{ pooled, (len / 4 - 1) * G.pool_row }, .{ gates, ((len - 1) % ring) * 256 }, .{ pending, 8191 } }) |x| {
            x[0][x[1]] +%= 1;
            try std.testing.expect(d != try digest(r, m, len, &scratch));
            x[0][x[1]] -%= 1;
        }
        try std.testing.expect(d != try digest(r, m, len + 1, &scratch));
    }
}

const prefill_loop = @import("cuda_prefill_loop.zig");

/// A head state kept at a mark: its cursor and its pending row.
const Kept = struct { cursor: Cursor, pending: usize };

/// A prompt pass as cuda_engine's Fill runs it (TF_GLM_FILL's fills and prefillWith alike: startPass, then passChunk
/// until done): the head entered at `begin`, the plan's chunks (cuda_prefill_loop.Plan / Cursor) fed in order with the
/// fill's prompt position starting at `begin` (Fill.prompt_at), the head checked after the keep point's run, as
/// takeSnap checks it. Returns the head's state at the keep point, if the plan keeps.
fn fillPass(h: *FakeHead, prompt: []const u32, begin: usize, keep_at: ?usize, limit: usize) !?Kept {
    try enter(h, 0, begin);
    const p = prefill_loop.Plan.of(prompt.len, begin, keep_at);
    var cur: prefill_loop.Cursor = .{};
    var at = begin;
    var kept: ?Kept = null;
    while (try cur.next(&p, limit)) |ch| {
        // promptTaps after the target ran the chunk: e.prompt_at is the fill's, the chunk's own start
        try std.testing.expectEqual(at, ch.pos);
        try feed(h, {}, prompt, at, ch.rows, ch.pos, ch.pos + ch.rows - 1);
        at += ch.rows;
        if (!cur.ran(&p, ch)) continue;
        if (p.keep and ch.run == 0) {
            try keepable(h, 0, p.runs[0].end);
            kept = .{ .cursor = h.cursor, .pending = h.pending.? };
        }
    }
    try std.testing.expectEqual(prompt.len, at);
    return kept;
}

test "MTP head through a Fill's plan (prefillWith = startPass + passChunk): cache off, every row once; a keep point, the lanes' mark segments and a resume absorb what one cold pass does" {
    var buf: [5021]u32 = undefined;
    const prompt = testPrompt(&buf, 17);
    // cache off: one pass from 0, cut at the 64-grid point below the end (2048, 2048, 896, then 29 at 2048 a chunk)
    for ([_]usize{ 128, 2048 }) |limit| for ([_]usize{ 1, 2, 63, 64, 65, 2047, 2049, 5021 }) |n| {
        var h: FakeHead = .{};
        defer h.deinit();
        try std.testing.expect(try fillPass(&h, prompt[0..n], 0, null, limit) == null);
        try expectWhole(&h, prompt[0..n]);
    };
    for ([_]usize{ 128, 2048 }) |limit| {
        // one pass keeping at 2112 on its way: the state there is the head after the run to 2112
        var a: FakeHead = .{};
        defer a.deinit();
        const k = (try fillPass(&a, prompt, 0, 2112, limit)).?;
        try expectWhole(&a, prompt);
        try std.testing.expectEqual(@as(usize, 2111), k.cursor.len);
        try std.testing.expectEqual(@as(usize, 2111), k.pending);
        // the lanes' pass with marks 64 and 2112: a segment per mark (prefillWith(ids[0..m], keep_at = m) from the
        // last one, live rows standing), then the rest: the same rows as one cold pass
        var b: FakeHead = .{};
        defer b.deinit();
        const k64 = (try fillPass(&b, prompt[0..64], 0, 64, limit)).?;
        const k2112 = (try fillPass(&b, prompt[0..2112], 64, 2112, limit)).?;
        try std.testing.expect(try fillPass(&b, prompt, 2112, null, limit) == null);
        try std.testing.expectEqualSlices(Row, a.log.items, b.log.items);
        try std.testing.expectEqual(k, k2112);
        // resumed from each kept state on a head that ran another prompt since: what the cold pass did next
        for ([_]Kept{ k64, k2112 }, [_]usize{ 64, 2112 }) |kk, m| {
            var c: FakeHead = .{};
            defer c.deinit();
            _ = try fillPass(&c, prompt[100..300], 0, null, limit);
            c.log.clearRetainingCapacity();
            restored(&c, 0, m);
            try std.testing.expectEqual(kk.cursor, c.cursor);
            c.pending = kk.pending;
            try std.testing.expect(try fillPass(&c, prompt, m, null, limit) == null);
            try std.testing.expectEqualSlices(Row, a.log.items[m - 1 ..], c.log.items);
        }
    }
}
