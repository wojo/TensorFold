//! A kept prompt state of the GLM CUDA engine (the recipe's decode.Snapshot / take_snapshot / save_rows / restore):
//! one device buffer that holds a sequence's state after `at` prompt tokens, so a later request of any slot resumes
//! it and prefills the rest with the bits a cold prefill gives.
//!
//! What the state is (glm5_next/cuda/decode.py `_row_views`, `_ring_tail`, `restore`; multi.py `draft_window`):
//!   latent   the 11 MLA layers' fp8 latent rows [0, at)                      (G.latent_row bytes a row)
//!   pooled   their indexer pools [0, at/4 + 1)                               (G.pool_row bytes a row)
//!   rings    their indexer key and gate rings, whole (the recipe keeps only the n % 4 tail; the whole ring is a
//!            superset a few MiB big, and the grid keeps `at` a multiple of 64, so the tail is empty anyway)
//!   rec      the 34 KDA layers' recurrent states, the parity each layer's chunk wrote last
//!   conv     the 34 layers' conv windows
//!   draft    DFlash2's per-slot K and V rings (5 layers), whole, when a drafter runs
//!   mtp      MTP drafts (`withMtp`): the head's one DSA plane one row short, latent rows [0, at - 1) and their pools,
//!            its two index rings whole (sized from the head's own ring), and the target's normed hidden row at - 1
//!            (the tail: the head absorbs one row behind the target, cuda_mtp_keep.zig)
//! Nothing else carries over a chunk boundary: the HC streams, the router, the selections are chunk-local, and
//! `pos_dev` is written by every chunk. Pure arithmetic and copies: tests run on host memory standing in for the
//! device (`Copier`).
//! TF_GLM_KV_SHARD: the latent rows are on the stream's owner only, so only that rank's buffer holds them
//! (`Sections.rows`); every other section is on both ranks (cuda_shard_keep.zig puts them back, or carries them).
const std = @import("std");
const state = @import("cuda_state.zig");
const G = state.Geometry;

/// How a snapshot copies bytes between device addresses (one stream, in order).
pub const Copier = struct {
    ptr: *anyopaque,
    copy: *const fn (ptr: *anyopaque, dst: u64, src: u64, bytes: usize) anyerror!void,

    fn go(c: Copier, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try c.copy(c.ptr, dst, src, bytes);
    }
};

/// DFlash2's K and V rings of one stream slot: `layers` each, `kv` heads, `ring` rows of 128 bf16 per head, in a
/// cache of `cap` rows per head (the slot's rows start at slot * ring).
pub const Draft = struct {
    kc: [5]u64,
    vc: [5]u64,
    kv: usize,
    ring: usize,
    cap: usize,
    slot: usize,

    pub fn bytes(d: Draft) usize {
        return 10 * d.kv * d.ring * 128 * 2;
    }
};

pub const Sections = struct {
    at: usize,
    ring: usize,
    latent: usize,
    pooled: usize,
    rings: usize,
    rec: usize,
    conv: usize,
    draft: usize,
    total: usize,
    /// whether the buffer holds the latent rows (TF_GLM_KV_SHARD: on the rank that owned the stream's extent only)
    rows: bool = true,
    /// MTP drafts (`withMtp`): the head's section at `mtp`, its rows (at - 1) and ring; null: no head section
    mtp_rows: ?usize = null,
    mtp_ring: usize = 0,
    mtp: usize = 0,

    /// A latent plane's rows of the state (the latent section holds 11 of them when `rows`).
    pub fn latentBytes(s: Sections) usize {
        return s.at * G.latent_row;
    }
    pub fn pooledBytes(s: Sections) usize {
        return (s.at / 4 + 1) * G.pool_row;
    }
    pub fn ringBytes(s: Sections) usize {
        return s.ring * 128 * 2;
    }
    /// DFlash2's section (0 without a drafter): it ends where the head's begins, or at the end.
    pub fn draftBytes(s: Sections) usize {
        return (if (s.mtp_rows != null) s.mtp else s.total) - s.draft;
    }
    pub fn mtpLatentBytes(s: Sections) usize {
        return (s.mtp_rows orelse 0) * G.latent_row;
    }
    /// The head's complete pools and the row of the one still filling (as `pooledBytes`; its tokens are in the rings).
    pub fn mtpPooledBytes(s: Sections) usize {
        return if (s.mtp_rows) |m| (m / 4 + 1) * G.pool_row else 0;
    }
    pub fn mtpRingBytes(s: Sections) usize {
        return s.mtp_ring * 128 * 2;
    }
    /// Offsets in the head's section: latent, pooled, keys, gates, tail.
    pub fn mtpParts(s: Sections) [5]usize {
        var o: [5]usize = undefined;
        var off = s.mtp;
        for ([_]usize{ s.mtpLatentBytes(), s.mtpPooledBytes(), s.mtpRingBytes(), s.mtpRingBytes(), mtp_tail_bytes }, 0..) |n, i| {
            o[i] = off;
            off += std.mem.alignForward(usize, n, 256);
        }
        return o;
    }
};

/// The target's normed hidden row the head absorbs next (4096 bf16).
pub const mtp_tail_bytes: usize = 8192;

/// The MTP head's live plane (cuda_mtp.Head: one DSA layer's latent rows, pools and index rings) and its pending row.
pub const MtpPlane = struct { latent: u64, pooled: u64, keys: u64, gates: u64, ring: usize, pending: u64 };

/// `s` with the MTP head's section after the others: its plane one row short of `s.at` (cuda_mtp_keep), `ring` the
/// head's own index ring (Caches.initMtp; it need not be the target's).
pub fn withMtp(s: Sections, ring: usize) Sections {
    var t = s;
    t.mtp_rows = s.at -| 1;
    t.mtp_ring = ring;
    t.mtp = s.total;
    var end = t.mtp;
    for ([_]usize{ t.mtpLatentBytes(), t.mtpPooledBytes(), t.mtpRingBytes(), t.mtpRingBytes(), mtp_tail_bytes }) |n| end += std.mem.alignForward(usize, n, 256);
    t.total = end;
    return t;
}

/// The head's plane and pending row into `buf`'s MTP section (out) or back from it.
pub fn mtpCopy(c: Copier, s: Sections, buf: u64, m: MtpPlane, out: bool) !void {
    if (s.mtp_rows == null) return error.NoMtpSection;
    if (m.ring != s.mtp_ring) return error.MtpRingMismatch;
    const o = s.mtpParts();
    const live = [_]u64{ m.latent, m.pooled, m.keys, m.gates, m.pending };
    const n = [_]usize{ s.mtpLatentBytes(), s.mtpPooledBytes(), s.mtpRingBytes(), s.mtpRingBytes(), mtp_tail_bytes };
    for (live, o, n) |l, off, bytes| if (out) try c.go(buf + off, l, bytes) else try c.go(l, buf + off, bytes);
}

pub fn recLayerBytes(g: G) usize {
    return g.lin_heads * 128 * 128 * 4;
}

pub fn convLayerBytes(g: G) usize {
    return 3 * g.convDim() * 2;
}

/// The buffer's sections for a state of `at` tokens; `draft_bytes` 0 without a drafter. Every section 256-aligned.
pub fn sections(g: G, at: usize, ring: usize, draft_bytes: usize) Sections {
    return sectionsOf(g, at, ring, draft_bytes, true);
}

/// `sections`, the latent rows held or not (`rows`; TF_GLM_KV_SHARD: false on the rank that did not own the stream).
pub fn sectionsOf(g: G, at: usize, ring: usize, draft_bytes: usize, rows: bool) Sections {
    var s: Sections = .{ .at = at, .ring = ring, .latent = 0, .pooled = 0, .rings = 0, .rec = 0, .conv = 0, .draft = 0, .total = 0, .rows = rows };
    var off: usize = 0;
    s.latent = off;
    if (rows) off += std.mem.alignForward(usize, G.dsa_layers * s.latentBytes(), 256);
    s.pooled = off;
    off += std.mem.alignForward(usize, G.dsa_layers * s.pooledBytes(), 256);
    s.rings = off;
    off += std.mem.alignForward(usize, G.dsa_layers * 2 * s.ringBytes(), 256);
    s.rec = off;
    off += std.mem.alignForward(usize, G.kda_layers * recLayerBytes(g), 256);
    s.conv = off;
    off += std.mem.alignForward(usize, G.kda_layers * convLayerBytes(g), 256);
    s.draft = off;
    off += std.mem.alignForward(usize, draft_bytes, 256);
    s.total = off;
    return s;
}

/// Copy a sequence's state out of its live caches into `buf` (a device buffer of `s.total` bytes): the latent rows
/// when the sections hold them (a view with none here cannot give them: NoLatentRows, before any copy).
pub fn save(c: Copier, g: G, s: Sections, buf: u64, st: *const state.State, draft: ?Draft) !void {
    if (s.rows and !st.cache.hasLatent()) return error.NoLatentRows;
    for (0..G.dsa_layers) |i| {
        if (s.rows) try c.go(buf + s.latent + i * s.latentBytes(), st.cache.latent[i], s.latentBytes());
        try c.go(buf + s.pooled + i * s.pooledBytes(), st.cache.pooled[i], s.pooledBytes());
        try c.go(buf + s.rings + (2 * i) * s.ringBytes(), st.cache.keys[i], s.ringBytes());
        try c.go(buf + s.rings + (2 * i + 1) * s.ringBytes(), st.cache.gates[i], s.ringBytes());
    }
    for (0..G.kda_layers) |l| {
        try c.go(buf + s.rec + l * recLayerBytes(g), st.recurrent(g, l, st.cur[l]), recLayerBytes(g));
        try c.go(buf + s.conv + l * convLayerBytes(g), st.fixed.conv + l * convLayerBytes(g), convLayerBytes(g));
    }
    if (draft) |d| try draftCopy(c, d, buf + s.draft, true);
}

/// Put a saved state into a sequence's caches. The recurrent states go to the parity each layer holds now. `rows`: the
/// latent rows too, which needs them in the buffer and the view (NoLatentRows, before any copy); TF_GLM_KV_SHARD puts
/// them back only on the rank that holds both (cuda_shard_keep.latentRole), and a carry brings them otherwise.
pub fn restore(c: Copier, g: G, s: Sections, buf: u64, st: *const state.State, draft: ?Draft, rows: bool) !void {
    if (rows and (!s.rows or !st.cache.hasLatent())) return error.NoLatentRows;
    for (0..G.dsa_layers) |i| {
        if (rows) try c.go(st.cache.latent[i], buf + s.latent + i * s.latentBytes(), s.latentBytes());
        try c.go(st.cache.pooled[i], buf + s.pooled + i * s.pooledBytes(), s.pooledBytes());
        try c.go(st.cache.keys[i], buf + s.rings + (2 * i) * s.ringBytes(), s.ringBytes());
        try c.go(st.cache.gates[i], buf + s.rings + (2 * i + 1) * s.ringBytes(), s.ringBytes());
    }
    for (0..G.kda_layers) |l| {
        try c.go(st.recurrent(g, l, st.cur[l]), buf + s.rec + l * recLayerBytes(g), recLayerBytes(g));
        try c.go(st.fixed.conv + l * convLayerBytes(g), buf + s.conv + l * convLayerBytes(g), convLayerBytes(g));
    }
    if (draft) |d| try draftCopy(c, d, buf + s.draft, false);
}

/// K then V of each drafter layer, head by head: the slot's ring rows of the cache <-> the buffer.
fn draftCopy(c: Copier, d: Draft, buf: u64, out: bool) !void {
    const row = 128 * 2;
    var at: u64 = buf;
    for (0..5) |i| for ([_]u64{ d.kc[i], d.vc[i] }) |base| for (0..d.kv) |h| {
        const live = base + ((h * d.cap + d.slot * d.ring) * row);
        if (out) try c.go(at, live, d.ring * row) else try c.go(live, at, d.ring * row);
        at += d.ring * row;
    };
}

// -- tests: host memory stands in for the device ---------------------------------------------------------------------

const HostCopy = struct {
    fn copy(_: *anyopaque, dst: u64, src: u64, n: usize) anyerror!void {
        const d: [*]u8 = @ptrFromInt(dst);
        const s: [*]const u8 = @ptrFromInt(src);
        @memcpy(d[0..n], s[0..n]);
    }
    fn copier() Copier {
        return .{ .ptr = undefined, .copy = copy };
    }
};

const Fake = struct {
    gpa: std.mem.Allocator,
    g: G,
    rec: []u8,
    conv: []u8,
    latent: [11][]u8,
    pooled: [11][]u8,
    keys: [11][]u8,
    gates: [11][]u8,
    kc: [5][]u8,
    vc: [5][]u8,
    rows: usize,
    ring: usize,
    kv: usize,
    cap: usize,
    st: state.State,

    fn init(gpa: std.mem.Allocator, rows: usize, ring: usize, slot: usize) !Fake {
        const g = try G.init(2, 0);
        var f: Fake = undefined;
        f.gpa = gpa;
        f.g = g;
        f.rows = rows;
        f.ring = ring;
        f.kv = 2;
        f.cap = 2 * 16 + 64;
        f.rec = try gpa.alloc(u8, 2 * 34 * recLayerBytes(g));
        f.conv = try gpa.alloc(u8, 34 * convLayerBytes(g));
        for (0..11) |i| {
            f.latent[i] = try gpa.alloc(u8, rows * G.latent_row);
            f.pooled[i] = try gpa.alloc(u8, (rows / 4 + 2) * G.pool_row);
            f.keys[i] = try gpa.alloc(u8, ring * 256);
            f.gates[i] = try gpa.alloc(u8, ring * 256);
        }
        for (0..5) |i| {
            f.kc[i] = try gpa.alloc(u8, f.kv * f.cap * 256);
            f.vc[i] = try gpa.alloc(u8, f.kv * f.cap * 256);
        }
        var cache: state.CacheView = .{ .latent = undefined, .pooled = undefined, .keys = undefined, .gates = undefined, .ring_base = 0, .capacity = rows };
        for (0..11) |i| {
            cache.latent[i] = @intFromPtr(f.latent[i].ptr);
            cache.pooled[i] = @intFromPtr(f.pooled[i].ptr);
            cache.keys[i] = @intFromPtr(f.keys[i].ptr);
            cache.gates[i] = @intFromPtr(f.gates[i].ptr);
        }
        f.st = .{ .slot = slot, .base = 0, .capacity = rows, .fixed = undefined, .cache = cache };
        f.st.fixed.rec = @intFromPtr(f.rec.ptr);
        f.st.fixed.conv = @intFromPtr(f.conv.ptr);
        return f;
    }
    fn deinit(f: *Fake) void {
        f.gpa.free(f.rec);
        f.gpa.free(f.conv);
        for (0..11) |i| {
            f.gpa.free(f.latent[i]);
            f.gpa.free(f.pooled[i]);
            f.gpa.free(f.keys[i]);
            f.gpa.free(f.gates[i]);
        }
        for (0..5) |i| {
            f.gpa.free(f.kc[i]);
            f.gpa.free(f.vc[i]);
        }
    }
    fn draft(f: *const Fake, slot: usize) Draft {
        var d: Draft = .{ .kc = undefined, .vc = undefined, .kv = f.kv, .ring = 16, .cap = f.cap, .slot = slot };
        for (0..5) |i| {
            d.kc[i] = @intFromPtr(f.kc[i].ptr);
            d.vc[i] = @intFromPtr(f.vc[i].ptr);
        }
        return d;
    }
    fn fill(f: *Fake, seed: u8) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        r.bytes(f.rec);
        r.bytes(f.conv);
        for (0..11) |i| {
            r.bytes(f.latent[i]);
            r.bytes(f.pooled[i]);
            r.bytes(f.keys[i]);
            r.bytes(f.gates[i]);
        }
        for (0..5) |i| {
            r.bytes(f.kc[i]);
            r.bytes(f.vc[i]);
        }
    }
};

test "a saved state restores the rows, rings, recurrent parity and drafter ring of its prefix, nothing past it" {
    const gpa = std.testing.allocator;
    var a = try Fake.init(gpa, 256, 64, 1);
    defer a.deinit();
    var b = try Fake.init(gpa, 256, 64, 0);
    defer b.deinit();
    a.fill(1);
    b.fill(2);
    // a holds odd parities on some layers; b's parities differ: the restore goes to b's own
    a.st.cur[3] = 1;
    a.st.cur[33] = 1;
    b.st.cur[3] = 0;
    b.st.cur[5] = 1;
    const at = 192;
    const s = sections(a.g, at, 64, a.draft(1).bytes());
    const buf = try gpa.alloc(u8, s.total);
    defer gpa.free(buf);
    const c = HostCopy.copier();
    try save(c, a.g, s, @intFromPtr(buf.ptr), &a.st, a.draft(1));
    const before = try gpa.dupe(u8, b.latent[4]);
    defer gpa.free(before);
    try restore(c, a.g, s, @intFromPtr(buf.ptr), &b.st, b.draft(0), true);
    for (0..11) |i| {
        try std.testing.expectEqualSlices(u8, a.latent[i][0 .. at * G.latent_row], b.latent[i][0 .. at * G.latent_row]);
        try std.testing.expectEqualSlices(u8, a.pooled[i][0 .. (at / 4 + 1) * G.pool_row], b.pooled[i][0 .. (at / 4 + 1) * G.pool_row]);
        try std.testing.expectEqualSlices(u8, a.keys[i][0 .. 64 * 256], b.keys[i][0 .. 64 * 256]);
        try std.testing.expectEqualSlices(u8, a.gates[i][0 .. 64 * 256], b.gates[i][0 .. 64 * 256]);
    }
    // rows past the prefix are b's own
    try std.testing.expect(!std.mem.eql(u8, a.latent[4][at * G.latent_row ..], b.latent[4][at * G.latent_row ..]));
    try std.testing.expectEqualSlices(u8, before[at * G.latent_row ..], b.latent[4][at * G.latent_row ..]);
    const rl = recLayerBytes(a.g);
    for (0..34) |l| {
        const from = a.st.recurrent(a.g, l, a.st.cur[l]);
        const into = b.st.recurrent(b.g, l, b.st.cur[l]);
        const x: [*]const u8 = @ptrFromInt(from);
        const y: [*]const u8 = @ptrFromInt(into);
        try std.testing.expectEqualSlices(u8, x[0..rl], y[0..rl]);
    }
    try std.testing.expectEqualSlices(u8, a.conv, b.conv);
    // the drafter ring of slot 0 now holds slot 1's, head by head
    for (0..5) |i| for (0..a.kv) |h| {
        try std.testing.expectEqualSlices(u8, a.kc[i][(h * a.cap + 16) * 256 ..][0 .. 16 * 256], b.kc[i][(h * b.cap) * 256 ..][0 .. 16 * 256]);
        try std.testing.expectEqualSlices(u8, a.vc[i][(h * a.cap + 16) * 256 ..][0 .. 16 * 256], b.vc[i][(h * b.cap) * 256 ..][0 .. 16 * 256]);
    };
}

test "TF_GLM_KV_SHARD: latent rows a view does not have here are neither saved nor restored, before any copy; without them the rest is" {
    const gpa = std.testing.allocator;
    var a = try Fake.init(gpa, 256, 64, 0);
    defer a.deinit();
    a.st.cache.owner = false;
    a.st.cache.latent = @splat(state.no_latent);
    const Count = struct {
        n: usize = 0,
        fn copy(ptr: *anyopaque, _: u64, _: u64, _: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            c.n += 1;
        }
    };
    var count: Count = .{};
    const c: Copier = .{ .ptr = &count, .copy = Count.copy };
    const s = sections(a.g, 192, 64, 0);
    try std.testing.expectError(error.NoLatentRows, save(c, a.g, s, 0x1000, &a.st, null));
    try std.testing.expectError(error.NoLatentRows, restore(c, a.g, s, 0x1000, &a.st, null, true));
    try std.testing.expectError(error.NoLatentRows, restore(c, a.g, sectionsOf(a.g, 192, 64, 0, false), 0x1000, &a.st, null, true));
    try std.testing.expectEqual(@as(usize, 0), count.n);
    // the other rank's sections: the latent section empty, everything else alike
    const t = sectionsOf(a.g, 192, 64, 0, false);
    try std.testing.expectEqual(s.total - std.mem.alignForward(usize, G.dsa_layers * s.latentBytes(), 256), t.total);
    try std.testing.expectEqual(s.conv - s.pooled, t.conv - t.pooled);
}

test "TF_GLM_KV_SHARD: a state kept without its latent rows saves and restores everything else, the latent rows left alone" {
    const gpa = std.testing.allocator;
    var a = try Fake.init(gpa, 256, 64, 1);
    defer a.deinit();
    var b = try Fake.init(gpa, 256, 64, 0);
    defer b.deinit();
    a.fill(3);
    b.fill(4);
    const at = 128;
    const s = sectionsOf(a.g, at, 64, a.draft(1).bytes(), false);
    const buf = try gpa.alloc(u8, s.total);
    defer gpa.free(buf);
    const c = HostCopy.copier();
    // the rank without the stream's rows: its view has none
    a.st.cache.owner = false;
    a.st.cache.latent = @splat(state.no_latent);
    try save(c, a.g, s, @intFromPtr(buf.ptr), &a.st, a.draft(1));
    const before = try gpa.dupe(u8, b.latent[7]);
    defer gpa.free(before);
    try restore(c, a.g, s, @intFromPtr(buf.ptr), &b.st, b.draft(0), false);
    try std.testing.expectEqualSlices(u8, before, b.latent[7]);
    for (0..11) |i| try std.testing.expectEqualSlices(u8, a.pooled[i][0 .. (at / 4 + 1) * G.pool_row], b.pooled[i][0 .. (at / 4 + 1) * G.pool_row]);
    try std.testing.expectEqualSlices(u8, a.conv, b.conv);
}

test "section sizes follow the geometry and stay 256-aligned" {
    const g = try G.init(2, 0);
    const s = sections(g, 4096, 2112, 22 << 20);
    try std.testing.expectEqual(@as(usize, 0), s.total % 256);
    try std.testing.expectEqual(@as(usize, 11 * 4096 * 528), s.pooled - s.latent);
    try std.testing.expect(s.total > 34 * recLayerBytes(g));
    // a prompt twice as long holds twice the rows and the same fixed state
    const t = sections(g, 8192, 2112, 22 << 20);
    try std.testing.expectEqual(t.pooled - t.latent, 2 * (s.pooled - s.latent));
    try std.testing.expect(t.total > s.total);
}

test "MTP drafts: the head's section sits after the others, one row short of the state, its own ring, 256-aligned" {
    const g = try G.init(2, 0);
    const s = sections(g, 4096, 2112, 22 << 20);
    try std.testing.expectEqual(@as(?usize, null), s.mtp_rows);
    try std.testing.expectEqual(@as(usize, 22 << 20), s.draftBytes());
    const t = withMtp(s, 2048 + 64);
    try std.testing.expectEqual(@as(?usize, 4095), t.mtp_rows);
    try std.testing.expectEqual(s.total, t.mtp);
    try std.testing.expectEqual(s.draftBytes(), t.draftBytes());
    // the target's sections are untouched
    try std.testing.expectEqual(s.conv, t.conv);
    try std.testing.expectEqual(s.draft, t.draft);
    try std.testing.expectEqual(@as(usize, 4095 * G.latent_row), t.mtpLatentBytes());
    // 1023 complete pools and the one row of the pool still filling (its 3 tokens are in the rings)
    try std.testing.expectEqual(@as(usize, 1024 * G.pool_row), t.mtpPooledBytes());
    try std.testing.expectEqual(@as(usize, 2112 * 256), t.mtpRingBytes());
    const o = t.mtpParts();
    for (o) |x| try std.testing.expectEqual(@as(usize, 0), x % 256);
    try std.testing.expectEqual(t.total, o[4] + std.mem.alignForward(usize, mtp_tail_bytes, 256));
    try std.testing.expectEqual(@as(usize, 0), t.total % 256);
    // a head ring other than the target's sizes the head's rings, not the target's
    const u = withMtp(s, 64);
    try std.testing.expectEqual(t.total - 2 * (2112 - 64) * 256, u.total);
    // no drafter: the DFlash2 section stays empty with the head's after it
    const v = withMtp(sections(g, 64, 2112, 0), 2112);
    try std.testing.expectEqual(@as(usize, 0), v.draftBytes());
    try std.testing.expectEqual(@as(?usize, 63), v.mtp_rows);
    try std.testing.expectEqual(@as(usize, 16 * G.pool_row), v.mtpPooledBytes());
    // a state at 0 (the budget's fixed size) has no head rows
    try std.testing.expectEqual(@as(?usize, 0), withMtp(sections(g, 0, 2112, 0), 2112).mtp_rows);
}

test "MTP drafts: the head's plane and pending row round-trip through the section, nothing past its rows" {
    const gpa = std.testing.allocator;
    const cap = 256;
    const ring = 64;
    const Plane = struct {
        latent: []u8,
        pooled: []u8,
        keys: []u8,
        gates: []u8,
        pending: []u8,
        fn init(a: std.mem.Allocator, seed: u8) !@This() {
            const p: @This() = .{ .latent = try a.alloc(u8, cap * G.latent_row), .pooled = try a.alloc(u8, (cap / 4 + 2) * G.pool_row), .keys = try a.alloc(u8, ring * 256), .gates = try a.alloc(u8, ring * 256), .pending = try a.alloc(u8, mtp_tail_bytes) };
            var prng = std.Random.DefaultPrng.init(seed);
            for ([_][]u8{ p.latent, p.pooled, p.keys, p.gates, p.pending }) |b| prng.random().bytes(b);
            return p;
        }
        fn deinit(p: *@This(), a: std.mem.Allocator) void {
            for ([_][]u8{ p.latent, p.pooled, p.keys, p.gates, p.pending }) |b| a.free(b);
        }
        fn view(p: *const @This()) MtpPlane {
            return .{ .latent = @intFromPtr(p.latent.ptr), .pooled = @intFromPtr(p.pooled.ptr), .keys = @intFromPtr(p.keys.ptr), .gates = @intFromPtr(p.gates.ptr), .ring = ring, .pending = @intFromPtr(p.pending.ptr) };
        }
    };
    var a = try Plane.init(gpa, 5);
    defer a.deinit(gpa);
    var b = try Plane.init(gpa, 6);
    defer b.deinit(gpa);
    const g = try G.init(2, 0);
    const s = withMtp(sections(g, 128, 64, 0), ring);
    const buf = try gpa.alloc(u8, s.total);
    defer gpa.free(buf);
    const c = HostCopy.copier();
    try mtpCopy(c, s, @intFromPtr(buf.ptr), a.view(), true);
    const after = try gpa.dupe(u8, b.latent[127 * G.latent_row ..]);
    defer gpa.free(after);
    try mtpCopy(c, s, @intFromPtr(buf.ptr), b.view(), false);
    try std.testing.expectEqualSlices(u8, a.latent[0 .. 127 * G.latent_row], b.latent[0 .. 127 * G.latent_row]);
    try std.testing.expectEqualSlices(u8, after, b.latent[127 * G.latent_row ..]);
    try std.testing.expectEqualSlices(u8, a.pooled[0 .. 32 * G.pool_row], b.pooled[0 .. 32 * G.pool_row]);
    try std.testing.expectEqualSlices(u8, a.keys, b.keys);
    try std.testing.expectEqualSlices(u8, a.gates, b.gates);
    try std.testing.expectEqualSlices(u8, a.pending, b.pending);
    // no head section, or another ring: refused before any copy
    try std.testing.expectError(error.NoMtpSection, mtpCopy(c, sections(g, 128, 64, 0), @intFromPtr(buf.ptr), b.view(), false));
    var other = b.view();
    other.ring = 128;
    try std.testing.expectError(error.MtpRingMismatch, mtpCopy(c, s, @intFromPtr(buf.ptr), other, false));
}
