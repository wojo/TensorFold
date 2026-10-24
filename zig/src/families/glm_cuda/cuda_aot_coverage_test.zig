//! The captured kernel set (zig/tests/cuda/glm_p1/aot/aot.json, the recipe's glm-aot-sm121.tar.xz) against every
//! Triton lookup the GLM engine makes at TP=2 for a context, a stream count and a KV pool, offline: decode windows of
//! 1..64 rows over the pool's decode scratch, and prompt chunks of every row count a prompt can have (`chunk_rows`) at
//! every position a prompt or a resume reaches (the 64-token grid) in an extent anywhere from the prompt's own up to
//! the context's, the row split off and on. The engine's own launch code runs (cuda_prefill.Engine.chunk / dsaLayer,
//! cuda_forward.Forward.compute) against a driver that only hands out addresses: matmuls, KDA, copies and NCCL calls
//! are accepted and dropped, and each Triton launch is looked up as aot.Set.find would (aot.matches). `known_missing`
//! lists the variants the set lacks: a new miss fails the test, and so does a listed one the set has gained (drop it
//! from the list then). Not run here: the MTP head and the DFlash2 drafter (their shapes do not depend on the context),
//! and TF_GLM_KV_SHARD's own launches (cuda_shard_decode_test / cuda_shard_prompt_test check that each finds the
//! variant of its unsharded launch, which this sweep covers; the shard's smaller extents are in `capMax`).
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const abi = cuda.abi;
const fw = @import("cuda_forward.zig");
const pre = @import("cuda_prefill.zig");
const grid = @import("cuda_prefill_grid.zig");
const split = @import("cuda_prefill_split.zig");
const state = @import("cuda_state.zig");
const seg = @import("cuda_segments.zig");
const mm = @import("cuda_matmuls.zig");
const kda = @import("cuda_kda.zig");
const hc = @import("cuda_hc.zig");
const mla = @import("cuda_mla.zig");
const tri_mod = @import("cuda_triton.zig");
const engine = @import("cuda_engine.zig");
const G = @import("cuda_geometry.zig").Geometry;

const gpa = std.testing.allocator;

/// The variants a run at these settings looks up and the captured set lacks, as aot.find's MISSING-VARIANT lines: none.
/// (The set does lack the prompt-block selection of a block under 64 rows: `_select_split` with pools out, OUT int64
/// SEG=0 VIS=0 TOKS=0, every (CHS, CP) from 2048/4 to 4096/64; `_select_rows` BLOCK=4096 VIS=0; `_scores` R=1. A chunk
/// of 64 rows or more whose row count is not a multiple of 64 would reach them, e.g. 530 rows past ~32k tokens; the
/// engine never makes one, see `chunk_rows`.)
pub const known_missing = [_][]const u8{};

// -- a driver that hands out addresses --------------------------------------------------------------------------------

const Fake = struct {
    var next: u64 = 0x7000_0000_0000;
    /// every entry point: success (a C callee ignores the arguments it does not declare)
    fn ok() callconv(.c) c_int {
        return 0;
    }
    /// cuMemAlloc: 2 MiB steps (the driver's own alignment for large allocations is at least 256 bytes)
    fn alloc(p: *abi.DevicePtr, n: usize) callconv(.c) c_int {
        p.* = next;
        next += std.mem.alignForward(usize, @max(n, 1), 1 << 21);
        return 0;
    }
    fn driver() cuda.Driver {
        var api: abi.Api = undefined;
        const info = @typeInfo(abi.Api).@"struct";
        inline for (info.field_names) |name| @field(api, name) = @ptrCast(&ok);
        api.cuMemAlloc_v2 = alloc;
        return .{ .lib = undefined, .api = api };
    }
    fn nccl() cuda.nccl.Library {
        var api: cuda.nccl.Api = undefined;
        const info = @typeInfo(cuda.nccl.Api).@"struct";
        inline for (info.field_names) |name| @field(api, name) = @ptrCast(&ok);
        return .{ .lib = undefined, .api = api };
    }
};

var drv: cuda.Driver = undefined;

fn dev(bytes: usize) u64 {
    var p: abi.DevicePtr = 0;
    _ = Fake.alloc(&p, bytes);
    return p;
}
fn func() cuda.Function {
    return .{ .d = &drv, .handle = null };
}
fn q4(n: usize, k: usize) mm.Q4 {
    return .{ .w = dev(n * k / 2), .s = dev(n * 2), .b = dev(n * 2), .n = n, .k = k };
}
fn hcw() hc.Weights {
    return .{ .fn_w = dev(4096), .base = dev(4096), .scale = dev(4096), .norm = dev(4096) };
}
fn mlpw(width: usize) fw.MlpW {
    return .{ .gu = q4(2 * width, 4096), .down = q4(4096, width) };
}

/// The model's weights with each tensor its own allocation (cuda_model_weights' put) and the shapes that reach a
/// Triton launch (the MLP widths, the projections' row widths) as the checkpoint gives them at this rank.
fn weights(g: G) !*fw.Weights {
    const w = try gpa.create(fw.Weights);
    w.* = .{ .embed = dev(1 << 20), .norm = dev(8192), .head = dev(1 << 20), .head_scale = dev(1 << 20), .layers = undefined };
    const h = g.lin_heads;
    for (&w.layers, 0..) |*l, li| {
        l.attn = hcw();
        l.ffn = hcw();
        l.mixer = if (G.isDsa(li)) .{ .dsa = .{ .proj = q4(2048, 4096), .qb = q4(g.heads * 256, 1536), .o = q4(4096, g.heads * 256), .kw = q4(160, 4096), .iqb = q4(4096, 1536), .q_norm = dev(4096), .kv_norm = dev(4096), .wk = dev(1 << 20), .wv = dev(1 << 20), .gate = dev(4096), .ln_w = dev(4096), .ln_b = dev(4096), .ape = dev(4096) } } else .{ .kda = .{ .proj = q4(g.projWidth(), 4096), .fb = q4(h * 128, 128), .gb = q4(h * 128, 128), .o = q4(4096, h * 128), .fa_off = 3 * h * 128, .ga_off = 3 * h * 128 + 128, .b_off = 3 * h * 128 + 256, .conv = dev(1 << 20), .a_log = dev(4096), .dt_bias = dev(4096), .norm = dev(4096) } };
        l.mlp = if (li < 3) .{ .dense = mlpw(g.dense_width) } else .{ .moe = .{ .router = dev(1 << 20), .bias = dev(4096), .shared = mlpw(g.shared_width), .experts = .{ .gt = dev(1 << 20), .ut = dev(1 << 20), .dt = dev(1 << 20), .suh_g = dev(4096), .suh_u = dev(4096), .svh_g = dev(4096), .svh_u = dev(4096), .suh_d = dev(4096), .svh_d = dev(4096), .width = g.moe_width, .count = 288 } } };
    }
    return w;
}

// -- the lookups -------------------------------------------------------------------------------------------------------

/// aot.find's MISSING-VARIANT line of a launch (its argument classes and constexprs in launch order), each pointer's
/// type added in brackets: with it the line is the lookup's whole key (aot.matches reads the types too).
fn describe(out: *std.ArrayList(u8), name: []const u8, args: []const aot.Arg, consts: []const aot.Const) !void {
    try out.print(gpa, "MISSING-VARIANT fn={s}", .{name});
    for (args) |a| switch (a.value) {
        .ptr => |p| try out.print(gpa, " {s}={s}({s})", .{ a.name, if (p.addr % 16 == 0) "p16" else "p", p.ty }),
        .i32 => |x| try out.print(gpa, " {s}={s}", .{ a.name, if (x == 1) "i1" else if (@mod(x, 16) == 0) "i16" else "i" }),
        .u64 => |x| try out.print(gpa, " {s}={s}", .{ a.name, if (x % 16 == 0) "u16" else "u" }),
        .f32 => try out.print(gpa, " {s}=f", .{a.name}),
    };
    for (consts) |c| {
        if (c.int) |x| try out.print(gpa, " {s}:{d}", .{ c.name, x });
        if (c.f32) |x| try out.print(gpa, " {s}:f32bits:{d}", .{ c.name, @as(u32, @bitCast(x)) });
    }
}

/// Every distinct lookup seen (by its `describe` line) and whether the set has it.
const Lookups = struct {
    ks: []const aot.KernelJson,
    seen: std.StringHashMapUnmanaged(bool) = .empty,
    /// the lookups of the run in progress (Sweep.once's memo, the sampled chunks)
    run: std.ArrayList([]const u8) = .empty,
    line: std.ArrayList(u8) = .empty,
    fn deinit(l: *Lookups) void {
        var it = l.seen.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        l.seen.deinit(gpa);
        l.run.deinit(gpa);
        l.line.deinit(gpa);
    }
    fn sink(ctx: *anyopaque, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) anyerror!void {
        const l: *Lookups = @ptrCast(@alignCast(ctx));
        l.line.clearRetainingCapacity();
        try describe(&l.line, name, args, consts);
        const got = try l.seen.getOrPut(gpa, l.line.items);
        if (!got.found_existing) {
            got.key_ptr.* = try gpa.dupe(u8, l.line.items);
            got.value_ptr.* = for (l.ks) |k| {
                if (std.mem.eql(u8, k.@"fn", name) and aot.matches(k, args, consts)) break true;
            } else false;
        }
        try l.run.append(gpa, got.key_ptr.*);
    }
};

// -- the engine's pieces on the fake driver ------------------------------------------------------------------------------

const Rig = struct {
    g: G,
    w: *fw.Weights,
    entries: *mm.Entries,
    kk: *kda.Kernels,
    comm: *fw.Comm,
    rec: *tri_mod.Recorder,
    stream: cuda.Stream,
    slots: state.Slots,

    fn init(rank: usize, streams: usize, l: *Lookups) !Rig {
        const g = try G.init(2, rank);
        const entries = try gpa.create(mm.Entries);
        for (&entries.functions) |*f| f.* = func();
        const kk = try gpa.create(kda.Kernels);
        const info = @typeInfo(kda.Kernels).@"struct";
        inline for (info.field_names, info.field_types) |name, T| if (T == cuda.Function) {
            @field(kk.*, name) = func();
        };
        const comm = try gpa.create(fw.Comm);
        comm.* = .{ .lib = Fake.nccl(), .comm = null, .rank = @intCast(rank), .world = 2 };
        const rec = try gpa.create(tri_mod.Recorder);
        rec.* = .{ .sink = .{ .ctx = l, .call = Lookups.sink } };
        return .{ .g = g, .w = try weights(g), .entries = entries, .kk = kk, .comm = comm, .rec = rec, .stream = .{ .d = &drv, .handle = null }, .slots = try state.Slots.init(&drv, g, streams, 64) };
    }
    fn deinit(r: *Rig) void {
        gpa.destroy(r.w);
        gpa.destroy(r.entries);
        gpa.destroy(r.kk);
        gpa.destroy(r.comm);
        gpa.destroy(r.rec);
    }
    fn tri(r: *const Rig) tri_mod.Tri {
        return .{ .rec = r.rec, .s = r.stream };
    }
    /// cuda_engine's prompt engine (its 2,048-row buffers and scratch for `extent`), slot 0 bound to [0, cap).
    fn prompt(r: *const Rig, caches: *const state.Caches, extent: usize, cap: usize, s: split.Settings) !pre.Engine {
        const st = try state.State.bind(r.slots, caches.*, 0, 0, @min(cap, caches.rows));
        var e: pre.Engine = .{ .g = r.g, .tri = r.tri(), .entries = r.entries, .weights = r.w, .b = try state.Buffers.init(&drv, r.g, 2048, true), .s = try pre.Scratch.init(&drv, r.g, 2048, extent), .cache = st.cache, .conv = st.fixed.conv, .rec = st.fixed.rec, .ring = caches.ring, .pos_dev = st.fixed.pos_dev, .kda = .{ .k = r.kk, .s = r.stream }, .comm = r.comm, .split = s };
        e.lane = if (s.on and s.overlap != 0) try split.Lane.init(&drv) else null;
        return e;
    }
    /// cuda_engine's decode forward over a pool of `pool` rows (its scratch sized by decodeScratchRows).
    fn decode(r: *const Rig, caches: *const state.Caches, tables: *const seg.SegRows, pool: usize) !fw.Forward {
        return .{ .g = r.g, .tri = r.tri(), .hc = .{ .mod = .{ .d = &drv, .handle = null }, .partial = func() }, .entries = r.entries, .weights = r.w, .b = try state.Buffers.init(&drv, r.g, 64, false), .s = try fw.Scratch.init(&drv, r.g, 64, engine.decodeScratchRows(pool)), .slots = &r.slots, .caches = caches, .tables = tables, .kseg = 0, .comm = r.comm, .kda = kda.install(r.kk) };
    }
};

// -- what a configuration reaches -----------------------------------------------------------------------------------------

pub const Config = struct { context: usize, streams: usize, pool: usize, shard: bool = false };

fn extentOf(c: Config) usize {
    return std.mem.alignForward(usize, c.context + 64, G.extent);
}
/// The largest extent a request gets: the context's, at most the pool (TF_GLM_KV_SHARD: a part).
fn capMax(c: Config) usize {
    const room = if (c.shard) c.pool / 2 else c.pool;
    return @min(extentOf(c), room / G.extent * G.extent);
}

/// What a chunk's selection launches depend on beyond its rows (cuda_prefill.selectRows / choose / splitCall): each
/// block's grid.selectLaunch and, split, splitVariants' (CHS, CP) for the pool count; and the count's own class.
fn selClass(rows: usize, np: usize) u64 {
    var h = std.hash.Wyhash.init(rows);
    const blocks: [2]usize = if (rows >= grid.prompt_rows) .{ @min(rows, grid.select_rows), (rows - 1) % grid.select_rows + 1 } else .{ rows, rows };
    for (blocks) |n| {
        const l = grid.selectLaunch(n, np, rows >= grid.prompt_rows) catch return 0;
        h.update(&.{ @backingInt(l.kind), @intFromBool(l.vis), @intCast(l.warps) });
        h.update(std.mem.asBytes(&l.block));
        if (l.kind == .split) h.update(std.mem.asBytes(&[2]usize{ mla.splitChunk(np), mla.splitChunks(np) }));
    }
    h.update(&.{ @intFromBool(np == 1), @intFromBool(np % 16 == 0) });
    return h.final();
}

/// The pool count a chunk past the dense limit selects over (a tail under 64 rows: the power-of-two bucket).
fn poolsOf(pos: usize, rows: usize, cap: usize) usize {
    return if (rows >= grid.prompt_rows) grid.poolCount(pos, rows, cap / 4) else grid.poolBucket(pos, rows, cap / 4);
}

/// The rows a prompt chunk has: cuda_engine.prefillWith fills [begin, point) and then [point, n), `point` the prompt's
/// 64-token grid point (grid.gridPoint), and every begin (a kept state, a resume, keep_at) is on that grid; the loop
/// (cuda_prefill_loop.run) cuts at grid.chunkLimit, 2,048 or 1,024 rows. So a chunk is a multiple of 64 rows up to
/// 2,048, or a prompt's last 1..63 rows. (TF_GLM_PROMPT_MICROBATCH halves and TF_GLM_PROMPT_OWN_FRONT, both off at
/// TP=2 by default, would cut other row counts: not covered here.)
pub const chunk_rows = blk: {
    var out: [grid.prompt_rows - 1 + grid.chunk_rows / grid.state_grid]usize = undefined;
    for (1..grid.prompt_rows) |r| out[r - 1] = r;
    for (1..grid.chunk_rows / grid.state_grid + 1) |k| out[grid.prompt_rows - 2 + k] = k * grid.state_grid;
    break :blk out;
};

const any_rows = blk: {
    @setEvalBranchQuota(10_000);
    var out: [grid.chunk_rows]usize = undefined;
    for (&out, 1..) |*r, n| r.* = n;
    break :blk out;
};
/// `chunk_rows`, or with TF_GLM_AOT_COVERAGE_ALL_ROWS=1 every count 1..2,048 (a diagnostic: what a chunk planner that
/// did not cut at the grid would look up; the test then fails on those lookups).
fn sweptRows() []const usize {
    return if (std.c.getenv("TF_GLM_AOT_COVERAGE_ALL_ROWS") != null) &any_rows else &chunk_rows;
}

/// The chunks a prompt of `n` tokens resumed at `begin` fills (prefillWith's cut, the loop's chunks), into `out`.
pub fn promptChunks(n: usize, begin: usize, busy: bool, out: []grid.Span) ![]grid.Span {
    const step = grid.chunkLimit(busy, grid.chunk_rows);
    var k: usize = 0;
    const point = grid.gridPoint(n, begin, grid.state_grid);
    var at = begin;
    if (point) |p| if (p > begin and p < n) {
        k = (try grid.chunks(p, begin, step, out)).len;
        at = p;
    };
    return out[0 .. k + (try grid.chunks(n, at, step, out[k..])).len];
}

const Chunk = struct { rows: usize, pos: usize, cap: usize };
const first_sparse = std.mem.alignForward(usize, grid.sparse_from, grid.state_grid);

/// A chunk of `rows` rows past the dense limit that selects over `np` pools in this configuration, if one exists: the
/// count reached by the prompt's length (the largest extent), or as a request's own smaller extent (its cap).
fn realize(c: Config, rows: usize, np: usize) ?Chunk {
    const top = capMax(c);
    const forms = [2]struct { t: usize, cap: usize }{ .{ .t = @min(4 * (np - 1), c.context), .cap = top }, .{ .t = @min(4 * np, c.context), .cap = 4 * np } };
    for (forms) |f| {
        if (f.cap > top or f.t < rows) continue;
        const pos = std.mem.alignBackward(usize, f.t - rows, grid.state_grid);
        if (pos < first_sparse or pos + rows > @min(c.context, f.cap)) continue;
        if (poolsOf(pos, rows, f.cap) == np) return .{ .rows = rows, .pos = pos, .cap = f.cap };
    }
    return null;
}

/// Records one chunk's DSA layer (cuda_prefill.dsaLayer: its index, selection and attention launches), its request's
/// extent [0, cap) bound.
fn dsaChunk(e: *pre.Engine, r: *const Rig, caches: *const state.Caches, k: Chunk) !void {
    e.cache = try caches.view(0, k.cap, 0);
    try e.dsaLayer(r.w.layers[3].mixer.dsa, 0, k.rows, k.pos, null);
}

// -- the sweep ---------------------------------------------------------------------------------------------------------

const Lines = std.StringHashMapUnmanaged(void);

/// Lookups by what decides them, each run once: a prompt chunk's DSA layer by (rank, rows, selection class) or, under
/// the dense limit, by (rank, rows, pos, cap); a decode window by (rank, decode scratch rows); whole chunks by rank.
const Sweep = struct {
    l: Lookups,
    rigs: [2]Rig = undefined,
    caches: [2]state.Caches = undefined,
    engines: std.AutoHashMapUnmanaged(u64, *pre.Engine) = .empty,
    memo: std.AutoHashMapUnmanaged(u64, [][]const u8) = .empty,
    /// per rows: the selection classes over pool counts 512, 1024, ... as runs [lo, hi] (cuda_prefill.selectRows)
    runs: [grid.chunk_rows + 1]std.ArrayList([3]u64) = @splat(.empty),

    fn init(s: *Sweep, ks: []const aot.KernelJson, cap_top: usize) !void {
        s.* = .{ .l = .{ .ks = ks } };
        for (0..2) |rank| {
            s.rigs[rank] = try Rig.init(rank, 8, &s.l);
            s.caches[rank] = try state.Caches.init(&drv, cap_top, 8, grid.chunk_rows);
        }
        for (sweptRows()) |rows| {
            var np: usize = 512;
            while (np <= cap_top / 4) : (np += 512) {
                const c = selClass(rows, np);
                const r = &s.runs[rows];
                if (r.items.len > 0 and r.items[r.items.len - 1][0] == c) r.items[r.items.len - 1][2] = np else try r.append(gpa, .{ c, np, np });
            }
        }
    }
    fn deinit(s: *Sweep) void {
        var it = s.memo.valueIterator();
        while (it.next()) |v| gpa.free(v.*);
        s.memo.deinit(gpa);
        var ei = s.engines.valueIterator();
        while (ei.next()) |e| gpa.destroy(e.*);
        s.engines.deinit(gpa);
        for (&s.runs) |*r| r.deinit(gpa);
        for (&s.rigs) |*r| r.deinit();
        s.l.deinit();
    }
    fn engineFor(s: *Sweep, rank: usize, extent: usize, cfg: split.Settings) !*pre.Engine {
        const k = (extent << 8) | (rank << 4) | (@as(u64, @intFromBool(cfg.on)) << 1) | cfg.overlap;
        const got = try s.engines.getOrPut(gpa, k);
        if (!got.found_existing) {
            got.value_ptr.* = try gpa.create(pre.Engine);
            got.value_ptr.*.* = try s.rigs[rank].prompt(&s.caches[rank], extent, grid.chunk_rows, cfg);
        }
        return got.value_ptr.*;
    }
    /// The lookups of `k`, made by `run` the first time.
    fn once(s: *Sweep, k: u64, out: *Lines, ctx: anytype, comptime run: fn (@TypeOf(ctx)) anyerror!void) !void {
        const got = try s.memo.getOrPut(gpa, k);
        if (!got.found_existing) {
            s.l.run.clearRetainingCapacity();
            run(ctx) catch |e| {
                _ = s.memo.remove(k);
                return e;
            };
            got.value_ptr.* = try gpa.dupe([]const u8, s.l.run.items);
        }
        for (got.value_ptr.*) |line| try out.put(gpa, line, {});
    }
};

fn keyOf(parts: []const usize) u64 {
    return std.hash.Wyhash.hash(0x61_6f_74, std.mem.sliceAsBytes(parts));
}

const DsaRun = struct {
    e: *pre.Engine,
    r: *const Rig,
    caches: *const state.Caches,
    k: Chunk,
    fn run(c: DsaRun) anyerror!void {
        try dsaChunk(c.e, c.r, c.caches, c.k);
    }
};

/// A pool count of selection-class run `run` (rows' [class, lo, hi]) some chunk reaches here: from either end.
fn pick(c: Config, rows: usize, run: [3]u64) ?Chunk {
    const hi = @min(run[2], capMax(c) / 4);
    if (hi < run[1]) return null;
    var np: usize = run[1];
    while (np <= hi) : (np += 512) if (realize(c, rows, np)) |k| return k;
    return null;
}

/// The prompt chunks' DSA lookups of configuration `c`: under the dense limit every (pos, rows) and both ends of a
/// request's extent; past it every rows and selection class (a pool count reaching it).
fn promptLines(s: *Sweep, c: Config, out: *Lines) !void {
    const top = capMax(c);
    for (0..2) |rank| {
        const e = try s.engineFor(rank, extentOf(c), .{});
        const r = &s.rigs[rank];
        var pos: usize = 0;
        while (pos < grid.sparse_from) : (pos += grid.state_grid) for (sweptRows()) |rows| {
            for ([2]usize{ std.mem.alignForward(usize, pos + rows, G.extent), top }) |cap| {
                const sparse = pos + rows - 1 >= grid.sparse_from;
                const np = if (sparse) poolsOf(pos, rows, cap) else 0;
                try s.once(keyOf(&.{ 1, rank, rows, pos, np }), out, DsaRun{ .e = e, .r = r, .caches = &s.caches[rank], .k = .{ .rows = rows, .pos = pos, .cap = cap } }, DsaRun.run);
            }
        };
        for (sweptRows()) |rows| for (s.runs[rows].items) |run| {
            if (run[0] == 0) continue;
            const k = pick(c, rows, run) orelse continue;
            try s.once(keyOf(&.{ 2, rank, rows, run[0] }), out, DsaRun{ .e = e, .r = r, .caches = &s.caches[rank], .k = k }, DsaRun.run);
        };
    }
}

const DecodeRun = struct {
    s: *Sweep,
    rank: usize,
    pool: usize,
    fn run(c: DecodeRun) anyerror!void {
        const r = &c.s.rigs[c.rank];
        var tables = try seg.SegRows.init(gpa, 64, 8, c.s.caches[c.rank].ring);
        defer tables.deinit();
        try tables.initDevice(&drv);
        const f = try r.decode(&c.s.caches[c.rank], &tables, c.pool);
        for (1..65) |rows| try f.compute(rows);
    }
};

/// Decode windows of 1..64 rows over pool `c.pool` (cuda_engine sizes the decode scratch by decodeScratchRows).
fn decodeLines(s: *Sweep, c: Config, out: *Lines) !void {
    for (0..2) |rank| try s.once(keyOf(&.{ 3, rank, engine.decodeScratchRows(c.pool) }), out, DecodeRun{ .s = s, .rank = rank, .pool = c.pool }, DecodeRun.run);
}

const ChunkRun = struct {
    s: *Sweep,
    rank: usize,
    cfg: split.Settings,
    fn run(c: ChunkRun) anyerror!void {
        const e = try c.s.engineFor(c.rank, 133120, c.cfg);
        var parity: [34]u1 = @splat(0);
        for (sweptRows()) |rows| {
            e.cache = try c.s.caches[c.rank].view(0, std.mem.alignForward(usize, rows, G.extent), 0);
            try e.chunk(rows, 0, &parity, true);
        }
    }
};

/// Whole prompt chunks of 1..2,048 rows (every layer, the glue, the head), the row split off and on (the recipe's
/// SPLIT: TF_GLM_HC_SPLIT=1 with TF_GLM_PREFILL_OVERLAP=2): what does not depend on the context.
fn chunkLines(s: *Sweep, out: *Lines) !void {
    for (0..2) |rank| for ([2]split.Settings{ .{}, .{ .on = true, .overlap = 2 } }) |cfg| {
        try s.once(keyOf(&.{ 4, rank, @intFromBool(cfg.on) }), out, ChunkRun{ .s = s, .rank = rank, .cfg = cfg }, ChunkRun.run);
    };
}

pub const configs = blk: {
    const contexts = [_]usize{ 131072, 262144, 524288, 1048576 };
    // KV pools from ~0.8M to ~4M rows (g1-g3: 825,344 with KV_POOL_GIB=4; 1,511,424 KV shard; 2,281,472 default)
    const pools = [_]usize{ 825344, 1511424, 2281472, 3014656, 4194304 };
    var out: [contexts.len * 2 * pools.len * 2]Config = undefined;
    var n: usize = 0;
    for (contexts) |ctx| for ([_]usize{ 8, 4 }) |p| for (pools) |pool| for ([_]bool{ false, true }) |sh| {
        out[n] = .{ .context = ctx, .streams = p, .pool = pool, .shard = sh };
        n += 1;
    };
    break :blk out;
};

/// The captured set (read from the repository root, the test step's working directory), or a skip without it.
fn loadSet() !std.json.Parsed(aot.SetJson) {
    const text = std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/cuda/glm_p1/aot/aot.json", gpa, .limited(1 << 24)) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer gpa.free(text);
    return std.json.parseFromSlice(aot.SetJson, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

fn missingOf(l: *const Lookups, lines: *const Lines, out: *Lines) !usize {
    var n: usize = 0;
    var it = lines.keyIterator();
    while (it.next()) |k| if (!l.seen.get(k.*).?) {
        try out.put(gpa, k.*, {});
        n += 1;
    };
    return n;
}

test "every Triton lookup of a TP=2 context, stream count and KV pool is in the captured kernel set, but the known gaps" {
    var set = try loadSet();
    defer set.deinit();
    drv = Fake.driver();
    var top: usize = 0;
    for (configs) |c| top = @max(top, capMax(c));
    const s = try gpa.create(Sweep);
    defer gpa.destroy(s);
    try s.init(set.value.kernels, top);
    defer s.deinit();
    const log = std.c.getenv("TF_GLM_AOT_COVERAGE_LOG") != null;
    var shared: Lines = .empty;
    defer shared.deinit(gpa);
    try chunkLines(s, &shared);
    var missing: Lines = .empty;
    defer missing.deinit(gpa);
    var per: [configs.len]Lines = @splat(.empty);
    defer for (&per) |*p| p.deinit(gpa);
    for (configs, &per) |c, *lines| {
        try promptLines(s, c, lines);
        try decodeLines(s, c, lines);
        var it = shared.keyIterator();
        while (it.next()) |k| try lines.put(gpa, k.*, {});
        const n = try missingOf(&s.l, lines, &missing);
        if (log) std.debug.print("aot coverage: context {d} parallel {d} pool {d}{s}: {d} lookups, {d} missing\n", .{ c.context, c.streams, c.pool, if (c.shard) " (KV shard parts)" else "", lines.count(), n });
    }
    // the sweep stands for every chunk: random prompts' chunks (any length up to the context, any resume point on the
    // grid, any extent, busy or not) look up nothing their configuration's sweep did not
    var prng = std.Random.DefaultPrng.init(0x1d0c);
    const rnd = prng.random();
    var spans: [1100]grid.Span = undefined;
    var strays: usize = 0;
    for (0..1000) |_| {
        const ci = rnd.uintLessThan(usize, configs.len);
        const c = configs[ci];
        const most = capMax(c);
        const n = rnd.intRangeAtMost(usize, 1, @min(c.context, most));
        const begin = if (rnd.boolean()) 0 else rnd.uintLessThan(usize, (n - 1) / grid.state_grid + 1) * grid.state_grid;
        const low = std.mem.alignForward(usize, n, G.extent);
        const cap = low + rnd.uintLessThan(usize, (most - low) / G.extent + 1) * G.extent;
        const cs = try promptChunks(n, begin, rnd.boolean(), &spans);
        for ([3]usize{ cs.len - 1, cs.len -| 2, rnd.uintLessThan(usize, cs.len) }) |i| {
            const rank = rnd.uintLessThan(usize, 2);
            s.l.run.clearRetainingCapacity();
            try dsaChunk(try s.engineFor(rank, extentOf(c), .{}), &s.rigs[rank], &s.caches[rank], .{ .rows = cs[i].rows, .pos = cs[i].start, .cap = cap });
            for (s.l.run.items) |line| if (!per[ci].contains(line)) {
                std.debug.print("a {d}-row chunk at {d} (prompt {d} from {d}, extent {d}, context {d}, pool {d}) looks up what the sweep did not: {s}\n", .{ cs[i].rows, cs[i].start, n, begin, cap, c.context, c.pool, line });
                strays += 1;
            };
        }
    }
    try std.testing.expectEqual(@as(usize, 0), strays);
    if (log) {
        var all = s.l.seen.iterator();
        while (all.next()) |x| std.debug.print("aot coverage lookup ({s}): {s}\n", .{ if (x.value_ptr.*) "captured" else "missing", x.key_ptr.*[16..] });
    }
    // the gaps, as a capture list
    var unexpected: usize = 0;
    var it = missing.keyIterator();
    while (it.next()) |k| {
        const known = for (known_missing) |m| {
            if (std.mem.eql(u8, m, k.*)) break true;
        } else false;
        if (!known) unexpected += 1;
        if (log or !known) std.debug.print("{s}{s}\n", .{ k.*, if (known) " (known)" else "" });
    }
    var gained: usize = 0;
    for (known_missing) |m| if (!missing.contains(m)) {
        std.debug.print("captured now (drop from known_missing): {s}\n", .{m});
        gained += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), unexpected);
    try std.testing.expectEqual(@as(usize, 0), gained);
}

test "a prompt's chunks are the rows and positions the sweep enumerates" {
    var prng = std.Random.DefaultPrng.init(0x64);
    const rnd = prng.random();
    var spans: [1100]grid.Span = undefined;
    for (0..2000) |_| {
        const n = rnd.intRangeAtMost(usize, 1, 1 << 20);
        const begin = rnd.uintLessThan(usize, (n - 1) / grid.state_grid + 1) * grid.state_grid;
        const cs = try promptChunks(n, begin, rnd.boolean(), &spans);
        var at = begin;
        for (cs) |k| {
            try std.testing.expectEqual(at, k.start);
            try std.testing.expect(k.start % grid.state_grid == 0);
            try std.testing.expect(std.mem.indexOfScalar(usize, &chunk_rows, k.rows) != null);
            at += k.rows;
        }
        try std.testing.expectEqual(n, at);
    }
    // prefillWith's own example: a 5,021-token prompt is 2,048, 2,048, 896, then 29
    const five = try promptChunks(5021, 0, false, &spans);
    try std.testing.expectEqual(@as(usize, 4), five.len);
    try std.testing.expectEqual(grid.Span{ .start = 4096, .rows = 896 }, five[2]);
    try std.testing.expectEqual(grid.Span{ .start = 4992, .rows = 29 }, five[3]);
}

test "a variant the set lacks is reported where it is reached: decode selection over a pool past ~2.1M rows" {
    var set = try loadSet();
    defer set.deinit();
    drv = Fake.driver();
    // the set without `_select_split` at CP=256 (the decode scratch's selection from 2,113,536 pool rows to 4M)
    var kept: std.ArrayList(aot.KernelJson) = .empty;
    defer kept.deinit(gpa);
    for (set.value.kernels) |k| {
        const cp = if (k.consts.map.get("CP")) |v| v.int else null;
        if (!(std.mem.eql(u8, k.@"fn", "_select_split") and cp != null and cp.? == 256)) try kept.append(gpa, k);
    }
    const s = try gpa.create(Sweep);
    defer gpa.destroy(s);
    try s.init(kept.items, 1 << 20);
    defer s.deinit();
    for ([_]usize{ 825344, 2281472 }, [_]usize{ 0, 5 }) |pool, want| {
        var lines: Lines = .empty;
        defer lines.deinit(gpa);
        try decodeLines(s, .{ .context = 131072, .streams = 8, .pool = pool }, &lines);
        var missing: Lines = .empty;
        defer missing.deinit(gpa);
        try std.testing.expectEqual(want, try missingOf(&s.l, &lines, &missing));
        var it = missing.keyIterator();
        while (it.next()) |k| try std.testing.expect(std.mem.indexOf(u8, k.*, "fn=_select_split") != null and std.mem.indexOf(u8, k.*, "CP:256 ") != null and std.mem.indexOf(u8, k.*, "SEG:1 ") != null);
    }
}
