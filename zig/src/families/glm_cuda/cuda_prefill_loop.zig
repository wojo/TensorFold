//! Host loop over a long prompt: 2048-row chunks (1024 while others decode), the 64-token grid, pool growth, one segment.
//! glm5_next/cuda/decode.py prefill / prefill_chunk. SPLIT exchanges and overlap are P8b. Snapshots use gridPoint; this loop runs to the end.
//! Ash Hart and TensorFold contributors; CUDA launch patterns by jschmied (PR #443).
const std = @import("std");
const cuda = @import("cuda");
const pool = @import("cuda_pool.zig");
const seg = @import("cuda_segments.zig");
const grid = @import("cuda_prefill_grid.zig");
const prefill = @import("cuda_prefill.zig");

pub const gridPoint = grid.gridPoint;

/// Grow an extent in place so it covers `tokens`. Moving a cache is a later package.
pub fn ensureExtent(p: *pool.Pool, eid: u64, tokens: usize) !usize {
    const need = grid.extentTokens(tokens);
    if (need > (try p.get(eid)).size) try p.resize(eid, need);
    return (try p.get(eid)).size;
}

/// One segment for this chunk. `fill` always; `stage` only when the table has device memory.
pub fn noteChunk(t: *seg.SegRows, base: usize, pos: usize, rows: usize, ring_base: ?usize) !bool {
    _ = try t.fill(&.{.{ .base = base, .pos = pos, .rows = rows, .ring_base = ring_base }}, grid.sparse_from);
    if (t.dev != null) try t.stage();
    return t.anySparse();
}

fn stageIds(e: *const prefill.Engine, ids: []const i32) !void {
    if (ids.len > e.b.rows) return error.BadRows;
    const view: cuda.DeviceBuffer = .{ .d = e.tri.s.d, .ptr = e.b.ids, .len = e.b.rows * 4 };
    try view.uploadAsync(0, std.mem.sliceAsBytes(ids), e.tri.s.handle);
}

pub const Job = struct {
    tokens: []const i32,
    tap_ctx: ?*anyopaque = null,
    on_taps: ?*const fn (*anyopaque, usize) anyerror!void = null,
    begin: usize = 0,
    busy: bool = false,
    limit: usize = grid.chunk_rows,
    /// null or 0 with chunked KDA selects the 64-token grid.
    prompt_grid: ?usize = null,
    /// Sample flag: the head runs on every chunk and overwrites the one-row logits buffer.
    head: bool = false,
    extents: ?*pool.Pool = null,
    extent: ?u64 = null,
    base: usize = 0,
    slot: usize = 0,
    ring: ?usize = null,
    segments: ?*seg.SegRows = null,
};

/// Validate the resume point and the chunk width. Returns the rows one chunk may take.
pub fn checkJob(len: usize, begin: usize, limit: usize, requested: ?usize, busy: bool) !usize {
    if (len == 0 or begin >= len) return error.BadChunk;
    const g = try grid.promptGrid(limit, requested, true);
    try grid.checkResume(begin, g);
    const step = grid.chunkLimit(busy, limit);
    if (g != 0 and step % g != 0) return error.BadGrid;
    return step;
}

/// Commit every chunk through `len`. Parity flips inside the chunk. The conv window is shifted there, not again here.
/// Returns the sequence position after the last row.
pub fn run(e: *prefill.Engine, parity: *[34]u1, job: Job) !usize {
    const step = try checkJob(job.tokens.len, job.begin, job.limit, job.prompt_grid, job.busy);
    var pos = job.begin;
    while (pos < job.tokens.len) pos = try chunkAt(e, parity, job, pos, @min(step, job.tokens.len - pos));
    return pos;
}

/// One chunk of `run`: the job's rows [pos, pos + rows) (rows within checkJob's step), committed as `run` commits each.
/// Returns the position after it. A sliced fill (cuda_engine's Fill) calls it once a step, `run` once a chunk.
pub fn chunkAt(e: *prefill.Engine, parity: *[34]u1, job: Job, pos: usize, rows: usize) !usize {
    const end = pos + rows;
    if (rows == 0 or end > job.tokens.len) return error.BadChunk;
    if (job.extents) |p| {
        const eid = job.extent orelse return error.NoExtent;
        const size = try ensureExtent(p, eid, end);
        if (size / 4 > e.s.pools) return error.ScoresScratch;
        e.cache.capacity = size;
    } else if (end > e.cache.capacity) return error.BadExtent;
    if (job.segments) |t| {
        const ring_base: ?usize = if (t.ring != null) job.slot * (job.ring orelse return error.BadRing) else null;
        _ = try noteChunk(t, job.base, pos, rows, ring_base);
    }
    try stageIds(e, job.tokens[pos..end]);
    try e.chunk(rows, pos, parity, job.head);
    if (job.on_taps) |hook| try hook(job.tap_ctx orelse return error.MissingTapContext, rows);
    return end;
}

/// One run of the loop in a prompt pass: rows [begin, end) of the prompt's first `end` tokens; `head`: the pass's last
/// run, whose last row's logits the first token is drawn from.
pub const Run = struct { begin: usize, end: usize, head: bool };

/// A prompt pass's runs of the loop, as cuda_engine's prefillWith makes them from `begin` (a resume, a live position or
/// 0): to the keep point when it lies past `begin` (`keep`: the state is kept after runs[0]); then, unless that is the
/// prompt's end (`draws` false: nothing is sampled), to the 64-token grid point below the end and the head's run from
/// there (a 5021-token prompt: 2048, 2048, 896, then 29). Each run is cut into chunks by `run`'s rule.
pub const Plan = struct {
    runs: [3]Run = undefined,
    n: usize = 0,
    keep: bool = false,
    draws: bool = true,

    pub fn of(len: usize, begin0: usize, keep_at: ?usize) Plan {
        var p: Plan = .{};
        var begin = begin0;
        if (keep_at) |k| if (k > begin) {
            p.add(.{ .begin = begin, .end = k, .head = false });
            p.keep = true;
            begin = k;
        };
        if (begin == len) {
            p.draws = false;
            return p;
        }
        const point = if (len % 64 == 0) len else len / 64 * 64;
        if (point > begin and point < len) p.add(.{ .begin = begin, .end = point, .head = false });
        p.add(.{ .begin = if (point < len and point > begin) point else begin, .end = len, .head = true });
        return p;
    }

    fn add(p: *Plan, r: Run) void {
        p.runs[p.n] = r;
        p.n += 1;
    }
};

/// One chunk of a plan: its run and rows [pos, pos + rows).
pub const Chunk = struct { run: usize, pos: usize, rows: usize };

/// Where a pass stands in its plan, a chunk at a time: the run, the next chunk's position, the run's chunk rows (0: the
/// run has not started, its rows are checked at its first chunk, as `run` checks them).
pub const Cursor = struct {
    run: usize = 0,
    pos: usize = 0,
    step: usize = 0,

    /// The next chunk (null once every run is done).
    pub fn next(c: *Cursor, p: *const Plan, limit: usize) !?Chunk {
        if (c.run >= p.n) return null;
        const r = p.runs[c.run];
        if (c.step == 0) {
            c.step = try checkJob(r.end, r.begin, limit, null, false);
            c.pos = r.begin;
        }
        return .{ .run = c.run, .pos = c.pos, .rows = @min(c.step, r.end - c.pos) };
    }

    /// The chunk `next` gave ran: whether it ended its run.
    pub fn ran(c: *Cursor, p: *const Plan, ch: Chunk) bool {
        c.pos = ch.pos + ch.rows;
        if (c.pos < p.runs[c.run].end) return false;
        c.run += 1;
        c.step = 0;
        return true;
    }
};

test "pool growth stays in place and a chunk is one segment" {
    var p = try pool.Pool.init(std.testing.allocator, 8192);
    defer p.deinit();
    const a = (try p.alloc(2048)).?;
    try std.testing.expectEqual(@as(usize, 4096), try ensureExtent(&p, a, 3000));
    const b = (try p.alloc(2048)).?;
    try std.testing.expectError(error.Overlap, ensureExtent(&p, a, 5000));
    try std.testing.expectError(error.BadExtent, ensureExtent(&p, b, 8192));
    var dense = try seg.SegRows.init(std.testing.allocator, 2048, 1, 2112);
    defer dense.deinit();
    try std.testing.expect(!try noteChunk(&dense, 0, 0, 2048, 0));
    var sparse = try seg.SegRows.init(std.testing.allocator, 8, 1, 2112);
    defer sparse.deinit();
    try std.testing.expect(try noteChunk(&sparse, 0, 2048, 8, 0));
    try std.testing.expectEqual(@as(i32, 1), sparse.plane(3)[3]);
    try std.testing.expectError(error.BadChunk, checkJob(0, 0, 2048, null, false));
    try std.testing.expectError(error.BadChunk, checkJob(5000, 5000, 2048, null, false));
    try std.testing.expectError(error.OffGrid, checkJob(5000, 100, 2048, null, false));
    try std.testing.expectEqual(@as(usize, 1024), try checkJob(5000, 2048, 2048, null, true));
    try std.testing.expectEqual(@as(?usize, 4992), gridPoint(5000, 0, 64));
}

/// The chunks the whole pass ran before sliced fills (cuda_engine's prefillWith up to 96964a78, its three `run`
/// calls written out as they were): [pos, rows, head, end of the run's tokens], in order.
fn wholeChunks(out: *std.ArrayList([4]usize), len: usize, begin0: usize, keep_at: ?usize) !void {
    const Ref = struct {
        fn run(o: *std.ArrayList([4]usize), tokens: usize, begin: usize, head: bool) !void {
            const step = try checkJob(tokens, begin, grid.chunk_rows, null, false);
            var pos = begin;
            while (pos < tokens) {
                const rows = @min(step, tokens - pos);
                try o.append(std.testing.allocator, .{ pos, rows, @intFromBool(head), tokens });
                pos += rows;
            }
        }
    };
    var begin = begin0;
    if (keep_at) |k| if (k > begin) {
        try Ref.run(out, k, begin, false);
        begin = k;
    };
    if (begin == len) return;
    const point = if (len % 64 == 0) len else len / 64 * 64;
    if (point > begin and point < len) try Ref.run(out, point, begin, false);
    try Ref.run(out, len, if (point < len and point > begin) point else begin, true);
}

test "a sliced pass runs the whole pass's chunks one at a time: same positions, rows, head and runs, the keep after its run" {
    const gpa = std.testing.allocator;
    var whole: std.ArrayList([4]usize) = .empty;
    defer whole.deinit(gpa);
    var sliced: std.ArrayList([4]usize) = .empty;
    defer sliced.deinit(gpa);
    const lens = [_]usize{ 1, 29, 63, 64, 65, 127, 2047, 2048, 2049, 4096, 5021, 6144, 38000, 38017, 38912 };
    var cases: usize = 0;
    for (lens) |len| {
        var begin: usize = 0;
        while (begin < len) : (begin += if (begin < 4096) 64 else 2048 + 64) {
            var keeps: [8]?usize = undefined;
            var nk: usize = 0;
            keeps[nk] = null;
            nk += 1;
            for ([_]usize{ 64, 2048, 4096, 4992, 6144, 37888 }) |k| if (k <= len) {
                keeps[nk] = k;
                nk += 1;
            };
            if (len % 64 == 0) {
                keeps[nk] = len;
                nk += 1;
            }
            for (keeps[0..nk]) |keep_at| {
                whole.clearRetainingCapacity();
                sliced.clearRetainingCapacity();
                try wholeChunks(&whole, len, begin, keep_at);
                const p = Plan.of(len, begin, keep_at);
                var c: Cursor = .{};
                var keep_after: ?usize = null; // the chunk count when the plan's keep is taken
                while (try c.next(&p, grid.chunk_rows)) |ch| {
                    const r = p.runs[ch.run];
                    try sliced.append(gpa, .{ ch.pos, ch.rows, @intFromBool(r.head), r.end });
                    if (c.ran(&p, ch) and p.keep and ch.run == 0) keep_after = sliced.items.len;
                }
                try std.testing.expectEqualSlices([4]usize, whole.items, sliced.items);
                // the state is kept where the whole pass kept it: after the keep point's run, at the keep point
                const kept = if (keep_at) |k| k > begin else false;
                try std.testing.expectEqual(kept, p.keep);
                if (kept) try std.testing.expectEqual(keep_at.?, sliced.items[keep_after.? - 1][0] + sliced.items[keep_after.? - 1][1]);
                // nothing is drawn when the keep point is the prompt's end (the lanes' marks); else the head's run ends it
                try std.testing.expectEqual(!(kept and keep_at.? == len), p.draws);
                if (p.draws) try std.testing.expectEqual(@as(usize, 1), sliced.items[sliced.items.len - 1][2]);
                cases += 1;
            }
        }
    }
    try std.testing.expect(cases > 300);
}

test "compile the prompt loop without a GPU" {
    std.testing.refAllDecls(@This());
    if (std.c.getenv("TF_GLM_P8_COMPILE_PROBE") != null) {
        var engine: prefill.Engine = undefined;
        var parity: [34]u1 = @splat(0);
        _ = try run(&engine, &parity, .{ .tokens = &.{ 1, 2, 3 } });
    }
}
