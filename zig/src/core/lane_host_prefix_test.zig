//! LaneHost's prefix wait (TF_GLM_PREFIX_WAIT, wojo's GLM 0110 on the Zig lane host): with prompts filling in slices
//! between rounds (GLM's TF_GLM_FILL), a request whose prompt shares a cut another fill is still computing waits for
//! that fill to keep it, then resumes from it, instead of computing the same prefix beside it (the fake backend).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;

const gpa = std.testing.allocator;

/// A lane host over the fake backend (sliced or whole passes) with GLM's kept-state policy: shared cuts of 16 tokens
/// or more, marks 4 apart. Requests are submitted before `start`, so every one is queued at the first admission.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    store: pc.Store,
    host: LaneHost,

    const vt: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop };
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }

    const Opts = struct { sliced: bool = true, slice: usize = 2, wait_ns: i96 = 30 * std.time.ns_per_s, lanes: u32 = 8 };

    fn init(r: *Rig, o: Opts) !void {
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa, .slice = o.slice };
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, if (o.sliced) r.target.backendSliced() else r.target.backend(), r.clock.clock());
        r.store = pc.Store.init(gpa, .{ .ptr = &r.target, .vtable = &vt }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 24);
        r.store.policy = .{ .recipe = true, .shared_least = 16 };
        r.host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = o.lanes });
        r.host.cache = &r.store;
        r.host.prefix_wait_ns = o.wait_ns;
    }

    fn deinit(r: *Rig) void {
        r.host.stop();
        r.store.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
    }

    fn counts(r: *Rig) LaneHost.HostCounts {
        r.host.mutex.lockUncancelable(std.testing.io);
        defer r.host.mutex.unlock(std.testing.io);
        return r.host.host_counts;
    }

    /// Until a request waits for a prefix (2 s at most: a host that never waits fails here, never hangs).
    fn waitForWaiter(r: *Rig) !void {
        for (0..2000) |_| {
            if (r.counts().prefix_waits > 0) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.NothingWaited;
    }

    fn waiting(r: *Rig) usize {
        r.host.mutex.lockUncancelable(std.testing.io);
        defer r.host.mutex.unlock(std.testing.io);
        return r.host.waitingCount();
    }
};

/// One request: its prompt's last two tokens are the generation prompt, its system block `system_len` long.
const Turn = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,
    prompt: std.ArrayList(u32) = .empty,
    request: Request = undefined,
    cut: [1]u32 = undefined,

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Turn = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .logprobs => {},
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
        }
    }

    /// `system` then `tail`, then the generation prompt; the system block's end is the request's shared cut.
    fn make(b: *Turn, system: []const u32, tail: []const u32) !void {
        try b.prompt.appendSlice(gpa, system);
        try b.prompt.appendSlice(gpa, tail);
        try b.prompt.appendSlice(gpa, &.{ 7, 7 });
        b.cut = .{@intCast(system.len)};
        b.request = .{ .prompt = b.prompt.items, .max_tokens = 6, .history_len = @intCast(b.prompt.items.len - 2), .shared_prefixes = &b.cut };
    }

    fn submit(b: *Turn, e: api.Engine, id: Id) !void {
        try e.submit(id, &b.request, .{ .ctx = b, .event = event });
    }

    fn wait(b: *Turn) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |reason| return reason;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }

    /// Its reply is the fake's own continuation of its prompt (any slip in a resumed state changes it).
    fn expectSerial(b: *Turn) !void {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, b.prompt.items);
        for (b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
    }

    fn deinit(b: *Turn) void {
        b.tokens.deinit(gpa);
        b.prompt.deinit(gpa);
    }
};

fn systemOf(n: usize, salt: u32) ![]u32 {
    const s = try gpa.alloc(u32, n);
    for (s, 0..) |*x, i| x.* = 20 + @as(u32, @intCast((i * 7 + salt) % 53));
    return s;
}

test "sliced fills on one shared system prompt compute it once: the oldest fills, the others wait and resume at its cut" {
    var r: Rig = undefined;
    try r.init(.{});
    defer r.deinit();
    const sys = try systemOf(40, 0);
    defer gpa.free(sys);
    const other = try systemOf(40, 5); // a prompt sharing nothing: it neither waits nor is waited for
    defer gpa.free(other);
    var turns: [5]Turn = @splat(.{});
    defer for (&turns) |*t| t.deinit();
    for (turns[0..4], 0..) |*t, i| try t.make(sys, &.{ 100 + @as(u32, @intCast(i)), 3, 1, 4, 1, 5 });
    try turns[4].make(other, &.{ 9, 9, 9, 9, 9, 9 });
    const e = r.host.engine();
    for (&turns, 1..) |*t, id| try t.submit(e, id);
    try r.host.start();
    for (&turns) |*t| {
        try std.testing.expectEqual(Reason.length, t.wait());
        try t.expectSerial();
    }
    try std.testing.expectEqual(@as(?u32, 0), turns[0].cached);
    for (turns[1..4]) |*t| try std.testing.expectEqual(@as(?u32, 40), t.cached); // resumed at the leader's cut
    try std.testing.expectEqual(@as(?u32, 0), turns[4].cached);
    const c = r.counts();
    try std.testing.expectEqual(@as(u64, 3), c.prefix_waits);
    try std.testing.expectEqual(@as(u64, 3 * 40), c.prefix_wait_tokens);
    try std.testing.expectEqual(@as(u64, 0), c.prefix_wait_timeouts);
    // the shared 40 tokens computed once: the leader's whole prompt, the others' tails, the unrelated prompt's whole
    try std.testing.expectEqual(@as(u64, 48 + 3 * 8 + 48), c.computed_tokens);
    try std.testing.expectEqual(@as(usize, 0), r.waiting());
}

test "a waiter of a cancelled fill goes on at once: it prefills the prefix itself, exactly" {
    var r: Rig = undefined;
    try r.init(.{ .slice = 1 });
    defer r.deinit();
    const sys = try systemOf(20000, 1); // thousands of steps: the cancel lands mid-fill
    defer gpa.free(sys);
    var turns: [2]Turn = @splat(.{});
    defer for (&turns) |*t| t.deinit();
    try turns[0].make(sys, &.{ 1, 2, 3, 4 });
    try turns[1].make(sys, &.{ 5, 6, 7, 8 });
    const e = r.host.engine();
    try turns[0].submit(e, 1);
    try turns[1].submit(e, 2);
    try r.host.start();
    try r.waitForWaiter();
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, turns[0].wait());
    try std.testing.expectEqual(Reason.length, turns[1].wait());
    try turns[1].expectSerial();
    try std.testing.expectEqual(@as(?u32, 0), turns[1].cached); // nothing was kept: it computed the prefix
    const c = r.counts();
    try std.testing.expectEqual(@as(u64, 1), c.prefix_waits);
    try std.testing.expectEqual(@as(u64, 0), c.prefix_wait_timeouts);
    try std.testing.expectEqual(@as(usize, 0), r.waiting());
}

test "a cancelled waiter leaves the wait list once; the fill it waited for runs on" {
    var r: Rig = undefined;
    try r.init(.{ .slice = 1 });
    defer r.deinit();
    const sys = try systemOf(20000, 2);
    defer gpa.free(sys);
    var turns: [2]Turn = @splat(.{});
    defer for (&turns) |*t| t.deinit();
    try turns[0].make(sys, &.{ 1, 2, 3, 4 });
    try turns[1].make(sys, &.{ 5, 6, 7, 8 });
    const e = r.host.engine();
    try turns[0].submit(e, 1);
    try turns[1].submit(e, 2);
    try r.host.start();
    try r.waitForWaiter();
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, turns[1].wait());
    try std.testing.expectEqual(Reason.length, turns[0].wait());
    try turns[0].expectSerial();
    try std.testing.expectEqual(@as(usize, 0), r.waiting());
}

test "a prefix wait is bounded (TF_GLM_PREFIX_WAIT_MS): past it the waiter prefills beside the fill, exactly" {
    var r: Rig = undefined;
    try r.init(.{ .slice = 1, .wait_ns = 1 }); // the bound expires at the first check
    defer r.deinit();
    const sys = try systemOf(4000, 3);
    defer gpa.free(sys);
    var turns: [2]Turn = @splat(.{});
    defer for (&turns) |*t| t.deinit();
    try turns[0].make(sys, &.{ 1, 2, 3, 4 });
    try turns[1].make(sys, &.{ 5, 6, 7, 8 });
    const e = r.host.engine();
    try turns[0].submit(e, 1);
    try turns[1].submit(e, 2);
    try r.host.start();
    for (&turns) |*t| {
        try std.testing.expectEqual(Reason.length, t.wait());
        try t.expectSerial();
    }
    try std.testing.expectEqual(@as(?u32, 0), turns[1].cached);
    const c = r.counts();
    try std.testing.expectEqual(@as(u64, 1), c.prefix_waits);
    try std.testing.expectEqual(@as(u64, 1), c.prefix_wait_timeouts);
}

test "whole prompt passes are unchanged: nothing waits (each pass ends before the next is admitted), the cut still serves" {
    var r: Rig = undefined;
    try r.init(.{ .sliced = false });
    defer r.deinit();
    const sys = try systemOf(40, 0);
    defer gpa.free(sys);
    var turns: [4]Turn = @splat(.{});
    defer for (&turns) |*t| t.deinit();
    for (&turns, 0..) |*t, i| try t.make(sys, &.{ 100 + @as(u32, @intCast(i)), 3, 1, 4, 1, 5 });
    const e = r.host.engine();
    for (&turns, 1..) |*t, id| try t.submit(e, id);
    try r.host.start();
    for (&turns) |*t| {
        try std.testing.expectEqual(Reason.length, t.wait());
        try t.expectSerial();
    }
    for (turns[1..]) |*t| try std.testing.expectEqual(@as(?u32, 40), t.cached);
    const c = r.counts();
    try std.testing.expectEqual(@as(u64, 0), c.prefix_waits);
    try std.testing.expectEqual(@as(u64, 48 + 3 * 8), c.computed_tokens);
}

test "the prefix wait off (0): sliced fills on one system prompt each compute it, as before" {
    var r: Rig = undefined;
    try r.init(.{ .wait_ns = 0 });
    defer r.deinit();
    const sys = try systemOf(40, 0);
    defer gpa.free(sys);
    var turns: [3]Turn = @splat(.{});
    defer for (&turns) |*t| t.deinit();
    for (&turns, 0..) |*t, i| try t.make(sys, &.{ 100 + @as(u32, @intCast(i)), 3, 1, 4, 1, 5 });
    const e = r.host.engine();
    for (&turns, 1..) |*t, id| try t.submit(e, id);
    try r.host.start();
    for (&turns) |*t| {
        try std.testing.expectEqual(Reason.length, t.wait());
        try t.expectSerial();
        try std.testing.expectEqual(@as(?u32, 0), t.cached);
    }
    try std.testing.expectEqual(@as(u64, 0), r.counts().prefix_waits);
}

test "a shared run under the shared-prefix minimum is not waited for" {
    var r: Rig = undefined;
    try r.init(.{});
    defer r.deinit();
    const sys = try systemOf(12, 0); // 12 < shared_least 16
    defer gpa.free(sys);
    var turns: [2]Turn = @splat(.{});
    defer for (&turns) |*t| t.deinit();
    for (&turns, 0..) |*t, i| try t.make(sys, &.{ 100 + @as(u32, @intCast(i)), 3, 1, 4, 1, 5 });
    const e = r.host.engine();
    for (&turns, 1..) |*t, id| try t.submit(e, id);
    try r.host.start();
    for (&turns) |*t| {
        try std.testing.expectEqual(Reason.length, t.wait());
        try t.expectSerial();
    }
    try std.testing.expectEqual(@as(u64, 0), r.counts().prefix_waits);
}

test "TF_GLM_PREFIX_WAIT / TF_GLM_PREFIX_WAIT_MS: on by default for 60 s, a bound in ms, 0 off; anything else refused" {
    const ms = std.time.ns_per_ms;
    try std.testing.expectEqual(@as(i96, 60_000 * ms), try LaneHost.prefixWaitSetting(null, null));
    try std.testing.expectEqual(@as(i96, 60_000 * ms), try LaneHost.prefixWaitSetting("1", " "));
    try std.testing.expectEqual(@as(i96, 2_500 * ms), try LaneHost.prefixWaitSetting(null, "2500"));
    try std.testing.expectEqual(@as(i96, 0), try LaneHost.prefixWaitSetting("0", "2500"));
    try std.testing.expectEqual(@as(i96, 0), try LaneHost.prefixWaitSetting("1", "0"));
    try std.testing.expectError(error.BadPrefixWait, LaneHost.prefixWaitSetting("yes", null));
    try std.testing.expectError(error.BadPrefixWait, LaneHost.prefixWaitSetting(null, "-5"));
    try std.testing.expectError(error.BadPrefixWait, LaneHost.prefixWaitSetting(null, "1s"));
}
