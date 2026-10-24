//! Pictures in the prompt cache and its durable tier (the DeepSeek port's prompt_cache_media_test, in GLM's shapes:
//! Request.media's rows and canvases, cache_key and cache_media): what the store and the lane host hand a tier for a
//! prompt with pictures (Want.images/media at lookup, Write.media/raw at keep), memory keys that tell two pictures at
//! the same rows apart, the media guards a request must pass to be kept at all, 0111 and TF_GLM_IMAGE_MARK_SNAP, over
//! the fake lane backend and the in-memory tier.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const tt = @import("lane_host_tier_test.zig");
const LaneHost = api.LaneHost;
const Request = api.Request;
const Reason = api.Reason;
const Box = tt.Box;
const Rig = tt.Rig;
const gpa = std.testing.allocator;
const io = std.testing.io;

/// The in-memory tier, recording what the store hands it for pictures (lookups and keeps), everything else its own.
const Rec = struct {
    inner: pc.Tier,
    mutex: std.Io.Mutex = .init,
    lookups: u32 = 0,
    picture_lookups: u32 = 0, // with Want.images
    want_media: [4]pc.Media = undefined,
    want_n: usize = 0,
    keeps: u32 = 0,
    picture_keeps: u32 = 0,
    kept_media: [4]pc.Media = undefined,
    kept_n: usize = 0,
    kept_raw: [64]u32 = undefined,
    kept_raw_n: usize = 0,
    kept_key: [64]u32 = undefined,
    kept_key_n: usize = 0,

    fn tier(r: *Rec) pc.Tier {
        return .{ .ptr = r, .vtable = &vtable };
    }
    const vtable: pc.Tier.VTable = .{ .lookup = lookup, .begin = begin, .poll = poll, .cancel = cancel, .keep = keep, .release = release, .share = share, .touch = touch, .forget = forget, .pump = pump, .busy = busy, .flush = flush, .counts = counts };

    fn of(p: *anyopaque) *Rec {
        return @ptrCast(@alignCast(p));
    }
    fn in(p: *anyopaque) pc.Tier {
        return of(p).inner;
    }
    fn begin(p: *anyopaque, point: *const pc.Point) anyerror!u64 {
        const t = in(p);
        return t.vtable.begin(t.ptr, point);
    }
    fn poll(p: *anyopaque, job: u64, budget: u64) pc.Poll {
        const t = in(p);
        return t.vtable.poll(t.ptr, job, budget);
    }
    fn cancel(p: *anyopaque, job: u64) void {
        const t = in(p);
        t.vtable.cancel(t.ptr, job);
    }
    fn release(p: *anyopaque, point: u64, saved: pc.Saved) void {
        const t = in(p);
        t.vtable.release.?(t.ptr, point, saved);
    }
    fn share(p: *anyopaque, point: u64) void {
        const t = in(p);
        t.vtable.share.?(t.ptr, point);
    }
    fn touch(p: *anyopaque, point: u64) void {
        const t = in(p);
        t.vtable.touch.?(t.ptr, point);
    }
    fn forget(p: *anyopaque, point: u64) void {
        const t = in(p);
        t.vtable.forget(t.ptr, point);
    }
    fn pump(p: *anyopaque, budget: u64) void {
        const t = in(p);
        t.vtable.pump.?(t.ptr, budget);
    }
    fn busy(p: *anyopaque) bool {
        const t = in(p);
        return t.vtable.busy.?(t.ptr);
    }
    fn flush(p: *anyopaque, budget: u64) u64 {
        const t = in(p);
        return t.vtable.flush(t.ptr, budget);
    }
    fn counts(p: *anyopaque, out: []pc.TierCounts) usize {
        const t = in(p);
        return t.vtable.counts(t.ptr, out);
    }
    fn lookup(p: *anyopaque, want: *const pc.Want) ?pc.Point {
        const r = of(p);
        r.mutex.lockUncancelable(io);
        r.lookups += 1;
        if (want.images) r.picture_lookups += 1;
        r.want_n = @min(want.media.len, r.want_media.len);
        @memcpy(r.want_media[0..r.want_n], want.media[0..r.want_n]);
        r.mutex.unlock(io);
        const t = in(p);
        return t.vtable.lookup(t.ptr, want);
    }
    fn keep(p: *anyopaque, w: *const pc.Write) ?u64 {
        const r = of(p);
        r.mutex.lockUncancelable(io);
        r.keeps += 1;
        if (w.images) {
            r.picture_keeps += 1;
            r.kept_n = @min(w.media.len, r.kept_media.len);
            @memcpy(r.kept_media[0..r.kept_n], w.media[0..r.kept_n]);
            r.kept_raw_n = @min(w.raw.len, r.kept_raw.len);
            @memcpy(r.kept_raw[0..r.kept_raw_n], w.raw[0..r.kept_raw_n]);
            r.kept_key_n = @min(w.key.len, r.kept_key.len);
            @memcpy(r.kept_key[0..r.kept_key_n], w.key[0..r.kept_key_n]);
        }
        r.mutex.unlock(io);
        const t = in(p);
        return t.vtable.keep(t.ptr, w);
    }
};

/// A rig whose tier is recorded (`rec` wraps the rig's in-memory tier).
fn withRec(rec: *Rec, o: Rig.Opts) !*Rig {
    const r = try Rig.createWith(.{ .load_ms = 5, .flush_ms = 1000 }, o);
    rec.* = .{ .inner = r.tier.tier() };
    r.store.setTier(rec.tier(), 0);
    return r;
}

/// A picture of 3 rows (28 x 84) whose canvas is `fill`.
fn canvas(comptime fill: u8) [3 * 28 * 84]u8 {
    return @splat(fill);
}

/// A prompt of 14 tokens with one picture at [2, 5) (the placeholder 99 at its rows), keyed by the picture's canvas.
const Pic = struct {
    prompt: [14]u32 = .{ 3, 1, 99, 99, 99, 5, 9, 2, 6, 5, 7, 7, 4, 4 },
    rows: [3]u32 = .{ 2, 3, 4 },
    items: [1]lanes.MediaItem,
    media: lanes.Media = undefined,
    key: []u32 = &.{},
    ids: []pc.Media = &.{},

    fn init(p: *Pic, px: []const u8) !void {
        p.items = .{.{ .h = 28, .w = 84, .data = px }};
        p.media = .{ .rows = &p.rows, .items = &p.items };
        const k = (try pc.prompt_media.keyOf(gpa, &p.prompt, &p.media)).?;
        p.key = k.keyed;
        p.ids = k.media;
    }
    fn deinit(p: *Pic) void {
        gpa.free(p.key);
        gpa.free(p.ids);
    }
    fn request(p: *const Pic, max_tokens: u32) Request {
        return .{ .prompt = &p.prompt, .media = &p.media, .cache_key = p.key, .cache_media = p.ids, .max_tokens = max_tokens, .history_len = 12 };
    }
};

test "a prompt with pictures: its lookups carry the pictures' identities, its keeps their identities and model ids; a disk hit resumes it" {
    var rec: Rec = undefined;
    const r = try withRec(&rec, .{});
    defer r.destroy();
    const px = canvas(1);
    var p: Pic = .{ .items = undefined };
    try p.init(&px);
    defer p.deinit();
    // a point of 8 tokens on "disk" under the content key (its state the prompt's own model ids: the fake backend's
    // state is its history)
    _ = try r.tier.put(p.key[0..8], 8, std.mem.sliceAsBytes(p.prompt[0..8]), 0, 0, false);
    var b: Box = .{};
    defer b.deinit();
    const req = p.request(4);
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // restored, then resumed
    try b.exact(&p.prompt);
    rec.mutex.lockUncancelable(io);
    defer rec.mutex.unlock(io);
    try std.testing.expect(rec.picture_lookups >= 1);
    try std.testing.expectEqual(@as(usize, 1), rec.want_n);
    try std.testing.expect(rec.want_media[0].eql(p.ids[0]));
    try std.testing.expect(rec.picture_keeps >= 1);
    try std.testing.expectEqual(@as(usize, 1), rec.kept_n);
    try std.testing.expect(rec.kept_media[0].eql(p.ids[0]));
    try std.testing.expectEqualSlices(u32, &p.prompt, rec.kept_raw[0..rec.kept_raw_n]); // the model's ids, not the key
    try std.testing.expectEqualSlices(u32, p.key[0..rec.kept_key_n], rec.kept_key[0..rec.kept_key_n]); // the content key
}

test "two different pictures at the same rows never share a kept state; the same picture again resumes it" {
    var rec: Rec = undefined;
    const r = try withRec(&rec, .{});
    defer r.destroy();
    const pa = canvas(1);
    const pb = canvas(2);
    var a: Pic = .{ .items = undefined };
    try a.init(&pa);
    defer a.deinit();
    var b2: Pic = .{ .items = undefined };
    try b2.init(&pb);
    defer b2.deinit();
    try std.testing.expectEqualSlices(u32, &a.prompt, &b2.prompt); // the model's ids are the same placeholders
    const runs = [_]struct { p: *const Pic, cached: u32 }{ .{ .p = &a, .cached = 0 }, .{ .p = &b2, .cached = 0 }, .{ .p = &a, .cached = 12 } };
    for (runs, 1..) |run, id| {
        var box: Box = .{};
        defer box.deinit();
        const req = run.p.request(2);
        try r.host.engine().submit(id, &req, box.sink());
        try std.testing.expectEqual(Reason.length, box.wait());
        try std.testing.expectEqual(@as(?u32, run.cached), box.cached);
        try box.exact(&run.p.prompt);
    }
}

test "tierMedia: a request's media count only when they name exactly the rows the engine fills" {
    const px = canvas(1);
    var p: Pic = .{ .items = undefined };
    try p.init(&px);
    defer p.deinit();
    const ok = p.request(1);
    try std.testing.expectEqual(@as(usize, 1), LaneHost.tierMedia(&ok).len);
    try std.testing.expect(LaneHost.cacheable(&ok));
    try std.testing.expectEqualSlices(u32, p.key, LaneHost.keyOf(&ok));
    var moved = p.ids[0];
    moved.start = 3; // a span elsewhere than the rows
    var r2 = ok;
    r2.cache_media = (&moved)[0..1];
    try std.testing.expectEqual(@as(usize, 0), LaneHost.tierMedia(&r2).len);
    try std.testing.expect(!LaneHost.cacheable(&r2));
    var longer = p.ids[0];
    longer.len = 4; // another length
    r2.cache_media = (&longer)[0..1];
    try std.testing.expect(!LaneHost.cacheable(&r2));
    var unkeyed = ok;
    unkeyed.cache_key = &.{}; // TF_GLM_IMAGE_CACHE=0, features from another tower
    try std.testing.expect(!LaneHost.cacheable(&unkeyed));
    try std.testing.expectEqualSlices(u32, &p.prompt, LaneHost.keyOf(&unkeyed));
    const text: Request = .{ .prompt = &p.prompt, .max_tokens = 1 };
    try std.testing.expect(LaneHost.cacheable(&text) and LaneHost.tierMedia(&text).len == 0);
}

test "a prompt with pictures whose cache_media name other rows: no lookup, nothing kept" {
    var rec: Rec = undefined;
    const r = try withRec(&rec, .{});
    defer r.destroy();
    const px = canvas(1);
    var p: Pic = .{ .items = undefined };
    try p.init(&px);
    defer p.deinit();
    _ = try r.tier.put(p.key[0..8], 8, std.mem.sliceAsBytes(p.prompt[0..8]), 0, 0, false);
    var moved = p.ids[0];
    moved.start = 1;
    var req = p.request(4);
    req.cache_media = (&moved)[0..1];
    var b: Box = .{};
    defer b.deinit();
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached); // the point on disk is not resumed
    rec.mutex.lockUncancelable(io);
    defer rec.mutex.unlock(io);
    try std.testing.expectEqual(@as(u32, 0), rec.lookups);
    try std.testing.expectEqual(@as(u32, 0), rec.keeps);
    try std.testing.expectEqual(@as(usize, 0), r.store.entries.items.len);
}

test "a prompt with pictures and no content key (TF_GLM_IMAGE_CACHE=0): the tier is never asked, nothing is kept" {
    var rec: Rec = undefined;
    const r = try withRec(&rec, .{});
    defer r.destroy();
    const px = canvas(1);
    var p: Pic = .{ .items = undefined };
    try p.init(&px);
    defer p.deinit();
    var req = p.request(4);
    req.cache_key = &.{};
    req.cache_media = &.{};
    var b: Box = .{};
    defer b.deinit();
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    rec.mutex.lockUncancelable(io);
    defer rec.mutex.unlock(io);
    try std.testing.expectEqual(@as(u32, 0), rec.lookups);
    try std.testing.expectEqual(@as(u32, 0), rec.keeps);
    try std.testing.expectEqual(@as(usize, 0), r.store.entries.items.len);
}

test "a text prompt's lookups and keeps carry no pictures" {
    var rec: Rec = undefined;
    const r = try withRec(&rec, .{});
    defer r.destroy();
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7, 7, 5, 3 };
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &prompt, .max_tokens = 4, .history_len = 12 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    rec.mutex.lockUncancelable(io);
    defer rec.mutex.unlock(io);
    try std.testing.expect(rec.lookups >= 1 and rec.keeps >= 1);
    try std.testing.expectEqual(@as(u32, 0), rec.picture_lookups);
    try std.testing.expectEqual(@as(u32, 0), rec.picture_keeps);
}

test "0111: a kept picture state is never a conversation's continuation" {
    const r = try Rig.create(.{ .load_ms = 5, .flush_ms = 1000 });
    defer r.destroy();
    const px = canvas(1);
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(10 + i % 7);
    for (2..5) |i| prompt[i] = 99;
    const rows = [_]u32{ 2, 3, 4 };
    const items = [_]lanes.MediaItem{.{ .h = 28, .w = 84, .data = &px }};
    const m: lanes.Media = .{ .rows = &rows, .items = &items };
    const k = (try pc.prompt_media.keyOf(gpa, &prompt, &m)).?;
    defer gpa.free(k.keyed);
    defer gpa.free(k.media);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &prompt, .media = &m, .cache_key = k.keyed, .cache_media = k.media, .max_tokens = 2, .history_len = 38 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    var own: ?*pc.Entry = null;
    for (r.store.entries.items) |e| if (e.at == 38) {
        own = e;
    };
    const e = own orelse return error.NotKept;
    try std.testing.expect(e.images and e.own and e.chat != 0);
    // a later prompt repeating the state's ids past its conversation's first prompt and chat_match more: no continuation
    r.store.roots.put(gpa, e.chat, 1) catch {};
    var later: [48]u32 = @splat(8);
    @memcpy(later[0..38], k.keyed[0..38]);
    try std.testing.expect(r.store.continuedChat(&later) == null);
    e.images = false; // (the same state as text would be one: the check is the picture flag's)
    try std.testing.expect(r.store.continuedChat(&later) == e);
}

test "TF_GLM_IMAGE_MARK_SNAP=1: a mark inside a picture keeps its state at the picture's start instead" {
    const r = try Rig.create(.{ .load_ms = 5, .flush_ms = 1000 });
    defer r.destroy();
    r.host.mark_snap = true;
    const px = canvas(1);
    var p: Pic = .{ .items = undefined };
    try p.init(&px);
    defer p.deinit();
    // the picture at [10, 13) holds the history's cut (12)
    p.prompt = .{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 99, 99, 99, 4 };
    p.rows = .{ 10, 11, 12 };
    gpa.free(p.key);
    gpa.free(p.ids);
    const k = (try pc.prompt_media.keyOf(gpa, &p.prompt, &p.media)).?;
    p.key = k.keyed;
    p.ids = k.media;
    var b: Box = .{};
    defer b.deinit();
    const req = p.request(2);
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    var saw_start = false;
    for (r.store.entries.items) |e| {
        try std.testing.expect(!(e.at > 10 and e.at < 13)); // no state cut inside the picture
        saw_start = saw_start or e.at == 10;
    }
    try std.testing.expect(saw_start);
    // off (the default): the history's cut inside the picture is kept as it is
    const r2 = try Rig.create(.{ .load_ms = 5, .flush_ms = 1000 });
    defer r2.destroy();
    var b2: Box = .{};
    defer b2.deinit();
    try r2.host.engine().submit(1, &req, b2.sink());
    try std.testing.expectEqual(Reason.length, b2.wait());
    var saw_cut = false;
    for (r2.store.entries.items) |e| saw_cut = saw_cut or e.at == 12;
    try std.testing.expect(saw_cut);
}
