//! Spill v2's GLM tier with pictures (the DeepSeek port's spill-pictures tests, in GLM's shapes): a point whose cut a
//! picture reaches is written as format v3 with each picture's span and full SHA-256 and the prompt's model ids, found
//! only for the same pictures in the same places, restored from disk and after a restart byte for byte, and rank 1
//! writes its copy from its own model ids and the keep note's pictures. Two ranks in one thread (cuda_spill_tier_rig).
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const tier = @import("cuda_spill_tier.zig");
const rig = @import("cuda_spill_tier_rig.zig");
const st = @import("cuda_spill_state.zig");
const wire = @import("cuda_spill_wire.zig");

const spill = core.spill;
const format = spill.format;
const testing = std.testing;
const HostEngine = rig.HostEngine;

const at = 2112; // one whole block and a 64-token tail
/// GLM's image token: the model id at every picture row
const placeholder: u32 = 154854;

fn digestOf(tag: []const u8) format.Digest {
    var d: format.Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash(tag, &d, .{});
    return d;
}

/// A prompt with pictures: `raw` the model ids (the placeholder at every picture row and nowhere else), `keyed` its
/// content key (each picture's rows made from its digest, as the server makes them), `media` each picture's identity.
const Pics = struct {
    gpa: std.mem.Allocator,
    raw: []u32,
    keyed: []u32,
    media: []format.Media,

    fn init(gpa: std.mem.Allocator, n: usize, seed: u32, spans: []const [2]u32, digests: []const format.Digest) !Pics {
        const raw = try gpa.alloc(u32, n);
        errdefer gpa.free(raw);
        _ = rig.idsOf(raw, seed);
        for (raw) |*t| if (t.* == placeholder) {
            t.* = 7; // (a text id, never the placeholder)
        };
        const media = try gpa.alloc(format.Media, spans.len);
        errdefer gpa.free(media);
        for (spans, digests, media) |sp, d, *m| {
            m.* = .{ .start = sp[0], .len = sp[1], .digest = d };
            @memset(raw[sp[0]..][0..sp[1]], placeholder);
        }
        const keyed = try gpa.dupe(u32, raw);
        for (media) |m| for (0..m.len) |j| {
            keyed[m.start + j] = format.pictureId(&m.digest, @intCast(j));
        };
        return .{ .gpa = gpa, .raw = raw, .keyed = keyed, .media = media };
    }
    fn deinit(p: *Pics) void {
        p.gpa.free(p.raw);
        p.gpa.free(p.keyed);
        p.gpa.free(p.media);
    }
};

fn pointKeyOf(keyed: []const u32, media: []const format.Media) format.Digest {
    return format.pointKey(keyed, @intCast(st.block_tokens), media);
}

test "pointPictures: the pictures before the cut, the placeholder; disagreeing ids, an unnamed picture or 257 pictures give none" {
    const gpa = testing.allocator;
    var p = try Pics.init(gpa, 400, 1, &.{ .{ 10, 20 }, .{ 100, 50 }, .{ 300, 10 } }, &.{ digestOf("a"), digestOf("b"), digestOf("c") });
    defer p.deinit();
    const pp = tier.pointPictures(p.keyed, p.raw, p.media, 120).?; // the second straddles the cut, the third past it
    try testing.expectEqual(@as(usize, 2), pp.media.len);
    try testing.expectEqual(placeholder, pp.placeholder);
    try testing.expectEqual(@as(usize, 0), tier.pointPictures(p.keyed, p.raw, p.media, 10).?.media.len); // a text point
    try testing.expect(tier.pointPictures(p.keyed, p.raw, p.media[1..], 120) == null); // the first picture unnamed
    var other = p.media[0];
    other.digest[0] ^= 1; // another picture's ids
    try testing.expect(tier.pointPictures(p.keyed, p.raw, &.{ other, p.media[1], p.media[2] }, 120) == null);
    p.raw[15] = 5; // a picture row that does not hold the placeholder
    try testing.expect(tier.pointPictures(p.keyed, p.raw, p.media, 120) == null);
    p.raw[15] = placeholder;
    p.raw[50] = 6; // raw and keyed apart outside the pictures
    try testing.expect(tier.pointPictures(p.keyed, p.raw, p.media, 120) == null);
    // 256 pictures before the cut are recorded, 257 are not (a keep note carries 256 at most)
    var spans: [wire.max_note_media + 1][2]u32 = undefined;
    var ds: [wire.max_note_media + 1]format.Digest = undefined;
    for (&spans, &ds, 0..) |*sp, *d, i| {
        sp.* = .{ @intCast(2 * i), 1 };
        d.* = digestOf(std.mem.asBytes(&i));
    }
    var many = try Pics.init(gpa, 1024, 2, &spans, &ds);
    defer many.deinit();
    try testing.expectEqual(@as(usize, wire.max_note_media), tier.pointPictures(many.keyed, many.raw, many.media, 2 * wire.max_note_media).?.media.len);
    try testing.expect(tier.pointPictures(many.keyed, many.raw, many.media, 2 * wire.max_note_media + 1) == null);
}

test "followerIds: rank 0's ids from this rank's own and the noted pictures; a picture this rank has elsewhere gives none" {
    const gpa = testing.allocator;
    var p = try Pics.init(gpa, 200, 3, &.{ .{ 10, 20 }, .{ 100, 50 } }, &.{ digestOf("a"), digestOf("b") });
    defer p.deinit();
    var keyed: [120]u32 = undefined;
    var raw: [120]u32 = undefined;
    try testing.expect(tier.followerIds(p.raw, 120, p.media, placeholder, &keyed, &raw));
    try testing.expectEqualSlices(u32, p.keyed[0..120], &keyed);
    try testing.expectEqualSlices(u32, p.raw[0..120], &raw);
    // this rank's pictures one row later: a placeholder outside the noted span, and a noted row without one
    var moved = try gpa.dupe(u32, p.raw);
    defer gpa.free(moved);
    moved[10] = 9;
    moved[30] = placeholder;
    try testing.expect(!tier.followerIds(moved, 120, p.media, placeholder, &keyed, &raw));
    try testing.expect(!tier.followerIds(p.raw, 120, p.media, placeholder + 1, &keyed, &raw)); // another placeholder
}

test "pictures: a point is written on both ranks with its pictures, restored from disk byte for byte and resumes exactly; after a restart too; two different same-size pictures at the same rows, or the same seed bytes of another digest, miss" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "pictures", rig.Pair.settingsFor(1));
    defer p.deinit();
    const da = digestOf("picture a");
    var pa = try Pics.init(gpa, 2200, 11, &.{.{ 300, 400 }}, &.{da});
    defer pa.deinit();
    const k = try p.keepPictures(pa.raw, pa.keyed, pa.media, at, 1 << 32 | 1);
    const key = pointKeyOf(pa.keyed[0..at], pa.media);
    try testing.expectEqual(tier.pointId(key), k.point.?);
    try testing.expect(!std.mem.eql(u8, &key, &pointKeyOf(pa.keyed[0..at], &.{}))); // (the pictures are in the key)
    try p.settle();
    try testing.expect(p.lead0().side.store.presence(key));
    try testing.expect(p.fol.?.side.store.presence(key)); // rank 1 rebuilt rank 0's ids from its own and the note
    try testing.expectEqual(@as(u64, 0), p.fol.?.skipped);
    // each rank's own picture counters, rank 1's by the stats ask
    p.lead0().stats_at = 0;
    p.lead0().pump(1);
    var counts: [4]tier.Counts = undefined;
    try testing.expectEqual(@as(usize, 2), p.lead0().counts(&counts));
    try testing.expectEqual(@as(u64, 1), counts[0].pictures_kept);
    try testing.expectEqual(@as(u64, 1), counts[1].pictures_kept);
    const original = try gpa.dupe(u8, HostEngine.bytesOf(@as(*rig.Lanes.Kept, @ptrCast(@alignCast(k.saved))).snap));
    defer gpa.free(original);
    try p.evict(k.saved);
    // a picture of the same size at the same rows: other ids, no point
    var pb = try Pics.init(gpa, 2200, 11, &.{.{ 300, 400 }}, &.{digestOf("picture b")});
    defer pb.deinit();
    try testing.expectEqualSlices(u32, pa.raw, pb.raw); // (the model's ids are the same placeholders)
    try testing.expectError(error.TestNoPoint, p.restoreMedia(pb.keyed, true, pb.media));
    // the same ids from another payload (the first 8 bytes seed the ids): only the full digest tells them apart
    var forged = pa.media[0];
    forged.digest[20] ^= 1;
    try testing.expectError(error.TestNoPoint, p.restoreMedia(pa.keyed, true, &.{forged}));
    // pictures the key holds but no identity named, or the image cache off: refused before the store
    const refused = p.lead0().side.pictures_refused;
    try testing.expectError(error.TestNoPoint, p.restoreMedia(pa.keyed, true, &.{}));
    try testing.expectEqual(refused + 1, p.lead0().side.pictures_refused);
    // the same pictures: restored on both ranks byte for byte, resuming as a fresh pass of the model ids
    const r = try p.restoreMedia(pa.keyed, true, pa.media);
    try testing.expect(r == .ready);
    const k0: *rig.Lanes.Kept = @ptrCast(@alignCast(r.ready.saved));
    try testing.expect(HostEngine.same(&p.layout, k0.snap.sec, original, HostEngine.bytesOf(k0.snap)));
    const k1: *rig.Lanes.Kept = @ptrCast(@alignCast(p.r[1].savedFor(k0.key) orelse return error.TestNotOnRank1));
    try testing.expect(HostEngine.same(&p.layout, k1.snap.sec, original, HostEngine.bytesOf(k1.snap)));
    const firsts = try p.resumeBoth(pa.raw, r.ready.saved, at);
    try testing.expectEqual(HostEngine.first(pa.raw), firsts[0]);
    try testing.expectEqual(HostEngine.first(pa.raw), firsts[1]);
    // a restart: both stores recover the picture point (its ids checked against its media) and it restores again
    try p.restart();
    try testing.expectEqual(@as(u64, 1), p.lead0().side.store.counts().recovered_points);
    try testing.expectEqual(@as(u64, 1), p.fol.?.side.store.counts().recovered_points);
    try testing.expectError(error.TestNoPoint, p.restoreMedia(pb.keyed, true, pb.media));
    const r2 = try p.restoreMedia(pa.keyed, true, pa.media);
    try testing.expect(r2 == .ready);
    const f2 = try p.resumeBoth(pa.raw, r2.ready.saved, at);
    try testing.expectEqual(HostEngine.first(pa.raw), f2[0]);
    try testing.expectEqual(HostEngine.first(pa.raw), f2[1]);
    try testing.expectEqual(@as(usize, 0), p.pg.fatals);
}

test "pictures: one straddling the cut is recorded whole and another length misses; pictures before, straddling and after the cut; a later turn with one more resumes" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "straddle", rig.Pair.settingsFor(1));
    defer p.deinit();
    const d1 = digestOf("one");
    const d2 = digestOf("two");
    const d3 = digestOf("three");
    // [100, 400) before the cut, [2000, 2200) across it (at 2112), [2250, 2260) past it
    var pp = try Pics.init(gpa, 2400, 21, &.{ .{ 100, 300 }, .{ 2000, 200 }, .{ 2250, 10 } }, &.{ d1, d2, d3 });
    defer pp.deinit();
    const k = try p.keepPictures(pp.raw, pp.keyed, pp.media, at, 1);
    try p.settle();
    // the point holds the two pictures that start before its cut, the second whole (its span runs past the cut)
    const key = pointKeyOf(pp.keyed[0..at], pp.media[0..2]);
    try testing.expectEqual(tier.pointId(key), k.point.?);
    try testing.expect(p.fol.?.side.store.presence(key));
    try p.evict(k.saved);
    // the straddling picture one row shorter (its ids the same through the cut): a miss
    var short = try Pics.init(gpa, 2400, 21, &.{ .{ 100, 300 }, .{ 2000, 199 }, .{ 2250, 10 } }, &.{ d1, d2, d3 });
    defer short.deinit();
    try testing.expectEqualSlices(u32, pp.keyed[0..at], short.keyed[0..at]);
    try testing.expectError(error.TestNoPoint, p.restoreMedia(short.keyed, true, short.media));
    // another picture past the cut does not matter
    var later = try Pics.init(gpa, 2400, 21, &.{ .{ 100, 300 }, .{ 2000, 200 }, .{ 2250, 10 } }, &.{ d1, d2, digestOf("four") });
    defer later.deinit();
    const r = try p.restoreMedia(later.keyed, true, later.media);
    try testing.expect(r == .ready);
    const firsts = try p.resumeBoth(later.raw, r.ready.saved, at);
    try testing.expectEqual(HostEngine.first(later.raw), firsts[0]);
    try testing.expectEqual(HostEngine.first(later.raw), firsts[1]);
    // a later turn: the same prompt, a reply and one more picture after it, finds the point too
    var turn = try Pics.init(gpa, 3000, 21, &.{ .{ 100, 300 }, .{ 2000, 200 }, .{ 2250, 10 }, .{ 2600, 100 } }, &.{ d1, d2, d3, digestOf("five") });
    defer turn.deinit();
    @memcpy(turn.raw[0..2400], pp.raw);
    @memcpy(turn.keyed[0..2400], pp.keyed);
    try testing.expectEqual(@as(?u32, at), if (p.lead0().lookupMedia(turn.keyed, 0, 2999, &.{}, 0, true, turn.media)) |x| x.at else null);
    try testing.expectEqual(@as(usize, 0), p.pg.fatals);
}

test "pictures: a follower whose picture sits elsewhere writes no copy (counted refused), and the point restores on no rank" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "picelse", rig.Pair.settingsFor(1));
    defer p.deinit();
    var pa = try Pics.init(gpa, 2200, 31, &.{.{ 300, 400 }}, &.{digestOf("a")});
    defer pa.deinit();
    // rank 1's prompt holds the picture one row later (its own ids are not rank 0's)
    const moved = try gpa.dupe(u32, pa.raw);
    defer gpa.free(moved);
    moved[300] = 9;
    moved[700] = placeholder;
    var w: Hook = .{ .p = &p, .pa = &pa };
    var s0 = try lanes.Stream.init(gpa, .{ .id = "k", .prompt = pa.raw, .max_new = 4, .reuse = .{ .marks = &.{at}, .hook = .{ .ptr = &w, .at = Hook.at } } });
    defer s0.deinit(gpa);
    var s1 = try lanes.Stream.init(gpa, .{ .id = "k", .prompt = moved, .max_new = 4, .reuse = .{ .marks = &.{at} } });
    defer s1.deinit(gpa);
    try rig.Lanes.prefillFn(&p.r[0], &s0);
    try rig.Lanes.prefillFn(&p.r[1], &s1);
    rig.Lanes.releaseFn(&p.r[0], &s0);
    rig.Lanes.releaseFn(&p.r[1], &s1);
    try p.sink.take(&p.r[1]);
    try testing.expect(w.point != null);
    try p.settle();
    const key = pointKeyOf(pa.keyed[0..at], pa.media);
    try testing.expect(p.lead0().side.store.presence(key));
    try testing.expect(!p.fol.?.side.store.presence(key));
    try testing.expectEqual(@as(u64, 1), p.fol.?.skipped);
    try testing.expectEqual(@as(u64, 1), p.fol.?.side.pictures_refused);
    try p.evict(w.saved.?);
    try testing.expectError(error.SpillNotOnEveryRank, p.restoreMedia(pa.keyed, true, pa.media));
    try testing.expect(!p.lead0().side.store.presence(key)); // (every rank forgot it)
}

const Hook = struct {
    p: *rig.Pair,
    pa: *const Pics,
    saved: ?*anyopaque = null,
    point: ?u64 = null,
    fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        const saved = h.p.hooks.save(h.p.hooks.ptr, null, mark) catch return;
        h.saved = saved;
        h.point = h.p.lead0().keep(.{ .saved = saved, .key = h.pa.keyed[0..mark], .at = mark, .group = 1, .images = true, .media = h.pa.media, .raw = s.prompt() });
    }
};

test "pictures: the image cache off (TF_GLM_IMAGE_CACHE=0) writes no picture point and finds none; a follower with it off skips the note; a text point is unchanged" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    var s = rig.Pair.settingsFor(1);
    s.image_cache = false;
    try p.init(gpa, "picoff", s);
    defer p.deinit();
    var pa = try Pics.init(gpa, 2200, 41, &.{.{ 300, 400 }}, &.{digestOf("a")});
    defer pa.deinit();
    const k = try p.keepPictures(pa.raw, pa.keyed, pa.media, at, 1);
    try testing.expect(k.point == null);
    try testing.expectEqual(@as(u64, 1), p.lead0().side.pictures_refused);
    try testing.expect(p.lead0().lookupMedia(pa.keyed, 0, 2199, &.{}, 0, true, pa.media) == null);
    try testing.expectEqual(@as(u64, 2), p.lead0().side.pictures_refused);
    // a cut before every picture is a text point: written as before, its key without pictures
    const t = try p.keepPictures(pa.raw, pa.keyed, pa.media, 256, 2);
    try testing.expectEqual(tier.pointId(pointKeyOf(pa.raw[0..256], &.{})), t.point.?);
    try p.settle();
    try testing.expect(p.fol.?.side.store.presence(pointKeyOf(pa.raw[0..256], &.{})));
    // rank 0 on, rank 1 off (the fingerprints keep this from pairing; the follower still refuses): a skipped note
    p.lead0().image_cache = true;
    p.fol.?.image_cache = false;
    const k2 = try p.keepPictures(pa.raw, pa.keyed, pa.media, at, 3);
    try testing.expect(k2.point != null);
    try p.settle();
    try testing.expect(!p.fol.?.side.store.presence(pointKeyOf(pa.keyed[0..at], pa.media)));
    try testing.expectEqual(@as(u64, 1), p.fol.?.side.pictures_refused);
}

test "TF_GLM_KV_SHARD: picture points are written under each rank's view of their home part (the lineage salted with it), the same pictures kept from the other part name other blocks, and each restores into its home exactly" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.initWith(gpa, "shard-pictures", rig.Pair.settingsFor(1), .{ .sharded = true });
    defer p.deinit();
    var pa = try Pics.init(gpa, 2200, 71, &.{ .{ 300, 400 }, .{ 2050, 100 } }, &.{ digestOf("a"), digestOf("b") });
    defer pa.deinit();
    var pb = try Pics.init(gpa, 2200, 72, &.{.{ 500, 200 }}, &.{digestOf("c")});
    defer pb.deinit();
    // a kept in part 0 (an empty pool), b in part 1 (part 0 taken by another stream meanwhile)
    const ka = try p.keepPictures(pa.raw, pa.keyed, pa.media, at, 1);
    try p.settle();
    const other = try p.occupy(0, rig.pool_rows / 2);
    const kb = try p.keepPictures(pb.raw, pb.keyed, pb.media, at, 2);
    try p.settle();
    try p.vacate(other);
    const keys = [2]format.Digest{ pointKeyOf(pa.keyed[0..at], pa.media), pointKeyOf(pb.keyed[0..at], pb.media) };
    for ([_]*rig.Leader.S{ p.lead0().side, p.fol.?.side }) |side| {
        try testing.expectEqual(@as(?u32, 0), side.store.partOf(keys[0]).?.part);
        try testing.expectEqual(@as(?u32, 1), side.store.partOf(keys[1]).?.part);
        try testing.expectEqual(@as(u64, 2), side.pictures_kept);
    }
    try testing.expectEqual(@as(u64, 0), p.fol.?.skipped);
    var originals: [2][2][]u8 = undefined; // [point][rank]
    for ([2]*anyopaque{ ka.saved, kb.saved }, 0..) |sv, i| for (0..2) |r| {
        const snap = try p.snapOn(r, sv);
        try testing.expectEqual(@as(?u8, @intCast(i)), snap.home);
        originals[i][r] = try gpa.dupe(u8, HostEngine.bytesOf(snap));
    };
    defer for (originals) |o| for (o) |x| gpa.free(x);
    try p.evict(ka.saved);
    try p.evict(kb.saved);
    for ([_]*const Pics{ &pa, &pb }, 0..) |q, i| {
        const r = try p.restoreMedia(q.keyed, true, q.media);
        try testing.expect(r == .ready);
        for (0..2) |rank| {
            const snap = try p.snapOn(rank, r.ready.saved);
            try testing.expectEqual(@as(?u8, @intCast(i)), snap.home);
            try testing.expect(HostEngine.same(&p.layouts[rank], snap.sec, originals[i][rank], HostEngine.bytesOf(snap)));
        }
        const firsts = try p.resumeBoth(q.raw, r.ready.saved, at);
        try testing.expectEqual(HostEngine.first(q.raw), firsts[0]);
        try testing.expectEqual(HostEngine.first(q.raw), firsts[1]);
    }
    for (&p.e) |*e| try testing.expectEqual(@as(usize, 0), e.carries);
    try testing.expectEqual(@as(usize, 0), p.pg.fatals);
}

test "TF_GLM_KV_SHARD: a restored picture state whose home part has no room resumes carried into the other part, exactly on both ranks" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.initWith(gpa, "shard-piccarry", rig.Pair.settingsFor(1), .{ .sharded = true });
    defer p.deinit();
    var pa = try Pics.init(gpa, 2200, 73, &.{.{ 300, 400 }}, &.{digestOf("a")});
    defer pa.deinit();
    const k = try p.keepPictures(pa.raw, pa.keyed, pa.media, at, 1);
    try p.settle();
    try p.evict(k.saved);
    const full = try p.occupy(0, rig.pool_rows / 2);
    const r = try p.restoreMedia(pa.keyed, true, pa.media);
    try testing.expect(r == .ready);
    try testing.expectEqual(@as(?u8, 0), (try p.snapOn(0, r.ready.saved)).home);
    const firsts = try p.resumeBoth(pa.raw, r.ready.saved, at);
    try testing.expectEqual(HostEngine.first(pa.raw), firsts[0]);
    try testing.expectEqual(HostEngine.first(pa.raw), firsts[1]);
    for (&p.e) |*e| {
        try testing.expectEqual(@as(usize, 1), e.carries);
        try testing.expectEqual(e.resumed, e.exact);
    }
    try p.vacate(full);
    try testing.expectEqual(p.e[0].pool.?.digest(), p.e[1].pool.?.digest());
}
