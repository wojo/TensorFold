//! Pictures in the prompt cache (the DeepSeek port's picture keys, recipe 0006's scheme): a prompt with pictures or
//! clips is keyed by their content, not by the placeholder its rows hold, so two pictures never share a kept state and
//! one picture always does. Each picture's identity is its span of the prompt and the full SHA-256 of its canvas
//! (Python's FittedPicture: kind and shape, then the canvas bytes); the per-position cache ids are shortened from it
//! (`pictureId`), and the durable tier records the whole digest beside them (core/spill format v3).
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// A picture of a prompt as the durable tier names it: its positions [start, start + len) and the full SHA-256 of
/// its canvas. (extern: the same bytes as core/spill's format.Media, which a tier checks at comptime.)
pub const Media = extern struct {
    start: u32,
    len: u32,
    digest: [32]u8,

    pub fn eql(a: Media, b: Media) bool {
        return a.start == b.start and a.len == b.len and std.mem.eql(u8, &a.digest, &b.digest);
    }
};

/// Cache ids at or above this are pictures' (`pictureId`); a vocabulary id is always below it.
pub const picture_ids: u32 = 0x8000_0000;

/// The prompt cache's id for position `j` of a picture whose canvas's SHA-256 is `digest`: above every vocabulary id,
/// so it never equals a text token, and the same for the same picture at any place in any prompt (core/spill's
/// format.pictureId, scheme 1).
pub fn pictureId(digest: *const [32]u8, j: u32) u32 {
    const seed = std.mem.readInt(u64, digest[0..8], .little);
    return picture_ids | (@as(u32, @truncate(std.hash.Wyhash.hash(seed, std.mem.asBytes(&j)))) & 0x7fff_ffff);
}

/// TF_GLM_IMAGE_CACHE's one reading (the server's content keys, the disk tier and the rank fingerprint alike): on
/// unless it is exactly "0".
pub fn imageCacheOn(v: ?[]const u8) bool {
    return !std.mem.eql(u8, v orelse return true, "0");
}

/// TF_GLM_IMAGE_MARK_SNAP's reading: on only for exactly "1". On, a kept state's cut never falls inside a picture
/// (`snapMarks`): a resume then never runs the tower on a picture whose earlier rows it took from a kept state, so
/// it stays exact even if the tower's rows for a picture differed between runs. Off (the default): a cut may
/// straddle a picture, as Python's resume does (the tower encodes it whole, the chunks embed the rows past the cut).
pub fn markSnapOn(v: ?[]const u8) bool {
    return std.mem.eql(u8, v orelse return false, "1");
}

/// One picture's or clip's digest: SHA-256 of its kind and shape as Python's FittedPicture hashes them
/// (`image[3, H, W]`, `video[T, 3, H, W]`), a zero byte, then its canvas bytes.
pub fn itemDigest(it: lanes.MediaItem) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [96]u8 = undefined;
    const head = (if (it.video)
        std.fmt.bufPrint(&buf, "video[{d}, 3, {d}, {d}]", .{ it.frames, it.h, it.w })
    else
        std.fmt.bufPrint(&buf, "image[3, {d}, {d}]", .{ it.h, it.w })) catch unreachable;
    h.update(head);
    h.update("\x00");
    h.update(it.data);
    return h.finalResult();
}

/// A prompt's cache key and its pictures' identities (`keyed`: the prompt with each picture's positions made from its
/// digest; `media`: one per item, in prompt order).
pub const Keyed = struct { keyed: []u32, media: []Media };

/// The cache key of `prompt` whose pictures and clips are `m` (its canvases: the engine's own tower), in `a`; null when
/// the media cannot be keyed: features from another tower (no canvases to hash), a follower's copy, or an item whose
/// rows are not one run of consecutive prompt positions (a picture is one span).
pub fn keyOf(a: Allocator, prompt: []const u32, m: *const lanes.Media) Allocator.Error!?Keyed {
    if (m.follower_rows > 0 or m.features.len > 0 or m.items.len == 0) return null;
    var n: usize = 0;
    for (m.items) |it| n += it.rows();
    if (n != m.rows.len) return null;
    var off: usize = 0;
    for (m.items) |it| {
        const rows = m.rows[off..][0..it.rows()];
        if (rows.len == 0 or rows[rows.len - 1] >= prompt.len) return null;
        for (rows, 0..) |r, j| if (r != rows[0] + j) return null;
        off += rows.len;
    }
    const media = try a.alloc(Media, m.items.len);
    errdefer a.free(media);
    off = 0;
    for (m.items, media) |it, *slot| {
        slot.* = .{ .start = m.rows[off], .len = @intCast(it.rows()), .digest = itemDigest(it) };
        off += it.rows();
    }
    const keyed = try a.dupe(u32, prompt);
    for (media) |md| for (0..md.len) |j| {
        keyed[md.start + j] = pictureId(&md.digest, @intCast(j));
    };
    return .{ .keyed = keyed, .media = media };
}

/// The history of a prompt with pictures (`ids`, ending in the template's generation prompt `gen`): the prompt before
/// `gen`, where a later turn resumes; 0 when it does not end so, or the cut would fall inside a picture.
pub fn historyLen(ids: []const u32, gen: []const u32, media: []const Media) usize {
    if (gen.len == 0 or gen.len >= ids.len or !std.mem.endsWith(u32, ids, gen)) return 0;
    const h = ids.len - gen.len;
    for (media) |md| if (md.start < h and h < @as(usize, md.start) + md.len) return 0;
    return h;
}

/// Whether `media` names every picture in `key[0..n]` and nothing else: spans in order, not overlapping, not empty;
/// each span position below `n` holds `pictureId(digest, offset)`, every other position below `n` a vocabulary id.
/// Spans may run past `n` (a picture straddling a cut); spans starting at or past `n` are not looked at. The check
/// that binds the shortened per-position ids to the full digests a point records.
pub fn mediaKeys(key: []const u32, n: usize, media: []const Media) bool {
    if (n > key.len) return false;
    var at: usize = 0; // positions before it are checked
    var prev_end: u64 = 0;
    for (media) |m| {
        if (m.len == 0 or m.start < prev_end) return false;
        prev_end = @as(u64, m.start) + m.len;
        if (m.start >= n) continue;
        for (key[at..m.start]) |id| if (id >= picture_ids) return false;
        const stop: usize = @intCast(@min(prev_end, n));
        for (key[m.start..stop], 0..) |id, j| if (id != pictureId(&m.digest, @intCast(j))) return false;
        at = stop;
    }
    for (key[at..n]) |id| if (id >= picture_ids) return false;
    return true;
}

/// Whether the engine's media `m` places its items exactly at `media`'s spans, one for one (item i's rows are
/// [start, start + len) of media i): what binds the key and the tier's identities to the rows the engine fills.
pub fn spansMatch(m: *const lanes.Media, media: []const Media) bool {
    if (m.follower_rows > 0 or m.items.len != media.len or media.len == 0) return false;
    var off: usize = 0;
    for (m.items, media) |it, md| {
        const n = it.rows();
        if (n != md.len or off + n > m.rows.len) return false;
        for (m.rows[off..][0..n], 0..) |r, j| if (r != md.start + j) return false;
        off += n;
    }
    return off == m.rows.len;
}

/// The marks of a pass over pictures `media` with TF_GLM_IMAGE_MARK_SNAP on (sorted, in place): a mark inside a
/// picture (start < mark < start + len) moves back to the last `grid` point at or below the picture's start, and goes
/// when that is not past `from` (the pass's resume point) or another mark has it already. The marks left, in order.
pub fn snapMarks(marks: []u32, from: u32, media: []const Media, grid: u32) []u32 {
    const g = @max(grid, 1);
    var n: usize = 0;
    for (marks) |mk| {
        var at = mk;
        for (media) |md| if (md.start < at and at < @as(u64, md.start) + md.len) {
            at = md.start / g * g;
            break;
        };
        if (at <= from or (n > 0 and marks[n - 1] >= at)) continue;
        marks[n] = at;
        n += 1;
    }
    return marks[0..n];
}

test imageCacheOn {
    try std.testing.expect(!imageCacheOn("0"));
    try std.testing.expect(imageCacheOn(null) and imageCacheOn("1") and imageCacheOn(""));
    try std.testing.expect(imageCacheOn(" 0") and imageCacheOn("0 ")); // (exact)
    try std.testing.expect(!markSnapOn(null) and !markSnapOn("0") and !markSnapOn("") and markSnapOn("1"));
}

test "itemDigest: kind, shape and every canvas byte" {
    const px: [3 * 28 * 56]u8 = @splat(7);
    const a = itemDigest(.{ .h = 28, .w = 56, .data = &px });
    var other = px;
    other[100] = 8;
    try std.testing.expect(!std.mem.eql(u8, &a, &itemDigest(.{ .h = 28, .w = 56, .data = &other })));
    try std.testing.expect(!std.mem.eql(u8, &a, &itemDigest(.{ .h = 56, .w = 28, .data = &px }))); // same bytes, another shape
    try std.testing.expect(!std.mem.eql(u8, &a, &itemDigest(.{ .video = true, .frames = 2, .h = 28, .w = 28, .data = &px }))); // a clip
    try std.testing.expect(std.mem.eql(u8, &a, &itemDigest(.{ .h = 28, .w = 56, .data = &px })));
    // Python's head and canvas bytes: sha256(b"image[3, 28, 56]" + b"\0" + raw)
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("image[3, 28, 56]\x00");
    h.update(&px);
    try std.testing.expectEqualSlices(u8, &h.finalResult(), &a);
}

test "keyOf: picture positions keyed by content, text unchanged; unkeyable media give none" {
    const gpa = std.testing.allocator;
    const pa: [3 * 28 * 28]u8 = @splat(1);
    const pb: [3 * 28 * 56]u8 = @splat(2);
    const items = [_]lanes.MediaItem{ .{ .h = 28, .w = 28, .data = &pa }, .{ .h = 28, .w = 56, .data = &pb } };
    const prompt = [_]u32{ 5, 9, 99, 6, 99, 99, 7 };
    const rows = [_]u32{ 2, 4, 5 };
    const m: lanes.Media = .{ .rows = &rows, .items = &items };
    const k = (try keyOf(gpa, &prompt, &m)).?;
    defer gpa.free(k.keyed);
    defer gpa.free(k.media);
    try std.testing.expectEqual(@as(usize, 2), k.media.len);
    try std.testing.expect(k.media[0].start == 2 and k.media[0].len == 1 and k.media[1].start == 4 and k.media[1].len == 2);
    try std.testing.expectEqualSlices(u8, &itemDigest(items[1]), &k.media[1].digest);
    try std.testing.expectEqual(pictureId(&k.media[1].digest, 1), k.keyed[5]);
    try std.testing.expect(k.keyed[0] == 5 and k.keyed[3] == 6 and k.keyed[6] == 7);
    try std.testing.expect(mediaKeys(k.keyed, k.keyed.len, k.media));
    try std.testing.expect(spansMatch(&m, k.media));
    // another canvas at the same positions: other ids
    const pc2: [3 * 28 * 56]u8 = @splat(3);
    const items2 = [_]lanes.MediaItem{ items[0], .{ .h = 28, .w = 56, .data = &pc2 } };
    const k2 = (try keyOf(gpa, &prompt, &.{ .rows = &rows, .items = &items2 })).?;
    defer gpa.free(k2.keyed);
    defer gpa.free(k2.media);
    try std.testing.expect(k2.keyed[2] == k.keyed[2] and k2.keyed[4] != k.keyed[4]);
    // features from another tower, a follower's copy, rows split inside one picture: no key
    const feats: [3 * 8]u8 = @splat(0);
    try std.testing.expect((try keyOf(gpa, &prompt, &.{ .rows = &rows, .width = 4, .features = &feats })) == null);
    try std.testing.expect((try keyOf(gpa, &prompt, &.{ .rows = &.{}, .follower_rows = 3 })) == null);
    const split = [_]u32{ 2, 4, 6 };
    try std.testing.expect((try keyOf(gpa, &prompt, &.{ .rows = &split, .items = &items })) == null);
    try std.testing.expect(!spansMatch(&.{ .rows = &split, .items = &items }, k.media));
}

test historyLen {
    const d: [32]u8 = @splat(1);
    const m = [_]Media{.{ .start = 2, .len = 3, .digest = d }};
    try std.testing.expectEqual(@as(usize, 6), historyLen(&.{ 1, 2, 9, 9, 9, 3, 7, 8 }, &.{ 7, 8 }, &m));
    try std.testing.expectEqual(@as(usize, 0), historyLen(&.{ 1, 2, 9, 9, 9, 3, 7, 8 }, &.{ 6, 8 }, &m)); // another suffix
    try std.testing.expectEqual(@as(usize, 0), historyLen(&.{ 1, 2, 9, 9, 7, 8 }, &.{ 7, 8 }, &.{.{ .start = 2, .len = 3, .digest = d }})); // inside a picture
    try std.testing.expectEqual(@as(usize, 0), historyLen(&.{ 7, 8 }, &.{ 7, 8 }, &.{}));
}

test "mediaKeys: every picture id named by a span and made from its digest, nothing else" {
    var a: [32]u8 = @splat(1);
    var b: [32]u8 = @splat(2);
    a[9] = 4;
    b[9] = 5;
    var key = [_]u32{ 1, 2, 0, 0, 0, 3, 0, 0, 0, 0, 4 };
    for (0..3) |j| key[2 + j] = pictureId(&a, @intCast(j));
    for (0..4) |j| key[6 + j] = pictureId(&b, @intCast(j));
    const m = [_]Media{ .{ .start = 2, .len = 3, .digest = a }, .{ .start = 6, .len = 4, .digest = b } };
    try std.testing.expect(mediaKeys(&key, key.len, &m));
    try std.testing.expect(mediaKeys(&key, 8, &m)); // a cut inside the second picture
    try std.testing.expect(mediaKeys(&key, 2, &.{})); // a cut before every picture: text
    try std.testing.expect(!mediaKeys(&key, key.len, m[0..1])); // the second picture unnamed
    try std.testing.expect(!mediaKeys(&key, key.len, &.{})); // none named
    try std.testing.expect(!mediaKeys(&key, key.len, &.{ m[1], m[0] })); // out of order
    try std.testing.expect(!mediaKeys(&key, key.len, &.{ m[0], .{ .start = 4, .len = 2, .digest = b } })); // overlapping
    try std.testing.expect(!mediaKeys(&key, key.len, &.{ .{ .start = 2, .len = 3, .digest = b }, m[1] })); // another canvas
    try std.testing.expect(!mediaKeys(&key, key.len, &.{ .{ .start = 2, .len = 2, .digest = a }, m[1] })); // a short span
    try std.testing.expect(!mediaKeys(&key, key.len + 1, &m)); // past the key
    for (0..1000) |j| try std.testing.expect(pictureId(&a, @intCast(j)) >= picture_ids);
}

test "snapMarks: a cut inside a picture moves back to the grid at its start, or goes" {
    const d: [32]u8 = @splat(3);
    const media = [_]Media{ .{ .start = 100, .len = 300, .digest = d }, .{ .start = 600, .len = 50, .digest = d } };
    var marks = [_]u32{ 64, 192, 256, 448, 640, 704 };
    // 192 and 256 sit inside the first picture: both move to 64, which a mark has already; 640 inside the second: 576
    try std.testing.expectEqualSlices(u32, &.{ 64, 448, 576, 704 }, snapMarks(&marks, 0, &media, 64));
    var again = [_]u32{ 192, 448 };
    try std.testing.expectEqualSlices(u32, &.{448}, snapMarks(&again, 64, &media, 64)); // 64 is not past the resume point
    var none = [_]u32{ 64, 448 };
    try std.testing.expectEqualSlices(u32, &.{ 64, 448 }, snapMarks(&none, 0, &media, 64)); // no mark inside: unchanged
}
