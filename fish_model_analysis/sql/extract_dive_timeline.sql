-- Per-image timeline of every dive that owns a calibration or holds a
-- measurement: when its calibration frames (board or slate) were shot relative
-- to its measured frames. Backs the field-runnable checks on a dive's own
-- calibration -- bracketing and time gap -- which need no known length.
--
-- One row per image carrying a live laser label (`has_dot` false where the
-- labeler marked no dot; `dot_x`/`dot_y` then empty). `content` is the image's species label(s):
-- 'Calibration Targets, E4E Checkerboard' marks a board frame. `measured` is
-- true when the image has a Measurement; `slate` when it carries a completed,
-- live dive-slate label.
--
-- Read-only. Production stores no per-frame board pose -- the checkerboard is
-- re-detected from the image at fit time -- and one fitted beam per dive, so
-- neither the calibration's per-frame range spread nor a per-burst beam is
-- recoverable from here.
\pset format unaligned
\pset fieldsep '|'
\pset footer off
SELECT i.dive_id,
       i.id AS image_id,
       to_char(i.taken_datetime AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS taken_utc,
       bool_or(l.x IS NOT NULL) AS has_dot,
       -- the dot as the corpus takes it: the lowest-id live label with x/y,
       -- in the stored (undistorted pinhole) pixel frame
       (array_agg(l.x ORDER BY l.id) FILTER (WHERE l.x IS NOT NULL))[1] AS dot_x,
       (array_agg(l.y ORDER BY l.id) FILTER (WHERE l.x IS NOT NULL))[1] AS dot_y,
       EXISTS (SELECT 1 FROM measurement m WHERE m.image_id = i.id) AS measured,
       EXISTS (SELECT 1 FROM diveslatelabel s WHERE s.image_id = i.id AND s.completed
               AND NOT coalesce(s.superseded, false)) AS slate,
       coalesce((SELECT string_agg(DISTINCT sl.content_of_image, ';') FROM specieslabel sl
                 WHERE sl.image_id = i.id), '') AS content
FROM image i
JOIN laserlabel l ON l.image_id = i.id AND NOT coalesce(l.superseded, false)
WHERE i.dive_id IN (SELECT dive_id FROM laserextrinsics
                    UNION SELECT i2.dive_id FROM measurement m2 JOIN image i2 ON i2.id = m2.image_id)
GROUP BY i.id
ORDER BY i.dive_id, i.taken_datetime, i.id;
