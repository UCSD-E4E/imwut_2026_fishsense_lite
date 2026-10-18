-- Every image of every dive that owns a calibration or holds a measurement,
-- with what is recorded about the laser in it. Backs the search for laser
-- on/off switches within a session.
--
-- Laser state is not logged. What exists per image:
--   label_state  'dot'   a live human laser label with a position;
--                'empty' a live human label submitted with no dot -- the
--                        labeler saw none: laser off, OR the dot out of frame,
--                        washed out or hidden. Not a switch record by itself.
--                'none'  never labeled (or only superseded labels).
--   pred_conf    the automated detector's confidence for its best candidate,
--                where the detector ran (NULL where it did not).
--   pred_gate    the detector's gate verdict ('no_prediction' = nothing found).
-- Dot pixels are the corpus convention: lowest-id live label with x/y, in the
-- stored (undistorted pinhole) frame.
--
-- Read-only.
\pset format unaligned
\pset fieldsep '|'
\pset footer off
SELECT i.dive_id,
       i.id AS image_id,
       to_char(i.taken_datetime AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS taken_utc,
       i.is_canonical,
       CASE WHEN bool_or(l.x IS NOT NULL) THEN 'dot'
            WHEN count(l.id) > 0 THEN 'empty' ELSE 'none' END AS label_state,
       (array_agg(l.x ORDER BY l.id) FILTER (WHERE l.x IS NOT NULL))[1] AS dot_x,
       (array_agg(l.y ORDER BY l.id) FILTER (WHERE l.x IS NOT NULL))[1] AS dot_y,
       (SELECT max(p.confidence) FROM laserprediction p WHERE p.image_id = i.id) AS pred_conf,
       (SELECT string_agg(DISTINCT p.gate_verdict, ';') FROM laserprediction p WHERE p.image_id = i.id) AS pred_gate,
       EXISTS (SELECT 1 FROM measurement m WHERE m.image_id = i.id) AS measured
FROM image i
LEFT JOIN laserlabel l ON l.image_id = i.id AND NOT coalesce(l.superseded, false)
WHERE i.dive_id IN (SELECT dive_id FROM laserextrinsics
                    UNION SELECT i2.dive_id FROM measurement m2 JOIN image i2 ON i2.id = m2.image_id)
GROUP BY i.id
ORDER BY i.dive_id, i.taken_datetime, i.id;
