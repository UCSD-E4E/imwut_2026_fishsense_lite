-- Repeat positions of the same landmark on the same frame, for the labeling-noise
-- terms of section 5.7. Read-only.
--
-- kind = 'dot'      every human laser label with a position (live or superseded) on
--                   frames that carry two or more, plus the automated detector's
--                   position (source 'detector') on every frame with a live human dot.
-- kind = 'headtail' every human snout/fork label on frames that carry two or more,
--                   plus the automated head/tail predictor's (source 'predictor') on
--                   frames with a live human head/tail label.
-- line_a/b/c: the dive's fitted image line a x + b y + c = 0 (latest
-- divelaserline row), so a dot difference splits into along-line (range) and
-- across-line parts.
\pset format unaligned
\pset fieldsep '|'
\pset footer off
WITH line AS (
  SELECT DISTINCT ON (dive_id) dive_id, a, b, c FROM divelaserline ORDER BY dive_id, fitted_at DESC
), dot_rep AS (
  SELECT image_id FROM laserlabel WHERE x IS NOT NULL GROUP BY image_id HAVING count(*) > 1
), ht_rep AS (
  SELECT image_id FROM headtaillabel WHERE head_x IS NOT NULL AND tail_x IS NOT NULL GROUP BY image_id HAVING count(*) > 1
)
SELECT 'dot' AS kind, 'human' AS source, l.id AS label_id, l.user_id, coalesce(l.superseded, false) AS superseded,
       i.id AS image_id, i.dive_id, l.x, l.y, NULL::float AS x2, NULL::float AS y2, ln.a AS line_a, ln.b AS line_b, ln.c AS line_c
FROM laserlabel l JOIN image i ON i.id = l.image_id LEFT JOIN line ln ON ln.dive_id = i.dive_id
WHERE l.x IS NOT NULL AND l.image_id IN (SELECT image_id FROM dot_rep)
UNION ALL
SELECT 'dot', 'detector', p.id, NULL, false, i.id, i.dive_id, p.x, p.y, NULL, NULL, ln.a, ln.b, ln.c
FROM laserprediction p JOIN image i ON i.id = p.image_id LEFT JOIN line ln ON ln.dive_id = i.dive_id
WHERE p.x IS NOT NULL AND EXISTS (SELECT 1 FROM laserlabel l WHERE l.image_id = p.image_id
                                  AND NOT coalesce(l.superseded, false) AND l.x IS NOT NULL)
UNION ALL
SELECT 'dot', 'human_live', l.id, l.user_id, false, i.id, i.dive_id, l.x, l.y, NULL, NULL, ln.a, ln.b, ln.c
FROM laserlabel l JOIN image i ON i.id = l.image_id LEFT JOIN line ln ON ln.dive_id = i.dive_id
WHERE l.x IS NOT NULL AND NOT coalesce(l.superseded, false)
  AND EXISTS (SELECT 1 FROM laserprediction p WHERE p.image_id = l.image_id AND p.x IS NOT NULL)
UNION ALL
SELECT 'headtail', 'human', h.id, h.user_id, coalesce(h.superseded, false), i.id, i.dive_id,
       h.head_x, h.head_y, h.tail_x, h.tail_y, NULL, NULL, NULL
FROM headtaillabel h JOIN image i ON i.id = h.image_id
WHERE h.head_x IS NOT NULL AND h.tail_x IS NOT NULL AND h.image_id IN (SELECT image_id FROM ht_rep)
UNION ALL
SELECT 'headtail', 'predictor', q.id, NULL, false, i.id, i.dive_id, q.head_x, q.head_y, q.tail_x, q.tail_y, NULL, NULL, NULL
FROM headtailprediction q JOIN image i ON i.id = q.image_id
WHERE q.head_x IS NOT NULL AND q.tail_x IS NOT NULL
  AND EXISTS (SELECT 1 FROM headtaillabel h WHERE h.image_id = q.image_id AND NOT coalesce(h.superseded, false) AND h.head_x IS NOT NULL)
UNION ALL
SELECT 'headtail', 'human_live', h.id, h.user_id, false, i.id, i.dive_id, h.head_x, h.head_y, h.tail_x, h.tail_y, NULL, NULL, NULL
FROM headtaillabel h JOIN image i ON i.id = h.image_id
WHERE h.head_x IS NOT NULL AND h.tail_x IS NOT NULL AND NOT coalesce(h.superseded, false)
  AND EXISTS (SELECT 1 FROM headtailprediction q WHERE q.image_id = h.image_id AND q.head_x IS NOT NULL)
ORDER BY kind, image_id, source, label_id;
