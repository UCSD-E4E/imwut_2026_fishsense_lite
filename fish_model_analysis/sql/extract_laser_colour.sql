-- Laser colour of every measured frame, from the labeller's own tag on the dot.
--
-- One row per measurement. The colour is read off the laser label the pipeline
-- measured from -- the lowest-id, non-superseded label with coordinates, which
-- is the one `extract_corpus.sql` joins -- so it is the colour of the dot that
-- produced the depth, not a property assumed for the dive, the camera or the
-- date. A folder name is not evidence; the label is the closest thing to an
-- observation of the dot itself, though it is a labeller's call and can be
-- wrong.
--
-- `colour` is empty where that label carries no colour tag (1,900 current
-- labels in production have none); report those as unknown, never assign them.
-- `n_colours` counts the distinct colours among ALL current labels on the
-- frame, so a frame tagged both ways can be found.
SELECT DISTINCT ON (m.id)
       m.id                                   AS measurement_id,
       i.id                                   AS image_id,
       i.dive_id,
       d.camera_id,
       to_char(i.taken_datetime AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS taken_datetime,
       split_part(sl.content_of_image, ', ', 2) AS subject,
       coalesce(ll.label, '')                 AS colour,
       (SELECT count(DISTINCT l2.label) FROM laserlabel l2
         WHERE l2.image_id = i.id AND NOT coalesce(l2.superseded, false)
           AND coalesce(l2.label, '') <> '')  AS n_colours
FROM measurement m
JOIN image i       ON i.id = m.image_id
JOIN dive d        ON d.id = i.dive_id
LEFT JOIN specieslabel sl ON sl.image_id = i.id
LEFT JOIN laserlabel ll   ON ll.image_id = i.id AND ll.superseded = false
                         AND ll.x IS NOT NULL AND ll.y IS NOT NULL
-- Prefer a species label that says what the frame shows: some frames carry an
-- earlier, empty one, and taking it would blank the subject and let held-out
-- targets past a cohort filter.
ORDER BY m.id, (coalesce(sl.content_of_image, '') = ''), sl.id, ll.id;
