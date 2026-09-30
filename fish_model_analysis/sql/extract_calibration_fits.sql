-- Every stored laser calibration, with the object it was fitted from.
--
-- §4.3 claims the dive slate carries the same scale as the checkerboard, and
-- checks it on the laser baseline: the baseline is a property of the rig, so a
-- unit calibrated against both objects must report the same one either way.
-- That check needs the calibration OBJECT per fit, which no other export
-- carries -- `dive.calibration_target_id` is the only place it is recorded.
--
-- A refused calibration has no `laserextrinsics` row at all (the gate refuses
-- to persist rather than flagging), so this returns accepted fits only. That
-- is the intended cohort: the comparison is between calibrations the pipeline
-- would actually use.
--
-- The baseline is hypot(x, y) of `laser_position`; z is 0 by construction.
select
    le.dive_id,
    le.camera_id,
    d.name                as dive_name,
    case when d.calibration_target_id is not null
         then 'checkerboard' else 'slate' end as standard,
    -- The date is not decoration. In this corpus every checkerboard fit is
    -- 14-18 August 2023 and every slate fit 29-31 August, with the cameras
    -- shipped in between, so the calibration OBJECT is perfectly confounded
    -- with the epoch and nothing here can separate the two. Any analysis that
    -- compares the objects has to carry this column and say so.
    to_char(d.dive_datetime, 'YYYY-MM-DD') as dive_date,
    sqrt(power((le.laser_position->>0)::double precision, 2)
       + power((le.laser_position->>1)::double precision, 2)) as baseline_m,
    -- Everything below is appended rather than interleaved, so a reader of the
    -- earlier export finds the same columns in the same places.
    --
    -- The full timestamp orders calibrations taken minutes apart (dives 489
    -- and 490 are eight minutes apart and differ by 0.84 deg), which the date
    -- alone cannot.
    to_char(d.dive_datetime at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as dive_datetime,
    -- The fitted beam, split into scalar columns: the JSON arrays contain
    -- commas and this file is comma-separated.
    (le.laser_position->>0)::double precision as laser_x,
    (le.laser_position->>1)::double precision as laser_y,
    (le.laser_position->>2)::double precision as laser_z,
    (le.laser_axis->>0)::double precision     as axis_x,
    (le.laser_axis->>1)::double precision     as axis_y,
    (le.laser_axis->>2)::double precision     as axis_z,
    -- How many frames the fit rests on: frames carrying both a completed slate
    -- label and a laser label for a slate calibration, frames carrying a laser
    -- label for a checkerboard one (the board itself is found automatically).
    -- Two or three frames fix a line barely at all, so a reader must be able
    -- to see which fits are thin before reading anything into them.
    case when d.calibration_target_id is not null then (
        select count(distinct l.image_id) from laserlabel l join image i on i.id = l.image_id
        where i.dive_id = d.id and not coalesce(l.superseded, false))
    else (
        select count(distinct s.image_id) from diveslatelabel s
        join image i on i.id = s.image_id
        join laserlabel l on l.image_id = s.image_id and not coalesce(l.superseded, false)
        where i.dive_id = d.id and s.completed and not coalesce(s.superseded, false))
    end as n_frames
from laserextrinsics le
join dive d on d.id = le.dive_id
order by le.camera_id, standard, le.dive_id;
