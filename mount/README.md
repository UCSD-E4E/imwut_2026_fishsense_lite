# Laser mount

The 3D-printed mount that holds the laser pointer on the camera housing, as
used for the deployments this paper reports. Printed in PLA (§3 of the paper).
3MF, units in millimetres; each file is a single part.

## The stack

From the housing up:

**cold shoe → taper → mount → cap → pointer**

| file | part | what it does |
|---|---|---|
| `taper.3MF` | the taper, a wedge | Seats into the housing's cold-shoe channel and into a recess in the mount body at the same time. A screw passes down through the mount into a nut captured in the taper; tightening it drives the wedge into both and forces the mount flat against the shoe. It is there to take up the slack a cold shoe has natively, not to space or tilt the laser. |
| `mount.3MF` | the mount body | Carries the recess the taper wedges into, and the half-round cradle the pointer lies in (12.75 mm radius, so a pointer about 25.5 mm across), with two screw ears for the cap. |
| `cap.3MF` | the cap | Closes over the pointer onto the cradle (same 12.75 mm radius), held by two screws through the cap and mount ears, nutted underneath. Cap and cradle together are the clamp. |

Three printed parts, two screwed joints and a clamp:

- **mount to taper**: one screw into the taper's captured nut, which also preloads the wedge in the shoe;
- **cap to mount**: two screws with nuts, which close the clamp on the pointer.

## Where the laser can move

Section 4.3 measures how far the laser moves between calibrations: a median
0.53° vertically and 0.54° horizontally per step, with near-pure cases in both
directions. The stack was built to be rigid, so that motion is what remains
after the wedge preload, and it can arise at each interface and in the parts
themselves:

- the **taper in the cold shoe**, despite the preload;
- the **mount on the taper**, at the screwed and nutted joint;
- the **pointer in the clamp**, which is why each pointer carries an alignment
  mark against the mount;
- **flex of the printed PLA**, which the author observed in use and which is
  not an interface at all.

The calibrations cannot apportion the motion among these: with three to six
fits per unit, and the fitted laser origin and angle trading off inside each
fit, the attribution is underdetermined.
