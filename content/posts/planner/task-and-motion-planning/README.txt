TAMP carrying-stage lab
======================
Python 3.10 or later; standard library only. No pip installation required.

Malformed geometry raises ValueError before returning a witness or certificate:
coordinates must be finite, radii/tolerance nonnegative, and opening positive.
Unrepresentable segment differences, squared lengths and point projections also
raise ValueError; finite inputs alone do not guarantee representable arithmetic.
This input error is distinct from unknown for a valid but unresolved problem.

Unzip this directory and run:
  python3 -B tamp_constraints.py --check
  python3 -B tamp_search.py --check --summary
  python3 -B tamp_kinematics.py --check --summary

To save the traces:
  python3 -B tamp_search.py --output my-search-results.json

Use --summary for a short console table and three-step detour trace. Omitting
it prints the full JSON; --output always saves the full JSON even when the
console summary is selected.

Read assets/search-results.json for the reference output. The blocked scene
needs six refinement calls without proof reuse and three with reuse; both
return the same carry witness. The second scene is unknown after one path
attempt and succeeds when a finite-grid A* search generates a detour.

The A* grid uses 0.2 m spacing, x in [-2, 2] m and y in [-1.2, 1.2] m,
eight neighbors and Euclidean edge lengths/heuristic. Every entire edge is
collision-checked; a final check covers the reconstructed polyline. Exhausting
this finite grid or its expansion budget returns unknown, not a proof about
all continuous paths. Blocker relocation is still an abstract primitive.

To compare grid spacings while keeping the physical domain and geometry fixed:
  python3 -B tamp_search.py --resolutions --summary
  python3 -B tamp_search.py --resolutions --output my-resolution-results.json
Reference: assets/grid-resolution-results.json. Spacings 2, 1, 0.4, 0.2 and
0.1 m use 3, 15, 77, 273 and 1025 candidate grid centres (including colliding
centres); grid_node_count reports that total before collision filtering.
Only the 2 m grid is exhausted;
this is unknown even though continuous detours exist in the same scene.
grid_motion(..., step=...) accepts a positive finite spacing that places
the fixed endpoints on grid nodes. Nonaligned or numerically unrepresentable
spacings raise ValueError. The default step remains 0.2 m. Path lengths are
finite-graph optima under the specified edges, not continuous optima; expanded
node counts describe this fixed search order, not a runtime benchmark.

Separate endpoint-kinematics example:
  python3 -B tamp_kinematics.py --output my-grasp-results.json
Reference: assets/grasp-results.json. The planar 3R arm has link lengths
1, 1, 0.25 m and unrestricted revolute joints. The left endpoint supports
only back, the right only front; middle supports both. Releasing at middle
allows two different shared-grasp stages, but no connecting motions or
support/contact stability are checked. The circle model and planar arm are
independent examples and do not combine into one verified robot plan.
SE(2) poses use metres and radians; g = hand_T_object, so p = FK(q) g.
Singular wrist configurations can have continuous IK families; only
representatives, not a full enumeration of those families, are returned.

The short scheduler printed in the article checks its call budget and rejects
unrecognized result states. The full search also rechecks analytical failure
certificates before caching them, and checks the fixed start/goal as well as
every segment before returning a carry witness. A clear segment that remains
on the left side of the wall does not satisfy the requested transfer.
A new search() call starts over; there is
no cross-call checkpoint/resume API in this example.

Scope: a finite outer search around a 2D circular carrying envelope. Blocker
relocation and pick/place are abstract actions, not verified robot motions.
Disks are the modeled collision shapes. For real shapes enclosed in such a
disk, disk clearance is a conservative collision condition; disk failure does
not prove real-shape infeasibility. A long thin rectangle can fit a doorway
that its enclosing disk cannot. Robot kinematic realizability remains separate.
No arm IK, dynamics, perception, hardware or general completeness guarantee.
Distances are metres; contact counts as collision. This is a constructed
example, not a reproduction of the survey's cited experiments.

Article: https://chase6305.github.io/posts/planner/task-and-motion-planning/
