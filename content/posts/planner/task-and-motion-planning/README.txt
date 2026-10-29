TAMP carrying-stage lab
======================
Python 3.10 or later; standard library only. No pip installation required.

Unzip this directory and run:
  python3 -B tamp_constraints.py --check
  python3 -B tamp_search.py --check --summary

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

Scope: a finite outer search around a 2D circular carrying envelope. Blocker
relocation and pick/place are abstract actions, not verified robot motions.
No arm IK, dynamics, perception, hardware or general completeness guarantee.
Distances are metres; contact counts as collision. This is a constructed
example, not a reproduction of the survey's cited experiments.

Article: https://chase6305.github.io/posts/planner/task-and-motion-planning/
