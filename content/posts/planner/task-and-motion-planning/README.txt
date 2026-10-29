TAMP carrying-stage lab
======================
Python 3.10 or later; standard library only. No pip installation required.

Unzip this directory and run:
  python3 -B tamp_constraints.py --check
  python3 -B tamp_search.py --check

To save the traces:
  python3 -B tamp_search.py --output my-search-results.json

Read assets/search-results.json for the reference output. The blocked scene
needs six refinement calls without proof reuse and three with reuse; both
return the same carry witness. The second scene is unknown after one path
attempt and succeeds when a second, constructed detour is tried.

Scope: a finite outer search around a 2D circular carrying envelope. Blocker
relocation and pick/place are abstract actions, not verified robot motions.
No arm IK, dynamics, perception, hardware or general completeness guarantee.
Distances are metres; contact counts as collision. This is a constructed
example, not a reproduction of the survey's cited experiments.

Article: https://chase6305.github.io/posts/planner/task-and-motion-planning/
