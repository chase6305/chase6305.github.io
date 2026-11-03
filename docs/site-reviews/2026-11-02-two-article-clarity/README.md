# Two-article clarity review, 2026-11-02

Status: complete. The requested three-hour window started at
`2026-11-02T06:21:09Z` and finished at `2026-11-02T09:22:15.777298+00:00`
(10866.777 elapsed seconds). The elapsed-time and source guards
pass in `final-acceptance.json`.
This window does not commit, push or deploy.

## Editorial result

SuperMimic now distinguishes reference motion from control actions, explains why
a learned differentiable surrogate is useful alongside a known simulator, and
separates the 50 Hz control cycle from the eight-step model window and 32-step
actor window. An early parameter table explicitly divides the third phase into
actor improvement and critic fitting. The explanation defines Jacobians,
per-step discounting, the terminal-value contribution and lambda-return endpoints.
The optional published PyTorch example demonstrates frozen model weights with a
nonzero actor gradient. The existing scalar and interactive examples remain intact.

The post-training article follows one constructed code request through human
comparison, function tests and contrastive generation. It distinguishes fitting a
reward model from updating a policy, explains reward model/value critic/reference/
sampling-policy roles, and connects verifier scores to detached advantages and
log-probability gradients. SFT is explicitly part of post-training. RLCD's original
automatic branch labels and preference-model/PPO path are separated from a DPO
extension. Its 30B data generator is distinguished from the final 7B policy, and
independent human evaluation is distinguished from manually relabeling training
pairs. The reversed audit answer is corrected.

The existing SVG now splits preference pairs from verifier scores and shows the
two preference-training alternatives. All 28 labels and the arrows were inspected;
its dimensions are recorded; its public path and vector format are preserved.
`visuals/feedback-routes-before.png` and `feedback-routes-after.png` show the change.

## Source and numerical evidence

`primary-pdf-sources.json` hashes four primary PDFs. `source-clarity-ledger.json`
maps 14 teaching/factual claims to the supplied SuperMimic paper, InstructGPT,
RLCD, DeepSeek-R1, DPO and the PyTorch autograd notes. Full PDFs are not copied
into the repository.

`published-numbers.json` independently recomputes the current preference table
with 90-digit Decimal arithmetic, the terminal-value table with exact fractions,
the explicitly population-normalized four-answer advantages, and the two
constructed function-interface tests. It executes the published standard-library
finite-difference snippet and binds the current Torch snippet to its executed hash.
`torch-freeze.json` records four CPU autograd configurations under PyTorch
2.8.0+cpu: frozen weights, eval without freezing, detached state and no-grad.
There are zero optimizer steps and no robot, GPU or model-training reproduction.

`existing-scalar-checks.json` records the first-phase checks of the unchanged
survival helper (500 random trajectories) and interactive lab (100 closed-form
cases). The standalone logs were not captured; the passing tool output was
recorded in the session. Current source hashes bind these unchanged implementations.

`formula-layout-equivalence.json` compares mathematical tokens for eight
SuperMimic equations and the RLCD generation equation, ignoring layout, delimiter
sizing and explicit multiplication dots. The conditional-KL rewrite is a separate
clarification of the online objective.

## Reading and print

The final source is recorded in `source-manifest.json`; builds live outside the
repository under `/tmp/chase-two-article-clarity-20261102-n43qnb22`.
The **coherent** production and draft-inclusive previews use loopback ports
13651 and 13653. Browser checks run serially on private headless Chrome CDP 9242.

Evidence covers 48 reading layouts and 40 native chapter loads; 16 real table-arrow
cases and four exact asynchronous copies; 32 image-viewer cases; 48 search queries
across both language entrances; and 16 independent axe contexts. Closed-popup
`aria-controls` checks remain incomplete in axe, but their ID references resolve.
The formula-only loss header found by axe now includes readable text.

The required overflow checker originally assumed at least one formula would
overflow at desktop 200% text. Reflow removes that precondition for SuperMimic.
The repository checker now verifies the complete fitting state in that case.
The article-focused checker separately reports 12 actual arrow-scroll contexts
and six fitting-only font contexts; these counts are not combined as scroll tests.
Copy-check setup corrections exclude the closing-fence newline and select the
fenced block rather than an earlier inline code span. Viewer screenshots wait
for the opening transition. These setup errors are not reported as application bugs.

Normal draft-inclusive print checks cover 576 layouts; targeted 200% print checks
cover both revised articles in three themes, with and without page scripts.
Eight tagged native exports contain 33/88/33/93/12/32/12/34 pages, totaling **337**.
These are export counts, not a claim of manual review of every page. All 36 checks
of nine short explanatory units pass, including complete comparison tables and
the Torch code with its output. Six current PDF rasters have direct visual
acceptance records in `current-visual-review.json`; other selected pages and the
SVG were inspected during earlier iterations.

An extra whole-site 200% print run before reflow found 264 formula-bound failures
across 44 articles. The two requested articles are repaired; the other 42 articles
are unchanged. This review does not claim a passing whole-site 200% print run.
The normal whole-site print check is separate and passes.

## Preservation and reproduction

Original titles, dates, draft states, URLs, tags, neighboring links, all 62/21
prior page IDs and the SuperMimic media/download implementations are preserved.
Only acceptance records 83 and 91 change. There are no shared page-script,
stylesheet, template, workflow or theme-submodule changes. The overflow-checker
change is test-only. Generated output and article bytecode are not committed.

Run the review scripts from the repository root. Pass the fresh preview URL,
private CDP port and output directory to the browser scripts, serially. The
semantic checker also takes the external pinned axe file as its fourth argument.
`check_pdf_units.py` takes the native-export directory and `--output` path.
`check_published_numbers.py` takes `--torch-receipt` and `--output`; the numerical
checks use the standard library. `check_torch_freeze.py` additionally requires a
separate CPU PyTorch environment. These helpers are review evidence, not site
runtime dependencies.
