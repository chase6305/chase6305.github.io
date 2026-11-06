# Machine Learning for Robotics source review

Input: `/home/ubuntu/Downloads/TSP_CMC_81804.pdf`, 33 pages, SHA-256 `bf8b64eced7d559d5dd941525b40084b27ff17a6b584ee6964a5e6e3ed027212`.

Authors: Ahmed Ismail Ebada, Yasmeen Abu-Seif, Hrushikesh Pardeshi, Nesma El-Sayed. The PDF header gives online publication 3 July 2026. The [publisher record](https://www.techscience.com/cmc/v88n3/68093) gives volume 88(3) issue publication 23 July 2026. DOI: `10.32604/cmc.2026.081804`. These dates are not conflated.

The later three-hour revision is recorded in [the follow-up review](../2026-11-05-three-paper-threehour/README.md). This directory retains the initial evidence; use the follow-up report for current article hashes, added calculators and pinned-source statistical checks.

## Source and reasoning map

| Topic | Review location | Article treatment |
| --- | --- | --- |
| Learning taxonomy | Sections 2–4 | Organize learned perception, dynamics, planning, skills and control by function, then distinguish supervision/interaction sources. |
| PPO stability statements | Sections 2.3, 3.3, 7.4, 8.3 | A clipped surrogate does not impose a global hard trust region or prove collision avoidance. |
| Lyapunov claims | Sections 2.3, 3, 6.5 | Give a semidefinite counterexample and distinguish convergence from constraint invariance. |
| Sampled CBF | Section 6.5 and 9.2 motivation | Independent scalar counterexample; check the continuous and held-input trajectories separately. |
| PSL | Table 2 and sections 4.3, 8.4 | The [original v1](https://arxiv.org/html/2405.01534v1), section 3.5/D.3.2, uses DrQ-v2. Original table 2 lists NutAssembly as four stages at 0.96±0.08; K-MS-10 is a separate ten-stage task. |
| OpenVLA-OFT+ timing | Review section 6.3 | The [original v2](https://arxiv.org/html/2502.19645v2), table III, has ALOHA K=25, latency 0.321 seconds, throughput 77.9 actions/s, 25 Hz controller. Do not combine that row with 31.1 ms. |
| PPI and SureSim | Sections 7.2, 9.1 | Derive the pooled estimator and account for paired-sample covariance; population confidence is distinct from per-step feasibility. |
| SureSim percentages | Section 7.2 | [Original v1](https://arxiv.org/html/2510.04354v1): N=700 interval 0.187 to 0.16, 14.4% reduction; evaluation-effort savings depend on experiment and simulator predictivity. |
| NeME and subgoal evaluation | Sections 5.3, 7.3 | Evaluator classification scores, checkpoint choice and robot task success are separate metrics. No broad sequence-model ranking is inferred. |
| Conceptual frameworks | Section 8.7/table 10 and conclusion | CSE/SKS/LBSE/PBVD/BCKG organize research responsibilities. No unified implementation or theorem is invented. |
| Healthcare examples | Sections 5, 8.5 | Generic robotic benchmarks and proposed application contexts are not clinical-effectiveness evidence. No medical or regulatory advice is supplied. |

The control discussion also links [CLF/CBF-QP](https://arxiv.org/abs/1609.06408); the PPO discussion links its [original paper](https://arxiv.org/abs/1707.06347). Mathematical examples, figures and statistical derivations are newly constructed and labeled accordingly. No benchmark is claimed to be reproduced.

## Check the examples

```bash
python3 -B content/posts/ai/machine-learning-for-robotics/deployment_sanity.py --check
python3 -B scripts/render_robotics_ml_review.py --check
```

The PPI check enumerates all 256 IID draws for each of four small known joint populations, checking the mean and variance against independent moment calculations. The zero-failure bound uses Decimal as an independent numerical oracle. Scalar control checks cover feasible/infeasible actuator intervals, a semidefinite Lyapunov system, finite-difference dynamics and the sampled-control violation.

The examples do not implement SureSim's WSR intervals or any learning algorithm. Native print and reading drivers are shared in [the LAST/OmniCalib review directory](../2026-11-05-last-omnicalib/README.md). Figure prompts are kept in `docs/image-generation/robotics-ml-review-2026-11-05.json`, outside published content.
