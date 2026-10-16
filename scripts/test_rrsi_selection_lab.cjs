#!/usr/bin/env node
// Exercise selection boundaries and novelty history, independent of browser UI.
const assert = require('node:assert/strict');
const {configs, judge, selectRound} = require('../static/labs/rrsi/selection_lab.js');
const base = {score: .81, best: .82, delta: .017, cost: 1000000,
  candidate: .835, candidateCost: 1800000, touched: [], accepted: [], guardOK: true};
const cases = [
  [.835, 1800000, [], true, 'admissible'],
  [.819, 1260000, [], true, 'cost'],
  [.810, 950000, [], true, 'admissible'],
  [.800, 700000, [], true, 'floor'],
  [.809, 1000000, ['skill'], true, 'admissible'],
  [.850, 2000000, [], false, 'guard']
];
for (const [candidate, candidateCost, touched, guardOK, reason] of cases) {
  assert.equal(judge({...base, candidate, candidateCost, touched, guardOK}, configs.coding).reason, reason);
}
// Exactly on the noise-band edge uses the shaped branch, with its strict > 0.
const binary = {score: .5, best: .5, delta: .125, cost: 8,
  candidate: .625, candidateCost: 8, touched: [], accepted: [], guardOK: true};
const cfg = {beta0: .25, beta1: 1, ws: 0, wc: 1, wn: .5};
let result = judge(binary, cfg);
assert.equal(result.strong, false);
assert.equal(result.shaped, 0);
assert.equal(result.reason, 'cost');
// At the strong-gain cost allowance equality, <= admits the candidate.
result = judge({...binary, candidate: .75, candidateCost: 12}, cfg);
assert.equal(result.strong, true);
assert.equal(result.dc, result.allowance);
assert.equal(result.admissible, true);
// Floor equality is allowed; savings can admit a lower-scoring candidate.
result = judge({...binary, candidate: .375, candidateCost: 7}, cfg);
assert.equal(result.floorOK, true);
assert.equal(result.admissible, true);
assert.equal(judge({...binary, candidate: .374, candidateCost: 1}, cfg).reason, 'floor');
// Only distinct, never-winning structural types receive novelty credit.
const novel = {...binary, candidate: .5, touched: ['skill', 'skill', 'prompt']};
assert.equal(judge(novel, cfg).novelty, 1);
assert.equal(judge({...novel, accepted: ['skill']}, cfg).novelty, 0);
assert.equal(judge({...novel, accepted: ['skill']}, cfg).reason, 'cost');
assert.equal(judge({...novel, touched: ['skill', 'memory', 'subagent', 'client_tool']}, cfg).novelty, 4);
// A domain guard cannot be purchased with higher score or a token saving.
assert.equal(judge({...binary, candidate: .99, candidateCost: 1, guardOK: false}, cfg).reason, 'guard');
// Workspace and engineering can reward a small gain inside their own bands.
for (const domain of ['workspace', 'engineering']) {
  const c = configs[domain];
  assert.equal(judge({...base, best: .81, delta: c.delta, candidate: .812, candidateCost: 1000000}, c).admissible, true);
}
// The two configured branches need not join continuously at the noise edge.
const engineeringEdge = {...base, delta: .020, candidate: .830, candidateCost: 2000000};
assert(judge(engineeringEdge, configs.engineering).admissible);
assert.equal(judge({...engineeringEdge, candidate: .831}, configs.engineering).reason, 'cost');
assert(judge({...base, candidate: .81, candidateCost: 1030000, touched: ['skill']}, configs.coding).admissible);
assert.equal(judge({...base, candidate: .81, candidateCost: 1040000, touched: ['skill']}, configs.coding).reason, 'cost');
for (const patch of [{cost: 0}, {candidateCost: NaN}, {candidate: 1.1}, {best: .80}, {delta: -1}]) {
  assert.throws(() => judge({...base, ...patch}, configs.coding));
}
const A = {score: .835, cost: 1800000, touched: [], guardOK: true};
const C = {score: .810, cost: 950000, touched: [], guardOK: true};
const E = {score: .809, cost: 1000000, touched: ['skill'], guardOK: true};
let round = selectRound(base, [A, C, E], configs.coding);
assert.equal(round.winnerIndex, 0);
assert.equal(round.next.score, A.score);
assert.equal(round.next.best, A.score);
assert.deepEqual(round.next.accepted, [], 'A losing novel candidate must not consume novelty');
round = selectRound(base, [C, E], configs.coding);
assert.equal(round.winnerIndex, 0);
assert.deepEqual(round.next.accepted, []);
// A single admissible regression can win: the incumbent is only the fallback.
round = selectRound(base, [E], configs.coding);
assert.equal(round.next.score, .809);
assert.equal(round.next.best, .82);
assert.deepEqual(round.next.accepted, ['skill']);
let repeat = selectRound(round.next, [E], configs.coding);
assert.equal(repeat.decisions[0].novelty, 0);
assert.equal(repeat.winnerIndex, null);
assert.equal(repeat.next.score, .809);
// The historical best stays fixed through accepted regressions and rejected gains.
round = selectRound(base, [{score: .804, cost: 900000, touched: [], guardOK: true},
  {score: .99, cost: 1, touched: ['memory'], guardOK: false}], configs.coding);
assert.equal(round.next.score, .804);
assert.equal(round.next.best, .82);
assert.deepEqual(round.next.accepted, []);
repeat = selectRound(round.next, [{score: .802, cost: 1, touched: [], guardOK: true}], configs.coding);
assert.equal(repeat.decisions[0].reason, 'floor');
assert.equal(repeat.next.best, .82);
// Ties follow iteration order, rather than adding an undocumented cost ranking.
round = selectRound(base, [{...C, cost: 900000}, {...C, cost: 800000}], configs.coding);
assert.equal(round.winnerIndex, 0);
assert.equal(round.next.cost, 900000);
const baselineCopy = JSON.stringify(base), poolCopy = JSON.stringify([A, C, E]);
selectRound(base, [A, C, E], configs.coding);
assert.equal(JSON.stringify(base), baselineCopy);
assert.equal(JSON.stringify([A, C, E]), poolCopy);
round = selectRound(base, [], configs.coding);
assert.equal(round.winnerIndex, null);
assert.equal(round.next.score, base.score);
assert.notEqual(round.next.accepted, base.accepted);
assert.throws(() => selectRound({...base, cost: 0}, [], configs.coding));
console.log('RRSI single-candidate boundaries, pool ranking, fallback, multi-round floor/novelty, ties and immutability passed');
