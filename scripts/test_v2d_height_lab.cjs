#!/usr/bin/env node
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const {evaluate, makeTrace, presets} = require('../static/labs/video-to-data/height_lab.js');

const stable = evaluate(makeTrace(presets.stable));
assert(stable.success);
assert.equal(stable.sampleCount, 94);
assert.equal(stable.longest, 50);
// Preserve the exact downloadable preset samples across Python and JavaScript.
const companion = path.resolve(__dirname, '../content/posts/ai/video-to-data/data_contract_lab.py');
for (const [jsName, pythonName] of [['stable', 'stable_place'], ['aloft', 'held_aloft'], ['drop', 'abrupt_drop']]) {
  const payload = JSON.parse(execFileSync('python3', ['-B', companion, '--show-trace', pythonName], {encoding: 'utf8'}));
  assert.deepEqual(makeTrace(presets[jsName]), payload.heights_m, jsName + ' preset differs from article companion');
}
const aloft = evaluate(makeTrace(presets.aloft));
assert(!aloft.success && aloft.checks[0] && aloft.checks[1] && !aloft.checks[3]);
const drop = evaluate(makeTrace(presets.drop));
assert(!drop.success && !drop.checks[2] && drop.checks[4]);
assert.equal(drop.descent, .11);
assert(evaluate(makeTrace({...presets.stable, finalHeight: .03})).success);
assert(!evaluate(makeTrace({...presets.stable, finalHeight: .030001})).checks[3]);
assert(evaluate(makeTrace({...presets.stable, finalHeight: -.01})).success);
assert(!evaluate(makeTrace({...presets.stable, finalHeight: -.010001})).checks[5]);
assert(evaluate(makeTrace({...presets.stable, jitter: .0025})).success);
assert(!evaluate(makeTrace({...presets.stable, jitter: .0025001})).checks[4]);
// A short explicit tail can still have a complete, stable last-20 window.
assert(evaluate(makeTrace({...presets.stable, descentSteps: 200, terminalSamples: 1})).success);
assert(!evaluate([0, .11, ...Array(19).fill(0)]).success);
assert.equal(evaluate([.3]).sampleCount, 0);
assert(!evaluate([.3]).success);
assert(evaluate([0, .1, ...Array(19).fill(.1)]).checks[0]);
assert(evaluate([0, .1, ...Array(19).fill(.1)]).checks[1]);
assert(!evaluate([0, .1, ...Array(18).fill(.1)]).checks[1]);
// Compare the overall decision and value away from a decision threshold.
// A decimal shift can move an exact 0.05 tie across an IEEE-754 boundary.
const shifted = evaluate(makeTrace(presets.stable).map(h => h + .7));
assert.equal(shifted.success, stable.success);
assert(Math.abs(shifted.peak - stable.peak) < 1e-12);
for (const invalid of [[], [NaN], [0, Infinity], ['0'], null]) {
  assert.throws(() => evaluate(invalid));
}
for (const patch of [{descentSteps: 0}, {descentSteps: 2.5}, {terminalSamples: 201},
  {jitter: -.001}, {jitter: NaN}, {finalHeight: .21}]) {
  assert.throws(() => makeTrace({...presets.stable, ...patch}));
}
console.log('V2D presets, six height conditions, inclusive boundaries, sample windows, baseline and invalid inputs passed');
