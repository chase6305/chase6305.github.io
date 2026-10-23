#!/usr/bin/env node
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {evaluate, presets} = require('../static/labs/supermimic/gradient_lab.js');

function near(actual, expected, label = '') {
  assert(Math.abs(actual - expected) <= 1e-7 * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
}

const full = evaluate(presets.full);
assert.deepEqual(full.model.states, [0, .2, .4]);
near(full.model.totalReturn, -1);
near(full.full.total, 8);
near(full.finiteDifference, 8);
near(full.selected.contributions[0], 3.2);
near(full.selected.contributions[1], 4.8);
near(full.after.theta, .18);
near(full.after.reference.totalReturn, -.488);

// A detached state changes the backward dependency, not the forward function.
const detached = evaluate(presets.detached);
assert.deepEqual(detached.model, full.model);
near(detached.selected.stateDerivatives[2], 2);
near(detached.selected.total, 5.6);
near(detached.full.total, 8);
near(detached.finiteDifference, 8);
near(detached.after.theta, .156);
near(detached.after.reference.totalReturn, -.61472);

// Both logged states can match while the return derivative has the wrong sign.
const opposite = evaluate(presets.opposite);
assert.deepEqual(opposite.model.states, opposite.reference.states);
near(opposite.stateRMSE, 0);
near(opposite.referenceGradient, 8);
near(opposite.full.total, -8);
near(opposite.finiteDifference, -8);
near(opposite.after.theta, .02);
near(opposite.after.model.totalReturn, -.488);
near(opposite.after.reference.totalReturn, -1.768);
assert(opposite.after.modelReturnChange > 0 && opposite.after.referenceReturnChange < 0);

// Independent closed form and finite differences across signs and stationary points.
let cases = 0;
for (const theta of [-.5, -.1, .1, .3, .75]) {
  for (const gain of [-4, -2, 0, 2, 4]) {
    for (const offset of [-1, 0, .4, 1]) {
      const p = {theta, gain, offset};
      const result = evaluate(p);
      const increment = gain * theta + offset;
      const expected = -((increment - 1) ** 2) - (2 * increment - 1) ** 2;
      near(result.model.totalReturn, expected);
      near(result.full.total, 6 * gain - 10 * gain * increment);
      near(result.finiteDifference, result.full.total);
      near(result.referenceGradient, 12 - 40 * theta);
      const broken = evaluate({...p, detach: true});
      near(broken.selected.total, 4 * gain - 6 * gain * increment);
      assert.deepEqual(broken.model.states, result.model.states);
      cases++;
    }
  }
}
near(evaluate({theta: .3}).full.total, 0);
assert(evaluate({theta: .1, learningRate: .1}).after.referenceReturnChange < 0,
  'A large step can worsen return even with a correct gradient');
for (const invalid of [null, [], {theta: NaN}, {gain: Infinity}, {offset: 2},
  {learningRate: 0}, {theta: '0.1'}, {detach: 'false'}, {theta: .75001}]) {
  assert.throws(() => evaluate(invalid));
}

// The downloadable payload is recomputed from inputs; stored results are not trusted.
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'supermimic-gradient-'));
try {
  const file = path.join(directory, 'case.json');
  fs.writeFileSync(file, JSON.stringify({input: presets.opposite, result: {full: {total: 999}}}));
  const cli = JSON.parse(execFileSync(process.execPath,
    [path.resolve(__dirname, '../static/labs/supermimic/gradient_lab.js'), file], {encoding: 'utf8'}));
  near(cli.full.total, -8);
  near(cli.after.reference.totalReturn, -1.768);
} finally {
  fs.rmSync(directory, {recursive: true, force: true});
}
console.log(JSON.stringify({presets: 3, closedFormCases: cases, finiteDifferences: 'passed',
  detachedForward: 'unchanged', oppositeGradient: 'verified', invalidInputs: 'rejected', cliRecalculation: 'passed'}));
