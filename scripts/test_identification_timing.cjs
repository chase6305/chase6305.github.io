#!/usr/bin/env node
// Independent phase-expansion and Python oracles for the browser's two-column QR.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {calculate, presets} = require('../assets/js/identification-timing.js');
const bundle = path.join(__dirname, '../content/posts/robotics/dynamics/parameter-identification');
function close(actual, expected, tolerance = 1e-11) {
  assert(Number.isFinite(actual));
  assert(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), actual + ' != ' + expected);
}
const reference = JSON.parse(fs.readFileSync(path.join(bundle, 'assets/timing-reference.json')));
const result = calculate(presets.delayed);
for (const [key, field] of Object.entries({mass: 'mass_kg', damping: 'damping_n_s_per_m', trainRmse: 'train_rmse_n',
  validationLoggedRmse: 'validation_logged_rmse_n', validationReferenceRmse: 'validation_reference_rmse_n'})) close(result[key], reference[field]);
assert(result.trainRmse < 1e-12);
assert(result.validationLoggedRmse > .16 && result.validationReferenceRmse > .11);
const aligned = calculate(presets.aligned);
close(aligned.mass, 2); close(aligned.damping, .4);
assert(aligned.trainRmse < 1e-12 && aligned.validationReferenceRmse < 1e-12);
assert(aligned.validationLoggedRmse > .27); // A delayed record is a different comparison target.
const sameFrequency = calculate({...presets.delayed, validationHz: .7});
assert(sameFrequency.validationLoggedRmse < 1e-12 && sameFrequency.validationReferenceRmse > .01);
for (const corrected of [false, true]) {
  const zero = calculate({...presets.delayed, delayMs: 0, correctTrainingDelay: corrected});
  close(zero.mass, 2); close(zero.damping, .4);
  assert(zero.validationLoggedRmse < 1e-12 && zero.validationReferenceRmse < 1e-12);
}
let seed = 20261030;
function random() { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 2 ** 32; }
for (let index = 0; index < 500; index++) {
  const input = {delayMs: 100 * random() - 50, trainHz: .1 + 1.9 * random(), validationHz: .1 + 1.9 * random(), correctTrainingDelay: random() > .5};
  const before = JSON.stringify(input), actual = calculate(input), omega = 2 * Math.PI * input.trainHz;
  const phase = input.correctTrainingDelay ? 0 : omega * input.delayMs / 1000;
  close(actual.mass, 2 * Math.cos(phase) - .4 / omega * Math.sin(phase));
  close(actual.damping, .4 * Math.cos(phase) + 2 * omega * Math.sin(phase));
  assert(actual.trainRmse < 1e-11);
  if (input.correctTrainingDelay) assert(actual.validationReferenceRmse < 1e-11);
  assert.equal(JSON.stringify(input), before);
  assert.equal(actual.validation.length, 300);
}
let invalid = 0;
for (const name of ['delayMs', 'trainHz', 'validationHz']) {
  const bad = [NaN, Infinity, -Infinity, true, false, null, undefined, '', '0.7'];
  bad.push(...(name === 'delayMs' ? [-50.001, 50.001] : [0, -.1, .099, 2.001]));
  for (const value of bad) { assert.throws(() => calculate({...presets.delayed, [name]: value})); invalid++; }
}
for (const value of [0, 1, null, undefined, '', 'false']) { assert.throws(() => calculate({...presets.delayed, correctTrainingDelay: value})); invalid++; }
const inputs = [presets.delayed, presets.aligned, {...presets.delayed, delayMs: -20},
  {...presets.delayed, trainHz: .1, validationHz: 2, delayMs: 50},
  {...presets.delayed, trainHz: 2, validationHz: .1, delayMs: -50},
  {...presets.delayed, validationHz: .7}];
for (const input of inputs) {
  const args = ['-B', path.join(bundle, 'timing_counterexample.py'), '--delay-ms', String(input.delayMs),
    '--train-hz', String(input.trainHz), '--validation-hz', String(input.validationHz)];
  if (input.correctTrainingDelay) args.push('--correct-training-delay');
  const expected = JSON.parse(execFileSync('python3', args, {encoding: 'utf8', timeout: 10000}));
  const actual = calculate(input);
  for (const [key, field] of Object.entries({mass: 'mass_kg', damping: 'damping_n_s_per_m', trainRmse: 'train_rmse_n',
    validationLoggedRmse: 'validation_logged_rmse_n', validationReferenceRmse: 'validation_reference_rmse_n'})) close(actual[key], expected[field]);
}
console.log(JSON.stringify({randomPhaseOracleCases: 500, invalidInputs: invalid, independentPythonCases: inputs.length,
  correctedVsDelayedTargets: true, immutableInputs: true}));
