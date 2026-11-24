#!/usr/bin/env node
// Reachability oracle expressed as reverse sets of integer positions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {calculate} = require('../assets/js/contact-window.js');

function predecessors(targets, step, distance) {
  const reached = new Set();
  for (const target of targets) {
    for (let position = Math.max(0, target - step); position <= Math.min(distance, target + step); position++) {
      reached.add(position);
    }
  }
  return reached;
}

function check({distance, step, intervals, pinned}) {
  const actual = calculate({distance, step, intervals, pinned});
  const free = intervals - pinned;
  let best = distance;
  let levels;
  for (; best >= 0; best--) {
    levels = [new Set([best])];
    for (let length = 0; length < free; length++) {
      levels.push(predecessors(levels.at(-1), step, distance));
    }
    if (levels.at(-1).has(0)) break;
  }
  const expectedPath = Array(pinned + 1).fill(0);
  for (let remaining = free - 1; remaining >= 0; remaining--) {
    const current = expectedPath.at(-1);
    const available = [...levels[remaining]].filter(position => Math.abs(position - current) <= step);
    expectedPath.push(Math.min(...available));
  }
  const lastStep = free > 0 ? predecessors(new Set([0]), step, distance) : new Set([0]);
  const lateEnd = Math.max(...lastStep);
  assert.deepEqual(actual, {
    free_intervals: free,
    late_gap: distance - lateEnd,
    window_gap: distance - best,
    late_path: [...Array(intervals).fill(0), lateEnd],
    window_path: expectedPath,
  });
}

const fixturePath = path.join(__dirname, '../content/posts/robotics/contact-preserving-retargeting/assets/window-default.json');
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
assert.deepEqual(calculate(fixture.input), fixture.result);

let cases = 0;
for (const distance of [0, 1, 3, 7, 12, 23, 100]) {
  for (const step of [0, 1, 2, 4, 11, 100]) {
    for (const intervals of [1, 2, 3, 5, 12]) {
      for (let pinned = 0; pinned <= intervals; pinned++) {
        check({distance, step, intervals, pinned});
        cases++;
      }
    }
  }
}

const invalidInputs = [
  {distance: -1}, {distance: '12'}, {distance: 101}, {distance: true},
  {step: NaN}, {step: 1.5}, {step: Infinity}, {step: 101},
  {intervals: 0}, {intervals: 13}, {pinned: -1}, {pinned: 4},
];
for (const changes of invalidInputs) {
  assert.throws(() => calculate({...fixture.input, ...changes}), RangeError);
}

console.log(JSON.stringify({cases, invalidCases: invalidInputs.length, fallback: 'matches'}));
