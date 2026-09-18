#!/usr/bin/env node
// Pure arithmetic checks; interactive browser checks cover form behavior.
const assert = require('node:assert/strict');
const {calculate, presets} = require('../assets/js/llm-budget.js');
const example = calculate(presets.example);
assert.equal(example.weights, 14e9);
assert.equal(example.kv, 512 * 2 ** 20);
assert.equal(example.total, example.weights + example.kv);
const qwen = calculate(presets.qwen);
assert.equal(qwen.weights, 15231233024);
assert.equal(qwen.kv, 224 * 2 ** 20);
for (const factor of [2, 4, 8]) {
  const moreSequences = calculate({...presets.example, batch: factor});
  const longerContext = calculate({...presets.example, tokens: 4096 * factor});
  assert.equal(moreSequences.weights, example.weights);
  assert.equal(moreSequences.kv, example.kv * factor);
  assert.equal(longerContext.kv, moreSequences.kv);
}
const quantized = calculate({...presets.example, weightBits: 4});
assert.equal(quantized.weights, example.weights / 4);
assert.equal(quantized.kv, example.kv);
const keys = ['parameters', 'layers', 'kvHeads', 'headDim', 'batch', 'tokens'];
for (const key of keys) {
  for (const value of [NaN, Infinity, -1, 0, undefined, '7']) {
    assert.throws(() => calculate({...presets.example, [key]: value}), RangeError);
  }
}
for (const key of keys.filter(key => key !== 'parameters')) {
  assert.throws(() => calculate({...presets.example, [key]: 1.5}), RangeError);
}
for (const key of ['weightBits', 'kvBits']) {
  assert.throws(() => calculate({...presets.example, [key]: 3}), RangeError);
}
console.log('LLM payload calculator: reference values, dimensional scaling and invalid inputs passed');
