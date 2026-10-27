#!/usr/bin/env node
// Exercise the exact vendored search library with independent article fixtures.
const assert = require('node:assert/strict');
const FlexSearch = require('../assets/vendor/flexsearch/flexsearch.bundle.min.js');
const {normalize, queryTerms, encode} = require('../assets/js/core/search-terms.js');

assert.equal(normalize('Ｃ＋＋　Ａ＊'), 'c++ a*');
assert.deepEqual(queryTerms('Ｃｏｓｍｏｓ训练 World Model'), ['cosmos', '训练', 'world', 'model']);
assert.deepEqual(encode('四元数'), ['四元', '元数', '数']);
assert.deepEqual(encode('𠀀元'), ['𠀀元', '元']);
assert.deepEqual(encode('Ｃ＋＋ Ａ＊ std:: atomic_shared_ptr'), ['c++', 'a*', 'std', 'atomic_shared_ptr']);
assert.deepEqual(encode(' θ Δ '), ['θ', 'δ']);
assert.deepEqual(encode('+++ ∇'), []);
assert.deepEqual(encode(null), []);

const index = new FlexSearch.Index({tokenize: 'forward', encode});
const fixtures = new Map([
  [1, '机器人关节控制'],
  [2, '无关的章节'],
  [3, 'C++ 智能指针 std::shared_ptr'],
  [4, 'Cosmos CUDA CCTag c'],
  [5, 'A* 三维栅格搜索'],
  [6, 'Adam 优化器 a'],
  [7, '四元数旋转'],
  [8, '四种优化与变量元素'],
  [9, '变量 θ 与 Δ'],
]);
fixtures.forEach((text, id) => index.add(id, text));
const search = query => index.search(query, {suggest: false});
for (const query of ['关节', 'Ｃ＋＋', 'C++', 'A*', 'Ａ＊', '四元数', 'θ']) {
  const expected = {'关节': 1, 'Ｃ＋＋': 3, 'C++': 3, 'A*': 5, 'Ａ＊': 5, '四元数': 7, 'θ': 9}[query];
  assert.deepEqual(search(query), [expected], query);
}
assert(search('节').includes(1) && search('节').includes(2), 'Single-character query must still work');
assert(search('转').includes(7), 'Final CJK character must remain searchable');
assert.deepEqual(search('机器人 zzxxyy'), []);
assert.deepEqual(search('不存在的随机词zzxxyy'), []);
assert.deepEqual(search('+++'), []);
assert.deepEqual(search('Ｃｏｓｍｏｓ'), search('Cosmos'));

const document = new FlexSearch.Document({
  tokenize: 'forward', encode,
  document: {id: 'id', store: ['title'], index: ['title', 'headings', 'content']},
});
document.add({id: 1, title: '机器人滤波', headings: '四元数归一化', content: '四元数旋转误差'});
document.add({id: 2, title: '四元数入门', headings: '一个例子', content: '四元数旋转误差'});
document.add({id: 3, title: '训练指标', headings: '三种变量', content: '四种优化与变量元素'});
const fields = document.search('四元数', 20, {enrich: true, suggest: false, index: ['title', 'headings', 'content']});
assert.deepEqual(fields.find(f => f.field === 'title').result.map(r => r.id), [2]);
assert.deepEqual(fields.find(f => f.field === 'headings').result.map(r => r.id), [1]);
assert.deepEqual(new Set(fields.find(f => f.field === 'content').result.map(r => r.id)), new Set([1, 2]));
document.append({id: 2, content: 'additional context'});
const appended = document.search('四元数 additional', 20, {enrich: true, suggest: false, index: 'content'});
assert.deepEqual(appended[0].result.map(r => r.id), [2]);
assert.equal(appended[0].result[0].doc.title, '四元数入门');
console.log(JSON.stringify({library: 'FlexSearch 0.8.143', checks:
  'NFKC, CJK bigrams and final characters, supplementary Unicode, technical punctuation, no partial-term suggestions, field-specific ranking passed'}));
