#!/usr/bin/env node
// Exercise the exact vendored search library with independent article fixtures.
const assert = require('node:assert/strict');
const FlexSearch = require('../assets/vendor/flexsearch/flexsearch.bundle.min.js');
const {normalize, queryTerms, encode, pageIndexFields} = require('../assets/js/core/search-terms.js');

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
  document: {id: 'id', store: ['title'], index: pageIndexFields()},
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

// Repeating terms in chapters at different positions used to put one page ID
// into several score buckets. That could both duplicate single-term results
// and remove another matching page from a multi-term intersection.
const makePageIndex = () => new FlexSearch.Document({
  tokenize: 'forward', encode,
  document: {id: 'id', store: ['title'], index: pageIndexFields()},
});
const chapterIndex = makePageIndex();
chapterIndex.add({id: 1, title: 'Single chapter', content: 'qk norm'});
chapterIndex.add({id: 2, title: 'Several chapters', content: ''});
chapterIndex.append({id: 2, content: 'qk norm'});
chapterIndex.append({id: 2, content: 'pad qk norm'});
chapterIndex.add({id: 3, title: 'Terms across chapters', content: 'qk'});
chapterIndex.append({id: 3, content: 'norm'});
const bodyIds = (doc, query, limit = 100) =>
  doc.search(query, limit, {index: 'content', suggest: false})[0]?.result || [];
for (const query of ['qk', 'norm', 'qk norm', 'QK-Norm']) {
  const ids = bodyIds(chapterIndex, query);
  assert.equal(ids.length, 3, query);
  assert.deepEqual(new Set(ids), new Set([1, 2, 3]), query);
}

const limitedIndex = makePageIndex();
for (let id = 1; id <= 30; id++) {
  limitedIndex.add({id, title: 'Page ' + id, content: 'needle norm'});
  for (const content of ['pad needle norm', 'pad pad needle norm', 'needle pad norm']) {
    limitedIndex.append({id, content});
  }
}
for (const query of ['needle', 'needle norm']) {
  const ids = bodyIds(limitedIndex, query, 20);
  assert.equal(ids.length, 20, query);
  assert.equal(new Set(ids).size, 20, 'Repeated chapters consumed the result limit');
}

// Compare chapter-wise indexing with a whole-text oracle across Latin/CJK
// tokens, punctuation, prefix queries and independently shuffled chapters.
let seed = 7321;
const random = maximum => {
  seed = (1664525 * seed + 1013904223) >>> 0;
  return seed % maximum;
};
const vocabulary = ['qk', 'norm', 'rms', 'gradient', 'update', 'head_dim',
  'action', 'head', '机械臂', '轨迹', '奇异值', '噪声', '控制', 'C++', 'A*'];
const chunked = makePageIndex();
const whole = new FlexSearch.Document({
  tokenize: 'forward', encode,
  document: {id: 'id', index: ['content']},
});
for (let id = 1; id <= 12; id++) {
  const chapters = Array.from({length: 2 + random(5)}, () =>
    Array.from({length: 2 + random(8)}, () => vocabulary[random(vocabulary.length)]).join(' '));
  whole.add({id, content: chapters.join(' ')});
  chunked.add({id, title: 'Article ' + id, content: ''});
  for (let n = chapters.length - 1; n > 0; n--) {
    const other = random(n + 1);
    [chapters[n], chapters[other]] = [chapters[other], chapters[n]];
  }
  for (const content of chapters) chunked.append({id, content});
}
for (let n = 0; n < 400; n++) {
  const query = Array.from({length: 1 + random(3)}, () => {
    const word = vocabulary[random(vocabulary.length)];
    return /^[a-z_]+$/.test(word) && random(2) ? word.slice(0, Math.max(1, word.length - 1)) : word;
  }).join(' ');
  const ids = bodyIds(chunked, query);
  assert.equal(ids.length, new Set(ids).size, query);
  assert.deepEqual(new Set(ids), new Set(bodyIds(whole, query)), query);
}
console.log(JSON.stringify({library: 'FlexSearch 0.8.143', checks:
  'NFKC, CJK bigrams and final characters, supplementary Unicode, technical punctuation, no partial-term suggestions, field-specific ranking, chapter intersections, unique limited results and 400 whole-text oracle queries passed'}));
