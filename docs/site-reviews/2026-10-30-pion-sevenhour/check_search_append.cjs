#!/usr/bin/env node
// Compare real chapter-appended search content with a whole-text oracle.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../../..');
const args = process.argv.slice(2);
const option = name => {
  const index = args.indexOf(name);
  if (index === -1 || !args[index + 1]) throw Error('Required option: ' + name);
  return args[index + 1];
};
const publicDirectory = option('--public');
const output = option('--output');
const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const library = path.join(root, 'assets/vendor/flexsearch/flexsearch.bundle.min.js');
assert.equal(digest(library), '433e941a8a573ebb9931fc16fc75266ab6b93f569ac2fb4f3dc66882e0416f4c');
const FlexSearch = require(library);
const {encode, pageIndexFields} = require(path.join(root, 'assets/js/core/search-terms.js'));
const dataFiles = fs.readdirSync(publicDirectory)
  .filter(file => file.startsWith('en.search-data.') && file.endsWith('.json'));
assert.equal(dataFiles.length, 1, 'Use a fresh production build with one default-language search index');
const dataFile = path.join(publicDirectory, dataFiles[0]);
const data = JSON.parse(fs.readFileSync(dataFile));
const create = fields => new FlexSearch.Document({
  tokenize: 'forward', encode,
  document: {id: 'id', store: ['title', 'route'], index: fields},
});
const original = create(['title', 'headings', 'content']);
const current = create(pageIndexFields());
const reference = create(['content']);
const routes = new Map();
let id = 0;
for (const [route, page] of Object.entries(data)) {
  id++;
  routes.set(id, route);
  const parts = [];
  let headings = '';
  for (const [heading, content] of Object.entries(page.data)) {
    const separator = heading.indexOf('#');
    const hash = separator === -1 ? heading : heading.slice(0, separator);
    const title = (separator === -1 ? '' : heading.slice(separator + 1)) || page.title;
    parts.push(`${title} ${content}`);
    if (hash) headings += ' ' + title;
  }
  for (const index of [original, current]) {
    index.add({id, title: page.title, route, headings, content: ''});
    for (const content of parts) index.append({id, content});
  }
  reference.add({id, content: parts.join(' ')});
}
const queries = ['qk', 'norm', 'QK-Norm', 'qk rms', 'RMS', 'head_dim',
  'gradient norm', 'action head', '奇异值 噪声', 'Pion RMS', '形态 描述',
  'C++ 智能指针', 'Cosmos 训练', 'queue shutdown', '拷贝 构造', '不存在zzxxyy',
  'Gloo', '缓冲时序', 'missing_gradient decay', '通信 缓冲'];
const matches = (index, query, limit) =>
  index.search(query, limit, {index: 'content', suggest: false})[0]?.result || [];
const cases = [];
for (const query of queries) {
  const expected = matches(reference, query, 10000);
  const actual = matches(current, query, 10000);
  const before = matches(original, query, 10000);
  const limited = matches(current, query, 20);
  assert.equal(actual.length, new Set(actual).size, 'Duplicate current page: ' + query);
  assert.deepEqual(new Set(actual), new Set(expected), 'Whole-text mismatch: ' + query);
  assert.equal(limited.length, Math.min(20, expected.length), 'Repeated pages consumed the limit: ' + query);
  assert.equal(limited.length, new Set(limited).size);
  cases.push({query, expected: expected.map(id => routes.get(id)),
    actual: actual.map(id => routes.get(id)), previous: before.map(id => routes.get(id)),
    previousMissing: expected.filter(id => !before.includes(id)).map(id => routes.get(id)),
    previousUnexpected: before.filter(id => !expected.includes(id)).map(id => routes.get(id)),
    previousDuplicateCount: before.length - new Set(before).size});
}
const report = {scope: 'Actual public search data; page-body matching compared with a whole-text oracle. Title/heading and section relevance remain separate.',
  library: 'FlexSearch 0.8.143', pages: id, queries: cases.length,
  sourceSha256: {
    library: digest(library), data: digest(dataFile),
    terms: digest(path.join(root, 'assets/js/core/search-terms.js')),
    adapter: digest(path.join(root, 'assets/js/flexsearch.js')),
  },
  cases};
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({pages: id, queries: cases.length,
  previousIncompleteQueries: cases.filter(row => row.previousMissing.length).length,
  currentWholeTextMismatches: 0, currentDuplicatePages: 0}));
