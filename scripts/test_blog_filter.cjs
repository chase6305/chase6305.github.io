#!/usr/bin/env node
// Pure filter tests plus exact production/draft inventory checks. No browser needed.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {normalize, readState, selectArticles} = require("../assets/js/blog-filter.js");
const root = path.resolve(__dirname, "..");
const publicDir = path.resolve(process.argv[2] || path.join(root, "public"));
const drafts = process.argv.includes("--include-drafts");
const text = fs.readFileSync(path.join(publicDir, "posts/index.html"), "utf8");
const match = text.match(/<script\b[^>]*\bid=(?:"blog-filter-data"|'blog-filter-data'|blog-filter-data)[^>]*>([\s\S]*?)<\/script>/);
assert(match, "Missing filter inventory");
const articles = JSON.parse(match[1]);
const review = JSON.parse(fs.readFileSync(path.join(root, "docs/blog-editorial-review.json"))).posts;
const visible = review.filter(post => drafts || !post.draft);
const route = p => "/" + p.replace(/^content\//, "").replace(/\/index\.md$|\.md$/, "").toLowerCase() + "/";
assert.equal(articles.length, visible.length);
assert.deepEqual(new Set(articles.map(a => a.url)), new Set(visible.map(a => route(a.path))));
assert.equal(articles.filter(a => a.draft).length, drafts ? 3 : 0);
for (const post of visible) {
  const row = articles.find(a => a.url === route(post.path));
  assert.equal(row.summary, post.summary, "Escaped or altered summary: " + post.path);
}
assert(Buffer.byteLength(match[1]) < 100_000, "Metadata payload unexpectedly large");
const topicRecords = JSON.parse(fs.readFileSync(path.join(root, "data/blog_topics.json")));
const topics = new Set(topicRecords.map(t => t.id));
const defaults = {q: "", topic: "", sort: "newest"};
assert.equal(normalize(" ＰｙＴｈｏｎ "), "python");
assert.deepEqual(readState("?topic=unknown&sort=bad&q=%20Python%20", topics),
                 {q: "Python", topic: "", sort: "newest"});
assert.equal(readState("?q=" + "x".repeat(200), topics).q.length, 120);
assert.equal(readState("?topic=cpp&sort=shortest", topics).sort, "shortest");
for (const topic of topicRecords) {
  const selected = selectArticles(articles, {...defaults, topic: topic.id});
  const expected = topic.posts.map(p => route("content/" + p))
    .filter(url => visible.some(p => route(p.path) === url));
  assert.deepEqual(new Set(selected.map(a => a.url)), new Set(expected), topic.id);
}
for (const q of ["Python", "ｐｙｔｈｏｎ", "C++", "A*", "Jacobian", "零位"]) {
  assert(selectArticles(articles, {...defaults, q}).length > 0, q);
}
assert.equal(selectArticles(articles, {...defaults, q: "zzxxyy987notfound"}).length, 0);
const fixture = [
  {url: "/one/", title: "Python Qt", summary: "界面", tags: [], topic: "gui", date: "2025-01-01", minutes: 3},
  {url: "/two/", title: "C++", summary: "build", tags: [{name: "Qt"}], topic: "cpp", date: "2024-01-01", minutes: 7}
];
assert.deepEqual(selectArticles(fixture, {...defaults, q: "Python Qt"}).map(a => a.url), ["/one/"]);
assert.deepEqual(selectArticles(fixture, {...defaults, q: "Qt", topic: "cpp"}).map(a => a.url), ["/two/"]);
assert.equal(selectArticles(fixture, {...defaults, q: ".*"}).length, 0, "Query must not be a regex");
for (const sort of ["newest", "oldest", "shortest"]) {
  const selected = selectArticles(articles, {...defaults, sort});
  for (let i = 1; i < selected.length; i++) {
    const a = selected[i - 1], b = selected[i];
    if (sort === "shortest") assert(a.minutes <= b.minutes);
    else assert(sort === "newest" ? a.date >= b.date : a.date <= b.date);
  }
}
assert.deepEqual(articles, JSON.parse(match[1]), "Filtering must not mutate source order");
console.log(JSON.stringify({articles: articles.length, drafts: articles.filter(a => a.draft).length,
  topics: topics.size, metadata_bytes: Buffer.byteLength(match[1]),
  tests: "normalization, literal/AND matching, sorting, immutability, URL state, exact coverage passed"}, null, 2));
