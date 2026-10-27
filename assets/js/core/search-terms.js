// Shared encoder for the browser search indexes and their offline regression.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.chaseSearchTerms = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const normalize = value => String(value ?? '').normalize('NFKC').toLowerCase();
  const isCJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
  const words = /c\+\+|c#|a\*|[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+|[0-9A-Za-z_а-яÀ-ſЀ-ӿ؀-ۿঀ-৿Ḁ-ỿ֐-׿\u0370-\u03ff]+/gu;
  const queryTerms = value => normalize(value).match(words) || [];
  function encode(value) {
    const terms = [];
    for (const word of queryTerms(value)) {
      if (!isCJK.test(word)) {
        terms.push(word);
        continue;
      }
      const characters = Array.from(word);
      for (let i = 0; i + 1 < characters.length; i++) terms.push(characters[i] + characters[i + 1]);
      // The forward tokenizer indexes each bigram's leading character. Include
      // the last one as well so single-character queries still work at the end.
      terms.push(characters[characters.length - 1]);
    }
    return terms;
  }
  return {normalize, queryTerms, encode};
});
