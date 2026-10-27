// Copyright (c) 2023 Xin; MIT. Hextra 38d18a5 search data-plane override.
// License: /licenses/hextra.txt
// FlexSearch data plane. Builds the page + section indexes lazily on first
// use, runs queries, and returns normalized results. UI lives in
// search-dialog.js and consumes this via `window.hextraSearch`.

// Render the search data as JSON.
// {{ $searchDataFile := printf "%s.search-data.json" .Language.Lang }}
// {{ $searchData := resources.Get "json/search-data.json" | resources.ExecuteAsTemplate $searchDataFile . }}
// {{ if hugo.IsProduction }}
//   {{ $searchData = $searchData | minify | fingerprint }}
// {{ end }}

(function () {
  const searchDataURL = '{{ $searchData.RelPermalink }}';

  let pageIndex;
  let sectionIndex;
  let indexPromise = null;

  /**
   * Preloads the search index by fetching data and adding it to the FlexSearch index.
   * @returns {Promise<void>} A promise that resolves when the index is preloaded.
   */
  function preloadIndex() {
    if (indexPromise) return indexPromise;
    const attempt = (async () => {
      const tokenize = '{{- site.Params.search.flexsearch.tokenize | default  "forward" -}}';

      const {encode} = window.chaseSearchTerms;

      pageIndex = new FlexSearch.Document({
        tokenize,
        encode,
        cache: 100,
        document: {
          id: 'id',
          store: ['title', 'breadcrumbs', 'route'],
          index: ['title', 'headings', 'content']
        }
      });

      sectionIndex = new FlexSearch.Document({
        tokenize,
        encode,
        cache: 100,
        document: {
          id: 'id',
          store: ['title', 'content', 'url', 'display', 'crumb'],
          index: "content",
          tag: [{
            field: "pageId"
          }]
        }
      });

      const resp = await fetch(searchDataURL);
      if (!resp.ok) throw new Error('Search data request failed: ' + resp.status);
      const data = await resp.json();
      if (!data || typeof data !== 'object' || Array.isArray(data)
        || !Object.values(data).every(page => page && typeof page.title === 'string'
          && page.data && typeof page.data === 'object' && !Array.isArray(page.data)
          && Object.values(page.data).every(content => typeof content === 'string'))) {
        throw new Error('Invalid search data');
      }
      let pageId = 0;
      let sliceStarted = performance.now();
      const yieldIfNeeded = async () => {
        if (performance.now() - sliceStarted < 12) return;
        if (window.scheduler?.yield) await window.scheduler.yield();
        else await new Promise(resolve => setTimeout(resolve, 0));
        sliceStarted = performance.now();
      };
      for (const route in data) {
        const pageParts = [];
        let pageHeadings = '';
        ++pageId;
        const urlParts = route.split('/').filter(x => x !== "" && !x.startsWith('#'));

        let crumb = '';
        const crumbParts = [];
        let searchUrl = '/';
        for (let i = 0; i < urlParts.length; i++) {
          const urlPart = urlParts[i];
          searchUrl += urlPart + '/'

          const crumbData = data[searchUrl];
          if (!crumbData) {
            console.debug('Excluded page', searchUrl, '- will not be included for search result breadcrumb for', route);
            continue;
          }

          let title = data[searchUrl].title;
          if (title === "_index") {
            title = urlPart.split("-").map(x => x).join(" ");
          }
          crumbParts.push(title);
          crumb += title;

          if (i < urlParts.length - 1) {
            crumb += ' > ';
          }
        }

        for (const heading in data[route].data) {
          const separator = heading.indexOf('#');
          const hash = separator === -1 ? heading : heading.slice(0, separator);
          const text = separator === -1 ? '' : heading.slice(separator + 1);
          const url = hash ? `${route}#${hash}` : route;
          const title = text || data[route].title;

          const content = data[route].data[heading] || '';
          const paragraphs = content.split('\n').filter(Boolean);

          sectionIndex.add({
            id: url,
            url,
            title,
            crumb,
            pageId: `page_${pageId}`,
            content: title,
            ...(paragraphs[0] && { display: paragraphs[0] })
          });

          for (let i = 0; i < paragraphs.length; i++) {
            sectionIndex.add({
              id: `${url}_${i}`,
              url,
              title,
              crumb,
              pageId: `page_${pageId}`,
              content: paragraphs[i]
            });
            if ((i + 1) % 32 === 0) await yieldIfNeeded();
          }

          pageParts.push(`${title} ${content}`);
          if (hash) pageHeadings += ` ${title}`;
          await yieldIfNeeded();
        }

        pageIndex.add({
          id: pageId,
          title: data[route].title,
          route,
          headings: pageHeadings,
          breadcrumbs: crumbParts.slice(0, -1),
          content: ''
        });
        // Append one chapter at a time so a long article does not become one
        // multi-second indexing task on a slower device. The document store
        // (title/route/breadcrumbs) stays intact across these partial appends.
        for (const content of pageParts) {
          pageIndex.append({id: pageId, content});
          await yieldIfNeeded();
        }
      }
    })();
    // Clear the cached promise on failure so the next call retries instead of
    // returning the rejection forever.
    const wrapped = attempt.catch((err) => {
      if (indexPromise === wrapped) indexPromise = null;
      pageIndex = undefined;
      sectionIndex = undefined;
      throw err;
    });
    indexPromise = wrapped;
    return indexPromise;
  }

  /**
   * Run the actual FlexSearch query and return sorted, deduped page groups.
   * @param {string} query
   * @returns {Array<{id: string, route: string, title: string, breadcrumbs: string[], matches: Array<{id: string, route: string, title: string, content: string}>}>}
   */
  function performSearch(query) {
    const maxPageResults = parseInt('{{- site.Params.search.flexsearch.maxPageResults | default 20 -}}', 10);
    const maxSectionResults = parseInt('{{- site.Params.search.flexsearch.maxSectionResults | default 10 -}}', 10);
    // Rank title matches before chapter headings and full body matches. Search
    // all three fields before limiting, so long pages cannot displace a title.
    const fields = pageIndex.search(query, maxPageResults, {
      enrich: true, suggest: false, index: ['title', 'headings', 'content']
    });
    const candidates = new Map();
    for (const field of fields) {
      const tier = field.field === 'title' ? 2 : field.field === 'headings' ? 1 : 0;
      for (const result of field.result) {
        const previous = candidates.get(result.id);
        if (!previous || tier > previous.tier) candidates.set(result.id, {...result, tier});
      }
    }
    const pageResults = [...candidates.values()];

    const groups = [];
    const pageTitleMatches = {};

    for (let i = 0; i < pageResults.length; i++) {
      const result = pageResults[i];
      pageTitleMatches[i] = 0;

      const sectionResults = sectionIndex.search(query,
        { enrich: true, suggest: false, tag: { 'pageId': `page_${result.id}` } })[0]?.result || [];
      const occurred = new Set();
      const group = {
        _page_rk: i,
        _page_tier: result.tier,
        route: result.doc.route,
        title: result.doc.title,
        breadcrumbs: result.doc.breadcrumbs || [],
        matches: []
      };

      for (let j = 0; j < sectionResults.length && group.matches.length < maxSectionResults; j++) {
        const { doc } = sectionResults[j];
        const { url, title } = doc;
        if (occurred.has(url)) continue;
        occurred.add(url);
        if (doc.display !== undefined) pageTitleMatches[i]++;
        const content = doc.display || doc.content;
        if (!group.route) group.route = url.split('#')[0];
        group.matches.push({
          route: url,
          title,
          content
        });
      }

      if (group.route) groups.push(group);
    }

    let optionId = 0;
    return groups
      .sort((a, b) => {
        if (a._page_tier !== b._page_tier) return b._page_tier - a._page_tier;
        if (pageTitleMatches[a._page_rk] !== pageTitleMatches[b._page_rk]) {
          return pageTitleMatches[b._page_rk] - pageTitleMatches[a._page_rk];
        }
        return a._page_rk - b._page_rk;
      })
      .slice(0, maxPageResults)
      .map((res) => ({
        id: `hextra-search-opt-${optionId++}`,
        route: res.route,
        title: res.title,
        breadcrumbs: res.breadcrumbs,
        matches: res.matches.map((match) => ({
          id: `hextra-search-opt-${optionId++}`,
          route: match.route,
          title: match.title,
          content: match.content
        }))
      }));
  }

  window.hextraSearch = {
    preload: preloadIndex,
    async search(query) {
      await preloadIndex();
      return performSearch(String(query).slice(0, 120));
    },
  };
})();
