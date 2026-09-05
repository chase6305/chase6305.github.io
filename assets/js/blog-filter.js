// Progressive enhancement: native Hugo cards and pagination remain the fallback.
(() => {
  const sorts = new Set(["newest", "oldest", "shortest"]);
  const normalize = value => String(value).normalize("NFKC").toLocaleLowerCase("en").trim();
  const readState = (search, topics) => {
    const params = new URLSearchParams(search);
    return {
      q: (params.get("q") || "").trim().slice(0, 120),
      topic: topics.has(params.get("topic")) ? params.get("topic") : "",
      sort: sorts.has(params.get("sort")) ? params.get("sort") : "newest"
    };
  };
  const selectArticles = (articles, state) => {
    const words = normalize(state.q).split(/\s+/).filter(Boolean);
    return articles.filter(article => {
      if (state.topic && article.topic !== state.topic) return false;
      const haystack = normalize([article.title, article.summary, article.category,
        article.topicTitle, ...article.tags.map(tag => tag.name)].join(" "));
      return words.every(word => haystack.includes(word));
    }).sort((a, b) => {
      if (state.sort === "shortest" && a.minutes !== b.minutes) return a.minutes - b.minutes;
      const dates = a.date.localeCompare(b.date);
      if (dates) return state.sort === "oldest" ? dates : -dates;
      return a.url.localeCompare(b.url, "en");
    });
  };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {normalize, readState, selectArticles};
  }
  if (typeof document === "undefined") return;

  document.addEventListener("DOMContentLoaded", () => {
    const root = document.querySelector("[data-blog-index]");
    if (!root) return;
    const form = root.querySelector("[data-blog-filter]");
    const original = root.querySelector("[data-blog-original-list]");
    const pager = root.querySelector("[data-blog-original-pager]");
    const results = root.querySelector("#blog-filter-results");
    const empty = root.querySelector("[data-blog-empty]");
    const status = root.querySelector("[data-blog-filter-status]");
    const query = form.querySelector("[name=q]");
    const topic = form.querySelector("[name=topic]");
    const sort = form.querySelector("[name=sort]");
    const reset = form.querySelector("[data-blog-reset]");
    const defaultStatus = status.textContent;
    const topics = new Set(Array.from(topic.options).map(option => option.value));
    const localPath = value => typeof value === "string" && value.startsWith("/") && !value.startsWith("//");
    let articles;
    try {
      articles = JSON.parse(root.querySelector("#blog-filter-data").textContent);
      if (!Array.isArray(articles) || !articles.every(article =>
        localPath(article.url) && typeof article.title === "string" &&
        typeof article.summary === "string" && typeof article.date === "string" &&
        typeof article.minutes === "number" && Array.isArray(article.tags) &&
        article.tags.every(tag => typeof tag.name === "string" && localPath(tag.url)))) return;
    } catch {
      return; // Keep the server-rendered list usable if data cannot be read.
    }

    const element = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    };
    const link = (href, text, className) => {
      const node = element("a", className, text);
      node.href = href;
      return node;
    };
    const card = article => {
      const node = element("article", "blog-card");
      node.lang = article.lang || "zh-CN";
      const meta = element("div", "blog-card__meta");
      const date = element("time", "", article.date);
      date.dateTime = article.date;
      meta.append(date);
      if (article.category) meta.append(element("span", "", article.category));
      meta.append(element("span", "", "约 " + article.minutes + " 分钟"));
      if (article.draft) meta.append(element("span", "blog-draft-label", "草稿预览"));
      const heading = element("h2");
      heading.append(link(article.url, article.title));
      const footer = element("div", "blog-card__footer");
      const tags = element("div", "blog-card__tags");
      article.tags.forEach(tag => tags.append(link(tag.url, "#" + tag.name)));
      const more = link(article.url, "阅读全文 →", "blog-card__read");
      more.setAttribute("aria-label", "阅读 " + article.title);
      footer.append(tags, more);
      node.append(meta, heading, element("p", "", article.summary), footer);
      return node;
    };

    let timer;
    let composing = false;
    const apply = (updateURL = true) => {
      clearTimeout(timer);
      const state = {q: query.value.trim().slice(0, 120), topic: topic.value, sort: sort.value};
      const active = Boolean(state.q || state.topic || state.sort !== "newest");
      // Build before hiding the fallback, so a rendering failure cannot blank it.
      const matches = active ? selectArticles(articles, state) : [];
      const fragment = document.createDocumentFragment();
      matches.forEach(article => fragment.append(card(article)));
      results.replaceChildren(fragment);
      original.hidden = active;
      if (pager) pager.hidden = active;
      results.hidden = !active || !matches.length;
      empty.hidden = !active || matches.length > 0;
      status.textContent = active ? "找到 " + matches.length + " / " + articles.length + " 篇文章。" : defaultStatus;
      reset.disabled = !active;
      if (updateURL) {
        const url = new URL(location.href);
        for (const name of ["q", "topic", "sort"]) {
          const value = state[name];
          if (value && !(name === "sort" && value === "newest")) url.searchParams.set(name, value);
          else url.searchParams.delete(name);
        }
        try {
          if (url.href !== location.href) history.replaceState(history.state, "", url);
        } catch {
          // Restricted history APIs must not disable local filtering.
        }
      }
    };
    const restore = () => {
      const state = readState(location.search, topics);
      query.value = state.q;
      topic.value = state.topic;
      sort.value = state.sort;
      apply(false);
    };
    form.addEventListener("submit", event => { event.preventDefault(); if (!composing) apply(); });
    query.addEventListener("compositionstart", () => {
      composing = true;
      clearTimeout(timer);
    });
    query.addEventListener("input", event => {
      clearTimeout(timer);
      if (!event.isComposing && !composing) timer = setTimeout(apply, 180);
    });
    query.addEventListener("compositionend", () => {
      composing = false;
      clearTimeout(timer);
      timer = setTimeout(apply, 180);
    });
    topic.addEventListener("change", () => apply());
    sort.addEventListener("change", () => apply());
    reset.addEventListener("click", () => {
      query.value = "";
      topic.value = "";
      sort.value = "newest";
      apply();
      query.focus();
    });
    window.addEventListener("popstate", restore);
    window.addEventListener("pageshow", event => { if (event.persisted) restore(); });
    restore();
    form.hidden = false;
  });
})();
