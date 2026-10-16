/* Numerical companion to the article; no network calls or model execution. */
(function (root) {
  'use strict';
  const configs = {
    coding: {delta: 0.017, beta0: 0.10, beta1: 44.5, ws: 0, wc: 15, wn: 0.5},
    workspace: {delta: 0.004, beta0: 0.10, beta1: 35.4, ws: 1414, wc: 15, wn: 0.5},
    engineering: {delta: 0.020, beta0: 0.15, beta1: 24.4, ws: 244, wc: 2, wn: 0.5}
  };
  const types = ['client_tool', 'skill', 'memory', 'subagent'];
  function judge(input, cfg) {
    for (const name of ['score', 'best', 'candidate', 'delta', 'cost', 'candidateCost']) {
      if (!Number.isFinite(input[name])) throw new Error('请输入完整的有限数值。');
    }
    if ([input.score, input.best, input.candidate].some(n => n < 0 || n > 1)) {
      throw new Error('得分使用 [0,1] 小数，例如 0.835。');
    }
    if (input.best < input.score) throw new Error('历史最高分不能低于当前分数。');
    if (input.delta < 0 || input.delta > 1) throw new Error('噪声带须在 [0,1] 范围内。');
    if (input.cost <= 0 || input.candidateCost <= 0) throw new Error('两侧 token 成本都须为正，才能比较成本增长。');
    const touched = new Set(input.touched || []);
    const accepted = new Set(input.accepted || []);
    const novelty = types.filter(t => touched.has(t) && !accepted.has(t)).length;
    const ds = input.candidate - input.score;
    const dc = (input.candidateCost - input.cost) / input.cost;
    const floor = input.best - input.delta;
    const strong = ds > input.delta;
    const allowance = cfg.beta0 + cfg.beta1 * ds;
    const shaped = cfg.ws * ds - cfg.wc * dc + cfg.wn * novelty;
    const floorOK = input.candidate >= floor;
    const costOK = strong ? dc <= allowance : shaped > 0;
    const guardOK = Boolean(input.guardOK);
    return {ds, dc, novelty, floor, strong, allowance, shaped,
      floorOK, costOK, guardOK,
      admissible: floorOK && costOK && guardOK,
      reason: !floorOK ? 'floor' : !costOK ? 'cost' : !guardOK ? 'guard' : 'admissible'};
  }

  function selectRound(base, candidates, cfg) {
    // Validate the baseline even when the pool is empty. Keeping the incumbent
    // is a fallback, not another candidate that competes with admissible edits.
    judge({...base, candidate: base.score, candidateCost: base.cost,
      touched: [], guardOK: true}, cfg);
    const decisions = candidates.map(candidate => judge({...base,
      candidate: candidate.score, candidateCost: candidate.cost,
      touched: candidate.touched, guardOK: candidate.guardOK}, cfg));
    let winnerIndex = null;
    decisions.forEach((decision, index) => {
      if (decision.admissible && (winnerIndex === null ||
          candidates[index].score > candidates[winnerIndex].score)) winnerIndex = index;
    });
    const next = {score: base.score, best: base.best, cost: base.cost,
      delta: base.delta, accepted: [...new Set(base.accepted || [])]};
    if (winnerIndex !== null) {
      const winner = candidates[winnerIndex];
      next.score = winner.score;
      next.best = Math.max(base.best, winner.score);
      next.cost = winner.cost;
      next.accepted = [...new Set([...next.accepted, ...(winner.touched || [])])];
    }
    return {decisions, winnerIndex, next};
  }

  const api = {configs, types, judge, selectRound};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (!root.document) return;
  const doc = root.document;
  const form = doc.querySelector('#selection-form');
  if (!form) return;
  const field = name => form.elements.namedItem(name);
  const number = name => field(name).valueAsNumber;
  const checked = group => [...form.querySelectorAll(`input[name="${group}"]:checked`)].map(e => e.value);
  const fixed = n => Number(n.toPrecision(12)).toString();
  const percent = n => fixed(100 * n) + '%';
  const pool = [], rounds = [];
  let candidateSequence = 0;
  const button = id => doc.getElementById(id);
  function text(id, value) {
    const element = doc.getElementById(id);
    if (element.textContent !== value) element.textContent = value;
  }
  function readInput() {
    return {score: number('score'), best: number('best'), candidate: number('candidate'),
      delta: number('delta'), cost: number('cost'), candidateCost: number('candidateCost'),
      touched: checked('touched'), accepted: checked('accepted'), guardOK: field('guardOK').checked};
  }
  function updateScrollFocus() {
    const region = doc.getElementById('pool-scroll');
    region.tabIndex = region.scrollWidth > region.clientWidth + 1 ? 0 : -1;
  }
  const reasons = {floor: '低于下限', cost: '成本规则未过', guard: '领域约束未过', admissible: '合格'};
  function renderPool(input, cfg) {
    const selected = selectRound(input, pool, cfg);
    button('add-candidate').disabled = false;
    button('apply-round').disabled = pool.length === 0;
    button('clear-pool').disabled = pool.length === 0;
    doc.getElementById('pool-content').hidden = pool.length === 0;
    const body = doc.getElementById('pool-rows');
    body.replaceChildren();
    pool.forEach((candidate, index) => {
      const decision = selected.decisions[index], winner = selected.winnerIndex === index;
      const row = doc.createElement('tr');
      if (winner) row.classList.add('winner');
      const values = [`候选 ${candidate.id}`, fixed(candidate.score), fixed(candidate.cost),
        String(decision.novelty), reasons[decision.reason], winner ? '本轮赢家' : ''];
      values.forEach(value => {const cell = doc.createElement('td'); cell.textContent = value; row.append(cell);});
      const cell = doc.createElement('td'), remove = doc.createElement('button');
      remove.type = 'button'; remove.textContent = '删除';
      remove.setAttribute('aria-label', `删除候选 ${candidate.id}`);
      remove.addEventListener('click', () => {
        pool.splice(index, 1); render(); button('add-candidate').focus();
      });
      cell.append(remove); row.append(cell); body.append(row);
    });
    const winner = selected.winnerIndex === null ? null : pool[selected.winnerIndex];
    text('pool-summary', !pool.length ? '候选池为空：先加入一个或多个候选。' : winner
      ? `${pool.length} 个候选；赢家是候选 ${winner.id}，应用后 S=${fixed(winner.score)}，S*=${fixed(selected.next.best)}。`
      : `${pool.length} 个候选均不合格；应用本轮会保留当前版本。`);
    updateScrollFocus();
  }
  function renderHistory() {
    const history = doc.getElementById('round-history');
    const open = new Set([...history.querySelectorAll('details[open]')].map(d => d.dataset.round));
    history.replaceChildren();
    doc.getElementById('history-empty').hidden = rounds.length > 0;
    rounds.forEach((round, index) => {
      const item = doc.createElement('li'), details = doc.createElement('details');
      details.dataset.round = String(index);
      details.open = open.has(String(index)) || index === rounds.length - 1;
      const summary = doc.createElement('summary');
      const winner = round.winnerIndex === null ? null : round.candidates[round.winnerIndex];
      summary.textContent = `第 ${index + 1} 轮（t=${round.t}）：${winner ? '采用候选 ' + winner.id : '保留当前版本'}；S ${fixed(round.base.score)} → ${fixed(round.next.score)}`;
      const state = doc.createElement('p');
      state.textContent = `S* ${fixed(round.base.best)} → ${fixed(round.next.best)}；成本 ${fixed(round.base.cost)} → ${fixed(round.next.cost)}；获胜路径结构类型：${round.next.accepted.join('、') || '无'}。`;
      const outcomes = doc.createElement('ul');
      round.candidates.forEach((candidate, i) => {
        const entry = doc.createElement('li'), decision = round.decisions[i];
        const outcome = i === round.winnerIndex ? 'ACCEPTED' : decision.admissible ? 'LOST' : 'REJECTED';
        entry.textContent = `候选 ${candidate.id}：${outcome}；${reasons[decision.reason]}，ν=${decision.novelty}，得分 ${fixed(candidate.score)}。`;
        outcomes.append(entry);
      });
      details.append(summary, state, outcomes); item.append(details); history.append(item);
    });
    const download = doc.getElementById('download-history');
    download.hidden = rounds.length === 0;
    if (rounds.length) {
      download.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify({
        kind: 'constructed_rrsi_selection_rounds',
        sourceCommit: 'be50316e1db05914068a973f322770ef08ed7ba1', rounds
      }, null, 2));
    } else download.removeAttribute('href');
  }
  function render() {
    const cfg = configs[field('domain').value];
    text('parameters', `β₀=${cfg.beta0}，β₁=${cfg.beta1}；wₛ=${cfg.ws}，w꜀=${cfg.wc}，wₙ=${cfg.wn}`);
    try {
      const input = readInput(), result = judge(input, cfg);
      doc.getElementById('details').hidden = false;
      text('status', result.admissible ? '合格：有资格进入候选池' : '拒绝：未通过筛选');
      doc.getElementById('status').dataset.state = result.admissible ? 'pass' : 'fail';
      text('deltas', `ΔS=${fixed(result.ds)}（${fixed(result.ds * 100)} 个分数点），ΔC=${percent(result.dc)}，ν=${result.novelty}`);
      const boundary = doc.getElementById('noise-boundary');
      boundary.hidden = !(result.floorOK && fixed(result.ds) === fixed(input.delta) && result.ds !== input.delta);
      text('noise-boundary', `ΔS 的未舍入值为 ${result.ds.toString()}；δ=${input.delta.toString()}。按原始数值比较，走${result.strong ? '明确提分' : '噪声带内'}分支。`);
      text('floor-check', `历史下限 ${fixed(result.floor)}；候选 ${fixed(number('candidate'))} ${result.floorOK ? '≥' : '<'} 下限 → ${result.floorOK ? '通过' : '拒绝'}`);
      text('cost-check', result.strong
        ? `ΔS > δ：检查增长上限 ${percent(result.allowance)}；实际 ${percent(result.dc)} → ${result.costOK ? '通过' : '拒绝'}`
        : `ΔS ≤ δ：计算 wₛΔS − w꜀ΔC + wₙν = ${fixed(result.shaped)}；要求严格大于 0 → ${result.costOK ? '通过' : '拒绝'}`);
      text('guard-check', `领域约束 → ${result.guardOK ? '通过' : '拒绝'}`);
      for (const [id, active] of [['floor-check', true], ['cost-check', result.floorOK],
        ['guard-check', result.floorOK && result.costOK]]) {
        const step = doc.getElementById(id);
        step.dataset.skipped = String(!active);
        if (!active) step.textContent = '未执行（供比较）：' + step.textContent;
      }
      text('interpretation', result.admissible
        ? '合格不等于本轮获胜；还要与其他合格候选比较得分，才确定下一版本。'
        : '实际筛选在第一项失败时结束。标注“未执行”的步骤只供比较，不会推翻更早的拒绝。');
      renderPool(input, cfg);
    } catch (error) {
      text('status', error.message);
      doc.getElementById('status').dataset.state = 'invalid';
      doc.getElementById('details').hidden = true;
      doc.getElementById('pool-content').hidden = true;
      button('add-candidate').disabled = true;
      button('apply-round').disabled = true;
      button('clear-pool').disabled = pool.length === 0;
      text('pool-summary', '请先修正上方输入；候选池暂不判定。');
    }
  }
  const examples = {
    A: [0.835, 1800000, false, true], B: [0.819, 1260000, false, true],
    C: [0.810, 950000, false, true], D: [0.800, 700000, false, true],
    E: [0.809, 1000000, true, true], F: [0.850, 2000000, false, false]
  };
  doc.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
    const [candidate, cost, novelty, guard] = examples[button.dataset.example];
    field('candidate').value = candidate; field('candidateCost').value = cost;
    field('guardOK').checked = guard;
    form.querySelectorAll('input[name="touched"]').forEach(e => {
      e.checked = novelty && e.value === 'skill';
    });
    render();
  }));
  button('add-candidate').addEventListener('click', () => {
    const input = readInput();
    judge(input, configs[field('domain').value]);
    pool.push({id: ++candidateSequence, score: input.candidate, cost: input.candidateCost,
      touched: [...input.touched], guardOK: input.guardOK});
    render();
  });
  button('apply-round').addEventListener('click', () => {
    if (!pool.length) return;
    const input = readInput(), cfg = configs[field('domain').value];
    const selected = selectRound(input, pool, cfg);
    rounds.push({t: rounds.length, domain: field('domain').value, config: {...cfg, delta: input.delta},
      base: {score: input.score, best: input.best, cost: input.cost,
        delta: input.delta, accepted: [...input.accepted]},
      candidates: pool.map(c => ({...c, touched: [...c.touched]})), ...selected});
    field('score').value = selected.next.score; field('cost').value = selected.next.cost;
    field('best').value = selected.next.best;
    form.querySelectorAll('input[name="accepted"]').forEach(e => {
      e.checked = selected.next.accepted.includes(e.value);
    });
    pool.length = 0;
    text('round-status', `已应用第 ${rounds.length} 轮。当前候选区现在按新路径重算；上一轮结论见记录。`);
    renderHistory(); render(); button('add-candidate').focus();
  });
  button('clear-pool').addEventListener('click', () => {
    pool.length = 0; render(); button('add-candidate').focus();
  });
  button('reset-lab').disabled = false;
  function resetExperiment(message, preset = {}) {
    form.reset(); field('domain').value = 'coding';
    for (const [name, value] of Object.entries(preset)) field(name).value = value;
    pool.length = 0; rounds.length = 0; candidateSequence = 0;
    text('round-status', message);
    renderHistory(); render();
  }
  button('reset-lab').addEventListener('click', () => {
    resetExperiment('实验已重置为文章的 coding 起点。');
  });
  doc.querySelectorAll('[data-boundary]').forEach(button => {
    button.disabled = false;
    button.addEventListener('click', () => {
      const candidate = button.dataset.boundary === 'within' ? 0.830 : 0.831;
      resetExperiment(`实验已重置为工程边界算例，候选得分 ${candidate.toFixed(3)}。`, {
        domain: 'engineering', delta: configs.engineering.delta,
        candidate, candidateCost: 2000000
      });
    });
  });
  function newManualStart() {
    if (rounds.length) {
      rounds.length = 0; renderHistory();
      text('round-status', '当前路径被手动修改，已清空旧轮次记录；候选池按新起点重算。');
    }
  }
  field('domain').addEventListener('change', () => {
    field('delta').value = configs[field('domain').value].delta;
    newManualStart(); render();
  });
  form.addEventListener('input', event => {
    if (event.target.name === 'domain') return;
    if (['score', 'best', 'cost', 'delta', 'accepted'].includes(event.target.name)) newManualStart();
    render();
  });
  form.addEventListener('submit', event => event.preventDefault());
  root.addEventListener('resize', updateScrollFocus);
  renderHistory(); render();
})(typeof globalThis !== 'undefined' ? globalThis : this);
