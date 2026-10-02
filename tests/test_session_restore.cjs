// Run the page's real script with in-memory browser APIs; no packages or network needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

const script = fs.readFileSync('static/index.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const id = 'ras_0123456789ab';
const saved = {
  id, resume_text: '真实简历材料', jd_text: '原始岗位要求', state: 'awaiting_user_choice',
  analysis: { match_score: 67, summary: '保存的分析', score_breakdown: {} },
  initial_analysis: { match_score: 50 }, facts: [{ content: '已补充经历' }],
  pending_questions: [], review_items: [{ requirement: 'Python', requirement_id: 'req_python', category: 'skill' }],
  proposals: [{ id: 'proposal_saved', requirement: 'Python', category: 'skill', version: 2,
    status: 'accepted', before: '原文', after: '保存的已采纳表达' }], messages: [],
};

async function page({ query = '', stored = null, status = 200, failure = false, storageBlocked = false } = {}) {
  const elements = new Map();
  const classes = new Set();
  const data = new Map(stored ? [['jobfit.active-session', stored]] : []);
  const calls = [];
  const element = () => ({ value: '', textContent: '', innerHTML: '', className: '', disabled: false,
    handlers: {}, addEventListener(name, callback) { this.handlers[name] = callback; } });
  for (const selector of ['#form', '#submit', '#result', '#intake-shell', '#intake-notice', '#resume-text', '#resume', '#jd']) {
    elements.set(selector, element());
  }
  elements.get('#resume').files = [];
  const location = { href: 'http://localhost:9000/' + query };
  const context = vm.createContext({
    URL, FormData, console,
    document: { querySelector(selector) { return elements.get(selector) || null; },
      body: { classList: { add(name) { classes.add(name); }, remove(name) { classes.delete(name); } } } },
    window: { location, history: { replaceState(_, __, url) { location.href = String(url); } }, scrollTo() {} },
    sessionStorage: {
      getItem(key) { if (storageBlocked) throw Error('blocked'); return data.get(key) || null; },
      setItem(key, value) { if (storageBlocked) throw Error('blocked'); data.set(key, value); },
      removeItem(key) { if (storageBlocked) throw Error('blocked'); data.delete(key); },
    },
    async fetch(url) {
      calls.push(url);
      if (failure) throw Error('网络暂时不可用');
      return { ok: status === 200, status,
        async json() { return status === 200 ? saved : { detail: 'not found' }; } };
    },
  });
  vm.runInContext(script, context);
  await new Promise(setImmediate);
  return { context, elements, classes, data, calls, location };
}

test('refresh restores saved report, material, decision and version from URL', async () => {
  const state = await page({ query: '?session=' + id });
  assert.deepEqual(state.calls, ['/resume-agent/sessions/' + id]);
  assert.equal(state.elements.get('#resume-text').value, saved.resume_text);
  assert.equal(state.elements.get('#jd').value, saved.jd_text);
  assert(state.elements.get('#result').innerHTML.includes('保存的已采纳表达'));
  assert(state.elements.get('#result').innerHTML.includes('第 2 版 · 已采纳'));
  assert(state.classes.has('session-active'));
  assert.equal(state.data.get('jobfit.active-session'), id);
});

test('URL selects the correct session over a different tab reference', async () => {
  const state = await page({ query: '?session=' + id, stored: 'ras_aaaaaaaaaaaa' });
  assert.deepEqual(state.calls, ['/resume-agent/sessions/' + id]);
});

test('tab storage restores an active session without query parameters', async () => {
  const state = await page({ stored: id });
  assert(state.location.href.includes('session=' + id));
  assert(state.classes.has('session-active'));
});

test('404 removes stale references and allows creating a session', async () => {
  const state = await page({ query: '?session=' + id, stored: id, status: 404 });
  assert(!state.location.href.includes('session='));
  assert(!state.data.has('jobfit.active-session'));
  assert(state.elements.get('#intake-notice').textContent.includes('会话已不存在'));
  assert.equal(state.elements.get('#submit').disabled, false);
});

test('temporary failure preserves reference for a later retry', async () => {
  const state = await page({ query: '?session=' + id, failure: true });
  assert(state.location.href.includes(id));
  assert(state.elements.get('#intake-notice').textContent.includes('网络暂时不可用'));
});

test('invalid session query never triggers a request', async () => {
  const state = await page({ query: '?session=https://other.example' });
  assert.equal(state.calls.length, 0);
  assert(!state.location.href.includes('session='));
});

test('URL restoration still works when storage is blocked', async () => {
  const state = await page({ query: '?session=' + id, storageBlocked: true });
  assert(state.classes.has('session-active'));
});

test('starting a new session clears the old restoration reference', async () => {
  const state = await page({ query: '?session=' + id, stored: id });
  vm.runInContext('resetWorkspace()', state.context);
  assert(!state.location.href.includes('session='));
  assert(!state.data.has('jobfit.active-session'));
  assert(!state.classes.has('session-active'));
});

test('successful creation remembers only the session id and supports refresh', async () => {
  const state = await page();
  state.elements.get('#resume-text').value = '有足够长度的简历文本，用于创建真实会话';
  state.elements.get('#jd').value = '目标岗位';
  await state.elements.get('#form').handlers.submit({ preventDefault() {} });
  assert(state.location.href.includes('session=' + id));
  assert.deepEqual([...state.data.entries()], [['jobfit.active-session', id]]);
});
