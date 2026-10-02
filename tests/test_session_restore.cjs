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

async function page({ query = '', stored = null, status = 200, failure = false, storageBlocked = false, responseData = saved, onFetch } = {}) {
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
    async fetch(url, options) {
      calls.push(url);
      if (onFetch) onFetch(url, options);
      if (failure) throw Error('网络暂时不可用');
      return { ok: status === 200, status,
        async json() { return status === 200 ? responseData : { detail: 'not found' }; } };
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

function featureSession(paused = false) {
  const payload = structuredClone(saved);
  payload.document = {revision: 2, fields: [{id: 'fld_a', text: '保存后的完整简历'}], outdated_requirements: [],
    versions: [{revision: 2, label: '应用改写', fields: [{text: '保存后的完整简历'}]}]};
  payload.preview = {token: 'a'.repeat(64), base_revision: 2, text: '预览文本',
    changes: [{operation: 'replace', field_id: 'fld_a', before: '旧文本', after: '<script>新文本</script>',
      source_ids: ['fact_verified'], evidence_basis: ['真实事实'], checks: ['出处一致']}]};
  payload.active_interview_id = 'ir_saved';
  payload.interviews = [{id: 'ir_saved', revision: 3, state: paused ? 'paused' : 'active',
    source_document_revision: 1, position: 0, plan: [{}], turns: [],
    current_question: {id: 'iq_current', kind: 'followup', text: '继续说明方案', focus: '验证方法'}}];
  return payload;
}

test('refresh renders document version, exact preview and saved followup safely', async () => {
  const state = await page({query: '?session=' + id, responseData: featureSession()});
  const html = state.elements.get('#result').innerHTML;
  assert(html.includes('简历版本 · 第 2 版'));
  assert(html.includes('保存后的完整简历'));
  assert(html.includes('确认应用此预览'));
  assert(html.includes('&lt;script&gt;新文本&lt;/script&gt;'));
  assert(html.includes('第 1/1 题 · 追问'));
  assert(html.includes('使用简历第 1 版'));
});

test('apply sends the token and revision belonging to the displayed preview', async () => {
  const requests = [];
  const state = await page({query: '?session=' + id, responseData: featureSession(),
    onFetch(url, options) { if (options) requests.push({url, ...options}); }});
  state.context.trigger = {dataset: {previewToken: 'a'.repeat(64), documentRevision: '2'}, textContent: '应用'};
  await vm.runInContext('documentAction(trigger, "document-apply")', state.context);
  assert.equal(requests[0].url, '/resume-agent/sessions/' + id + '/document/apply');
  assert.deepEqual(JSON.parse(requests[0].body), {token: 'a'.repeat(64), expected_revision: 2});
});

test('paused interview restores its controls and resume uses the current checkpoint', async () => {
  const requests = [];
  const state = await page({query: '?session=' + id, responseData: featureSession(true),
    onFetch(url, options) { if (options) requests.push({url, ...options}); }});
  const html = state.elements.get('#result').innerHTML;
  assert(html.includes('继续练习'));
  assert(!html.includes('id="interview-answer"'));
  state.context.trigger = {dataset: {}, textContent: '继续'};
  await vm.runInContext('interviewAction(trigger, "interview-resume")', state.context);
  assert.deepEqual(JSON.parse(requests[0].body), {run_id: 'ir_saved', expected_revision: 3, action: 'resume'});
});

test('interview answer and explicit skip bind to the current question and run', async () => {
  const requests = [];
  const state = await page({query: '?session=' + id, responseData: featureSession(),
    onFetch(url, options) { if (options) requests.push({url, ...options}); }});
  state.elements.set('#interview-answer', {value: '我的真实回答'});
  state.context.trigger = {dataset: {}, textContent: '回答'};
  await vm.runInContext('interviewAction(trigger, "interview-answer")', state.context);
  assert.deepEqual(JSON.parse(requests[0].body), {run_id: 'ir_saved', expected_revision: 3,
    question_id: 'iq_current', answer: '我的真实回答', skip: false});
  await vm.runInContext('interviewAction(trigger, "interview-skip")', state.context);
  assert.equal(JSON.parse(requests[1].body).skip, true);
});

test('feature conflicts show a visible notice beside the action and reenable the button', async () => {
  const state = await page();
  const notice = {textContent: '', className: ''};
  state.elements.set('#document-notice', notice);
  state.context.trigger = {dataset: {agentAction: 'document-apply'}, textContent: '应用', disabled: false};
  await vm.runInContext('withBusy(trigger, "处理中", async () => { throw Error("预览已失效"); })', state.context);
  assert.equal(notice.textContent, '预览已失效');
  assert(notice.className.includes('show danger'));
  assert.equal(state.context.trigger.disabled, false);
  assert.equal(state.context.trigger.textContent, '应用');
});
