// Run with: node --test server/web_ui/performance_test.cjs
// Set GOPHERLLM_UI_BENCHMARK=1 to print local comparative timings.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { performance } = require('node:perf_hooks');
const source = fs.readFileSync(__dirname + '/script.js', 'utf8');
const context = vm.createContext({});
vm.runInContext(source.slice(source.indexOf('/* Untrimmed core:'), source.indexOf('/* IndexedDB')), context);
vm.runInContext(source.slice(source.indexOf('function createChatFilter('), source.indexOf('// Search only controls')), context);
const { splitThinkText, createThinkSplitter, createChatFilter, reconcileChildren } = context;
const plain = value => JSON.parse(JSON.stringify(value));

test('incremental reasoning matches the full parser at every token boundary', () => {
  const examples = [
    '', 'Plain answer with <html> and 🐹.',
    '  <think>Long reasoning\nwith <ordinary> tags.</think>\n Answer  ',
    'Reasoning without an opening tag.</think>Final answer.',
    'Start<think>first</think>middle<think>second</think>end',
    '<think></think><think> </think> tail',
    '<think>Nested <think> stays literal.</think> Answer </think> stays literal.',
    '</think>Answer<think>more</think>Done',
    'Literal <thi and </thi plus <<think>reasoning</think> answer',
    '\uFEFF\t<think> \n</think> \t<think>\u00a0thought\uFEFF</think>\nA \n<think>more </think> \tB\uFEFF',
    'Prefix <think>reasoning ends in a partial </thin',
    'Prefix ending in a partial <think',
  ];
  for (const text of examples) {
    const snapshot = createThinkSplitter();
    const deltas = createThinkSplitter();
    for (let length = 0; length <= text.length; length++) {
      const expected = plain(splitThinkText(text.slice(0, length)));
      assert.deepEqual(plain(snapshot(text.slice(0, length))), expected, `snapshot prefix ${length}: ${text}`);
      assert.deepEqual(plain(deltas.append(length ? text[length - 1] : '')), expected, `delta prefix ${length}: ${text}`);
    }
    for (let boundary = 0; boundary <= text.length; boundary++) {
      const split = createThinkSplitter();
      split.append(text.slice(0, boundary));
      assert.deepEqual(plain(split.append(text.slice(boundary))), plain(splitThinkText(text)), `packet boundary ${boundary}: ${text}`);
    }
  }
});

test('snapshot corrections reset reasoning state and remain compatible with appended deltas', () => {
  const split = createThinkSplitter();
  for (const text of ['<think>old', '<think>old</think>answer', 'replacement', '', 'prefix</think>', 'prefix</think>new']) {
    assert.deepEqual(plain(split(text)), plain(splitThinkText(text)));
  }
  assert.deepEqual(plain(split.append('<think>next</think>done')), plain(splitThinkText('prefix</think>new<think>next</think>done')));
  assert.deepEqual(plain(split('corrected <think>unfinished')), plain(splitThinkText('corrected <think>unfinished')));
});

test('reasoning parser handles mixed tag and text chunks', () => {
  let seed = 42;
  const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed; };
  const fragments = ['<think>', '</think>', '<thi', '</thi', 'nk>', '<', '>', ' words ', '\n', '🐹'];
  for (let run = 0; run < 60; run++) {
    const split = createThinkSplitter();
    let text = '';
    for (let token = 0; token < 40; token++) {
      const chunk = fragments[random() % fragments.length];
      text += chunk;
      assert.deepEqual(plain(split.append(chunk)), plain(splitThinkText(text)));
    }
  }
});

const chat = (id, content, extra = {}) => ({
  id, title: id, systemPrompt: '', pinned: false, updatedAt: 0,
  messages: content.map(text => ({ content: text })), ...extra
});
const ids = rows => Array.from(rows, item => item.id);

test('chat search preserves recent-message scope, pinned order, and input order', () => {
  const filter = createChatFilter();
  const chats = [
    chat('recent', ['needle'], { updatedAt: 20 }),
    chat('old', ['needle', ...Array(8).fill('other')], { updatedAt: 40 }),
    chat('instructions', [], { systemPrompt: 'NEEDLE instructions', updatedAt: 10, pinned: true }),
    chat('needle title', [], { updatedAt: 30 }),
  ];
  assert.deepEqual(ids(filter(chats, '  Needle  ')), ['instructions', 'needle title', 'recent']);
  assert.deepEqual(ids(filter(chats, '')), ['instructions', 'old', 'needle title', 'recent']);
  assert.deepEqual(ids(chats), ['recent', 'old', 'instructions', 'needle title']);
});

test('chat search invalidates changed titles, instructions, and message content', () => {
  const filter = createChatFilter();
  const item = chat('old title', ['old message']);
  const chats = [item];
  assert.equal(filter(chats, 'old').length, 1);
  item.title = 'renamed';
  item.messages[0].content = 'replacement';
  assert.equal(filter(chats, 'old').length, 0);
  assert.equal(filter(chats, 'replacement').length, 1);
  item.systemPrompt = 'new instructions';
  assert.equal(filter(chats, 'instructions').length, 1);
  item.messages.push(...Array.from({ length: 8 }, () => ({ content: 'newest' })));
  assert.equal(filter(chats, 'replacement').length, 0);
  assert.equal(filter(chats, 'newest').length, 1);
  item.pinned = true;
  item.updatedAt = 100;
  assert.deepEqual(ids(filter([chat('another', ['newest']), item], 'newest')), ['old title', 'another']);
});

test('repeated searches case-fold only the query until searchable content changes', () => {
  vm.runInContext(`
    globalThis.foldedCharacters = 0;
    const originalLowerCase = String.prototype.toLowerCase;
    String.prototype.toLowerCase = function () {
      foldedCharacters += this.length;
      return originalLowerCase.call(this);
    };
  `, context);
  const filter = createChatFilter();
  const chats = [chat('large', ['A'.repeat(100000)])];
  filter(chats, 'a');
  const warmed = context.foldedCharacters;
  for (const query of ['aa', 'aaa', 'aaaa']) filter(chats, query);
  assert.equal(context.foldedCharacters - warmed, 9);
  chats[0].messages[0].content = 'B'.repeat(100000);
  assert.equal(filter(chats, 'b').length, 1);
  assert.ok(context.foldedCharacters - warmed > 100000);
});

// A small DOM model exercises real insertion/removal semantics rather than
// testing a particular diff layout or requiring a browser package in CI.
function listDOM(names) {
  const parent = {
    children: [], mutations: 0,
    get firstChild() { return this.children[0] || null; },
    insertBefore(child, before) {
      const existing = this.children.indexOf(child);
      if (existing >= 0) this.children.splice(existing, 1);
      const index = before ? this.children.indexOf(before) : this.children.length;
      assert.ok(index >= 0, 'insertion anchor must be an existing child');
      this.children.splice(index, 0, child);
      this.mutations++;
    },
    removeChild(child) {
      const index = this.children.indexOf(child);
      assert.ok(index >= 0, 'removed node must be an existing child');
      this.children.splice(index, 1);
      this.mutations++;
    }
  };
  const nodes = Object.fromEntries(names.map(name => [name, {
    name,
    get nextSibling() { return parent.children[parent.children.indexOf(this) + 1] || null; }
  }]));
  parent.children = Object.values(nodes);
  return { parent, nodes };
}

test('chat list reconciliation performs no DOM mutations for unchanged rows', () => {
  const { parent, nodes } = listDOM(['a', 'b', 'c']);
  reconcileChildren(parent, [nodes.a, nodes.b, nodes.c]);
  assert.equal(parent.mutations, 0);
  reconcileChildren(parent, [nodes.c, nodes.a]);
  assert.deepEqual(parent.children.map(node => node.name), ['c', 'a']);
  assert.equal(parent.mutations, 2);
  const mutations = parent.mutations;
  reconcileChildren(parent, [nodes.c, nodes.a]);
  assert.equal(parent.mutations, mutations);
  reconcileChildren(parent, []);
  assert.deepEqual(parent.children, []);
});

test('chat list reconciliation reuses filtered rows when search clears', () => {
  const { parent, nodes } = listDOM(['a', 'b', 'c']);
  reconcileChildren(parent, [nodes.b]);
  assert.equal(parent.children[0], nodes.b);
  reconcileChildren(parent, [nodes.a, nodes.b, nodes.c]);
  assert.deepEqual(parent.children, [nodes.a, nodes.b, nodes.c]);
});

test('optional local benchmark compares long reasoning and repeated chat searches', { skip: !process.env.GOPHERLLM_UI_BENCHMARK }, t => {
  const elapsed = work => {
    const start = performance.now();
    work();
    return performance.now() - start;
  };
  const deltas = ['<think>', ...Array(6000).fill('reasoning word '), '</think>', 'answer'];
  const fullMS = elapsed(() => {
    let text = '';
    for (const chunk of deltas) { text += chunk; splitThinkText(text); }
  });
  const incrementalMS = elapsed(() => {
    const split = createThinkSplitter();
    for (const chunk of deltas) split.append(chunk);
  });
  const chats = Array.from({ length: 200 }, (_, i) => chat('chat ' + i,
    Array.from({ length: 8 }, () => 'Searchable content '.repeat(500)), { updatedAt: i }));
  const queries = ['search', 'content', 'nothing', 'chat', 'searchable'];
  const filter = createChatFilter();
  filter(chats, 'warmup');
  const uncachedMS = elapsed(() => {
    for (let repeat = 0; repeat < 10; repeat++) for (const query of queries) {
      chats.slice().sort((a, b) => b.updatedAt - a.updatedAt).filter(item =>
        (item.title + '\n' + item.systemPrompt + '\n' + item.messages.slice(-8).map(message => message.content).join('\n')).toLowerCase().includes(query));
    }
  });
  const cachedMS = elapsed(() => {
    for (let repeat = 0; repeat < 10; repeat++) for (const query of queries) filter(chats, query);
  });
  t.diagnostic(`84 kB reasoning: full parse ${fullMS.toFixed(1)} ms; incremental ${incrementalMS.toFixed(1)} ms (${(fullMS / incrementalMS).toFixed(1)}×).`);
  t.diagnostic(`200 chats, 50 searches: uncached ${uncachedMS.toFixed(1)} ms; cached ${cachedMS.toFixed(1)} ms (${(uncachedMS / cachedMS).toFixed(1)}×).`);
});
