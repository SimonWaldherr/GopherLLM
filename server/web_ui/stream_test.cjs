// Run with: node --test server/web_ui/stream_test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/script.js', 'utf8');
const context = vm.createContext({ TextDecoder, document: {
  createTextNode(data) { return { data, appendData(text) { this.data += text; } }; }
}});
vm.runInContext(source.slice(0, source.indexOf('const TICK')), context);
const { readSSE, createStreamTextRenderer } = context;

test('SSE decodes UTF-8 and CRLF across every byte boundary', async () => {
  const bytes = new TextEncoder().encode(': heartbeat\r\n\r\ndata: {"text":"Grüße 🐹"}\r\n\r\ndata: [DONE]\r\n\r\n');
  for (let boundary = 1; boundary < bytes.length; boundary++) {
    const events = [];
    const body = new ReadableStream({ start(c) { c.enqueue(bytes.slice(0,boundary)); c.enqueue(bytes.slice(boundary)); c.close(); } });
    await readSSE({body}, value => events.push(JSON.parse(value)));
    assert.deepEqual(events, [{text:'Grüße 🐹'}]);
    assert.equal(body.locked, false);
  }
});

test('SSE handles multiline data and trailing events at EOF', async () => {
  const events = [];
  const body = new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode('data: first\ndata: second\n\ndata: final')); c.close(); } });
  await readSSE({body}, value => events.push(value));
  assert.deepEqual(events, ['first\nsecond','final']);
});

test('DONE completes without waiting for server EOF', async () => {
  let cancelled = false;
  const body = new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode('data: [DONE]\n\n')); }, cancel() { cancelled = true; } });
  await readSSE({body}, () => assert.fail('DONE is not data'));
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
});

test('malformed events cancel and release the response', async () => {
  let cancelled = false;
  const body = new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode('data: invalid\n\n')); }, cancel() { cancelled = true; } });
  await assert.rejects(readSSE({body}, JSON.parse));
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
});

test('stream renderer retains text node and handles corrected output', () => {
  let replacements = 0;
  const content = { replaceChildren(node) { this.node = node; replacements++; } };
  const paint = createStreamTextRenderer(content);
  const node = content.node;
  for (const text of ['Hello','Hello world','Hello world','Correction','']) {
    paint(text);
    assert.equal(content.node, node);
    assert.equal(node.data, text);
  }
  assert.equal(replacements,1);
});
