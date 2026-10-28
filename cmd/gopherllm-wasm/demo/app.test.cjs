const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/app.js', 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));

async function harness(overrides = {}) {
  let created = 0, closed = 0, stopped = 0, layoutReads = 0;
  const frames = [];
  class Element {
    children = []; listeners = {}; value = ''; files = []; disabled = false;
    classList = {add(){}, remove(){}, toggle(){}};
    addEventListener(name, handler) { this.listeners[name] = handler; }
    appendChild(child) { this.children.push(child); return child; }
    get scrollHeight() { layoutReads++; return this.children.length * 20; }
    focus() {}
    click() { return this.listeners.click?.(); }
  }
  const elements = {};
  const nodes = [];
  const track = {stop(){stopped++;}};
  class AudioContext {
    sampleRate = 48000;
    destination = {};
    createMediaStreamSource() { return {connect(){},disconnect(){}}; }
    createGain() { return {gain:{},connect(){},disconnect(){}}; }
    createScriptProcessor() { const node = {connect(){},disconnect(){},onaudioprocess:null}; nodes.push(node); return node; }
    async close() { closed++; }
  }
  const context = vm.createContext({
    Float32Array, Promise, setTimeout, console,
    requestAnimationFrame(callback) { frames.push(callback); },
    document: {
      getElementById(id) { return elements[id] ||= new Element(); },
      createElement() { created++; return new Element(); },
      createTextNode(text) { created++; return {data:text,appendData(chunk){this.data+=chunk;}}; },
    },
    Go: class { importObject = {}; run() {} },
    WebAssembly: {instantiateStreaming: async()=>({instance:{}})},
    fetch: async()=>({ok:true}), AudioContext,
    navigator: {mediaDevices:{getUserMedia:async()=>({getTracks:()=>[track]})}},
    gopherllm_loadModelWithVision(){},
    gopherllm_webgpuStatus:async()=> 'unavailable',
    gopherllm_isGenerating:()=>false,
    gopherllm_startVoxtral:async()=>{},
    gopherllm_stopVoxtral:async()=>'',
    gopherllm_pushVoxtralPCM:async()=>'',
    ...overrides,
  });
  context.window = context;
  vm.runInContext(source.slice(0, source.lastIndexOf('main().catch')), context);
  await context.main();
  return {context,elements,nodes,
    frame() { const pending = frames.splice(0); pending.forEach(callback=>callback()); },
    get pendingFrames(){return frames.length;},get layoutReads(){return layoutReads;},
    get created(){return created;},get closed(){return closed;},get stopped(){return stopped;}};
}

test('streaming updates only the new text node and preserves preceding image rows', async()=>{
  let h, tokens = 0;
  h = await harness({gopherllm_generate: async(_,__,onToken)=>{
    const before = h.created;
    for(let i=0;i<512;i++) { onToken('x'); tokens++; }
    assert.equal(h.created,before,'streaming must not allocate DOM elements');
    return 'x'.repeat(512);
  }});
  const first = {role:'user',content:'prior message',imageDataURL:'data:image/png;base64,abc'};
  h.context.appendMessage(first);
  const prior = h.elements.messages.children[0];
  const image = prior.children[0];
  h.elements.promptInput.value = 'Hello';
  await h.elements.chatForm.listeners.submit({preventDefault(){}});
  assert.equal(tokens,512);
  assert.equal(h.elements.messages.children.length,3);
  assert.equal(h.elements.messages.children[0],prior);
  assert.equal(prior.children[0],image);
  assert.equal(h.elements.messages.children[2].children[0].data,'x'.repeat(512));
  assert.equal(h.elements.sendButton.disabled,false);
  assert.equal(h.layoutReads,0,'tokens must not force layout before the next frame');
  assert.equal(h.pendingFrames,1,'a token burst must share one scroll callback');
  h.frame();
  assert.equal(h.layoutReads,1);
  assert.equal(h.elements.messages.scrollTop,60);
  image.listeners.load();
  h.frame();
  assert.equal(h.layoutReads,2,'image decoding must update the final scroll position');
});

test('speech stop drains pending PCM and releases capture after a failed push',async()=>{
  let rejectChunk, finalized = 0;
  const h=await harness({
    gopherllm_pushVoxtralPCM:()=>new Promise((_,reject)=>{rejectChunk=reject;}),
    gopherllm_stopVoxtral:async()=>{finalized++;return '';},
  });
  await h.elements.voxtralRecordButton.click();
  const event={inputBuffer:{getChannelData:()=>new Float32Array(4096)}};
  h.nodes[0].onaudioprocess(event);
  h.nodes[0].onaudioprocess(event);
  const stopping=h.elements.voxtralRecordButton.click();
  rejectChunk(new Error('decoder failed'));
  await stopping;
  assert.equal(finalized,1);
  assert.equal(h.closed,1);
  assert.equal(h.stopped,1);
  assert.equal(h.nodes[0].onaudioprocess,null);
  assert.match(h.elements.voxtralStatus.textContent,/decoder failed/);
  assert.equal(h.elements.voxtralRecordButton.disabled,false);
  await h.elements.voxtralRecordButton.click();
  assert.equal(h.nodes.length,2,'failure must leave the next recording usable');
});

test('speech finalization failure still closes the audio context',async()=>{
  const h=await harness({gopherllm_stopVoxtral:async()=>{throw new Error('finalization failed');}});
  await h.elements.voxtralRecordButton.click();
  await h.elements.voxtralRecordButton.click();
  assert.equal(h.closed,1);
  assert.equal(h.stopped,1);
  assert.match(h.elements.voxtralStatus.textContent,/finalization failed/);
  assert.equal(h.elements.voxtralRecordButton.disabled,false);
});

test('speech permission rejection releases context and prevents duplicate starts',async()=>{
  let rejectPermission, finalized=0;
  const h=await harness({
    navigator:{mediaDevices:{getUserMedia:()=>new Promise((_,reject)=>{rejectPermission=reject;})}},
    gopherllm_stopVoxtral:async()=>{finalized++;},
  });
  const starting=h.elements.voxtralRecordButton.click();
  await settle();
  assert.equal(h.elements.voxtralRecordButton.disabled,true);
  rejectPermission(new Error('permission denied'));
  await starting;
  assert.equal(h.closed,1);
  assert.equal(finalized,1);
  assert.equal(h.elements.voxtralRecordButton.disabled,false);
});
