const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(__dirname+'/script.js','utf8');
const context=vm.createContext({});
vm.runInContext(source.slice(source.indexOf('function boundedNumber('),source.indexOf('function cleanAttachment(')),context);
const defaults={maxTokens:128,temperature:0.2,topP:0.9,topK:40,minP:0,repeatPenalty:1.1};
const plain=value=>JSON.parse(JSON.stringify(value));

test('conversation recall is a separate opt-in and full context clears it',()=>{
  assert.equal(context.cleanSettings({},defaults).contextRAG,false);
  assert.equal(context.cleanSettings({ragMode:true,documentSearch:true},defaults).contextRAG,false);
  assert.equal(context.cleanSettings({contextRAG:true,contextWindowMode:'recent'},defaults).contextRAG,true);
  assert.equal(context.cleanSettings({contextRAG:true,contextWindowMode:'autoCompress'},defaults).contextRAG,true);
  assert.equal(context.cleanSettings({contextRAG:true,contextWindowMode:'full'},defaults).contextRAG,false);
});

test('requests enable recall only for a bounded local server context',()=>{
  const settings={contextRAG:true,contextWindowMode:'recent'};
  assert.equal(context.contextRAGEnabled(settings,true,'server'),true);
  assert.equal(context.contextRAGEnabled(settings,false,'server'),false);
  assert.equal(context.contextRAGEnabled(settings,true,'browser'),false);
  assert.equal(context.contextRAGEnabled({...settings,contextWindowMode:'full'},true,'server'),false);
  assert.equal(context.contextRAGEnabled({...settings,contextWindowMode:'invalid'},true,'server'),false);
});

test('ordinary and full-context sends do not wait for a health request',()=>{
  assert.equal(context.contextRAGRequested({contextWindowMode:'recent'}),false);
  assert.equal(context.contextRAGRequested({contextRAG:false,contextWindowMode:'recent'}),false);
  assert.equal(context.contextRAGRequested({contextRAG:true,contextWindowMode:'full'}),false);
  assert.equal(context.contextRAGRequested({contextRAG:true,contextWindowMode:'recent'}),true);
  assert.match(source,/if \(contextRAGRequested\(chat.settings\)\) await refreshContextRAGAvailability\(controller.signal\);/);
});

test('stopping an opted-in send aborts its pending health request',async()=>{
  const controller=new AbortController();
  let observedSignal;
  const pendingContext=vm.createContext({
    contextRAGCheck:0,contextRAGAvailable:false,browserOnlyDeployment:false,preferences:{inferenceMode:'server'},
    syncContextRAGControls(){},updateContextWindowStatus(){},
    fetch(_url,options){
      observedSignal=options.signal;
      return new Promise((_resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('Stopped','AbortError')),{once:true}));
    }
  });
  vm.runInContext(source.slice(source.indexOf('  async function refreshContextRAGAvailability('),source.indexOf('  function updateContextWindowStatus(')),pendingContext);
  const pending=pendingContext.refreshContextRAGAvailability(controller.signal);
  assert.equal(observedSignal,controller.signal);
  controller.abort();
  await assert.rejects(pending,{name:'AbortError'});
});

test('recall annotations accept bounded counters and ignore source text',()=>{
  const value={mode:'bm25',indexed_turns:3,token_budget:128,added_tokens:52,sources:[{id:'history-1',content:'untrusted'}]};
  assert.deepEqual(plain(context.cleanContextRecall(value)),{indexedTurns:3,tokenBudget:128,addedTokens:52,turns:1});
  assert.equal(context.cleanContextRecall({...value,mode:'vector'}),null);
  assert.equal(context.cleanContextRecall({...value,added_tokens:129}),null);
  assert.equal(context.cleanContextRecall({...value,indexed_turns:0}),null);
  assert.equal(context.cleanContextRecall({...value,added_tokens:NaN}),null);
});

test('agent-round telemetry follows the local branch message count',()=>{
  vm.runInContext(source.slice(source.indexOf('  function currentContextWindow('),source.indexOf('  function compactCount(')),context);
  vm.runInContext(source.slice(source.indexOf('  function contextWindowFromValue('),source.indexOf('  function updateComposer(')),context);
  const chat={id:'branch',messages:[{},{},{},{}]};
  const info={mode:'recent',context_length:1000,prompt_budget:800,prompt_tokens:750,input_messages:10,retained_messages:7,dropped_messages:3,compressed_messages:0,
    recall:{mode:'bm25',indexed_turns:3,token_budget:100,added_tokens:80,sources:[{id:'history-1'}]}};
  context.lastContextWindow=context.contextWindowFromValue(info,chat);
  assert.equal(context.currentContextWindow(chat).recall.turns,1);
  assert.equal(context.currentContextWindow({...chat,id:'other-branch'}),null);
  chat.messages.push({});
  assert.equal(context.currentContextWindow(chat),null);
});

test('the UI offers a disabled-by-default checkbox without an embedding dependency',()=>{
  const html=fs.readFileSync(__dirname+'/chat.html','utf8');
  assert.match(html,/<input id="contextRAG" type="checkbox" disabled>/);
  assert.match(source,/gopherllm_context_rag: contextRAGEnabled\(chat.settings, contextRAGAvailable, preferences.inferenceMode\)/);
});
