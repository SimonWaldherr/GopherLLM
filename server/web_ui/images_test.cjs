const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function setup(fetch) {
  const elements = new Map();
  const el = id => { if (!elements.has(id)) elements.set(id, {value:'',checked:false,hidden:true,events:{},addEventListener(k,f){this.events[k]=f},setAttribute(k,v){this[k]=v},reportValidity(){return true}}); return elements.get(id); };
  el('prompt').value='A teapot'; el('size').value='256x256'; el('seed').value='42';
  vm.runInNewContext(fs.readFileSync(__dirname+'/images.js','utf8'), {document:{getElementById:el},window:{addEventListener(){}},AbortController,performance,fetch});
  return {el,submit:()=>el('imageForm').events.submit({preventDefault(){}})};
}
test('generation displays and downloads the returned PNG',async()=>{
  const png='iVBORw0KGgoAAA';
  const ui=setup(async(url,opts)=>{assert.equal(url,'/v1/images/generations');assert.deepEqual(JSON.parse(opts.body),{prompt:'A teapot',size:'256x256',seed:42,keep_alive:false});return {ok:true,json:async()=>({data:[{b64_json:png}]})};});
  await ui.submit();assert.equal(ui.el('result').src,'data:image/png;base64,'+png);assert.equal(ui.el('download').href,ui.el('result').src);assert.equal(ui.el('result').hidden,false);assert.equal(ui.el('generate').disabled,false);assert.match(ui.el('imageStatus').textContent,/Image ready/);
});
test('cancel aborts fetch and prevents overlapping submissions',async()=>{
  let calls=0;
  const ui=setup((url,{signal})=>{calls++;return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(Object.assign(new Error('cancelled'),{name:'AbortError'}))));});
  const pending=ui.submit();await ui.submit();assert.equal(calls,1);ui.el('cancel').events.click();await pending;
  assert.match(ui.el('imageStatus').textContent,/Cancelled/);assert.equal(ui.el('generate').disabled,false);assert.equal(ui.el('cancel').hidden,true);
});
test('API errors restore controls and preserve the preceding image',async()=>{
  const ui=setup(async()=>({ok:false,json:async()=>({error:{message:'request capacity exceeded; retry later'}})}));ui.el('result').src='previous';
  await ui.submit();assert.match(ui.el('imageStatus').textContent,/capacity/);assert.equal(ui.el('result').src,'previous');assert.equal(ui.el('generate').disabled,false);
});
test('retention checkbox is sent and unload preserves the generated image',async()=>{
  const calls=[];
  const ui=setup(async(url,opts)=>{calls.push({url,opts});return {ok:true,json:async()=>url.endsWith('/unload')?{unloaded:true}:{data:[{b64_json:'iVBORw0KGgoAAA'}]}}});
  ui.el('keepAlive').checked=true;
  await ui.submit();assert.equal(JSON.parse(calls[0].opts.body).keep_alive,true);assert.match(ui.el('imageStatus').textContent,/kept loaded/);
  const previous=ui.el('result').src;
  await ui.el('unload').events.click();assert.equal(calls[1].url,'/v1/images/unload');assert.equal(ui.el('keepAlive').checked,false);assert.equal(ui.el('result').src,previous);assert.match(ui.el('imageStatus').textContent,/Model unloaded/);
});
