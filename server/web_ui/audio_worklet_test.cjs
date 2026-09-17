const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const source=fs.readFileSync(__dirname+'/audio-worklet.js','utf8');

function capture(rate, block) {
  const messages=[];let Processor;
  vm.runInNewContext(source,{sampleRate:rate,ArrayBuffer,DataView,
    AudioWorkletProcessor:class {port={postMessage:data=>messages.push(data)};},
    registerProcessor:(_,value)=>{Processor=value;}});
  const processor=new Processor();
  const count=rate+Math.round(rate*.017);
  const samples=Float32Array.from({length:count},(_,i)=>Math.sin(i*.041)*.8);
  for(let i=0;i<count;i+=block)processor.process([[samples.subarray(i,i+block)]]);
  processor.port.onmessage({data:'stop'});
  assert.equal(messages.pop(),'stopped');
  assert.equal(processor.process([[samples]]),false);
  assert.ok(messages.slice(0,-1).every(buffer=>buffer.byteLength===6400));
  return Buffer.concat(messages.map(buffer=>Buffer.from(buffer)));
}
for(const rate of [16000,44100,48000])test(`worklet ${rate} Hz preserves sample count and phase across render quanta`,()=>{
  const want=capture(rate,128);
  assert.equal(want.byteLength,Math.floor((rate+Math.round(rate*.017))*16000/rate)*2);
  for(const size of [1,127,256,4096])assert.deepEqual(capture(rate,size),want);
});
