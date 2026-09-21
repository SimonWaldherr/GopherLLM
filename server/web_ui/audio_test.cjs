const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/audio.js', 'utf8');

function harness(fetch) {
  class Element {
    constructor() { this.value = ''; this.hidden = false; this.disabled = false; this.listeners = {}; this.options = []; this.classList = {toggle(){}}; this.files=[]; }
    addEventListener(name, fn) { this.listeners[name]=fn; }
    setAttribute(name,value) { this[name]=value; }
    replaceChildren() { this.options=[]; this.value=''; }
    appendChild(option) { this.options.push(option); if(!this.value) this.value=option.value; }
    click() { return this.listeners.click?.(); }
  }
  const elements = {};
  for(const id of ['audioToggle','audioPanel','audioModel','audioRefresh','audioUpload','audioFile','audioRecord','audioLive','audioCancel','audioStatus','audioResult','audioInsert','audioInputDevice']) elements[id]=new Element();
  elements.audioPanel.hidden = true;
  let closed=0, stopped=0, inserted='';
  const decoded={duration:1,length:16000,sampleRate:16000,numberOfChannels:1,getChannelData(){return new Float32Array(16000);}};
  class AudioContext { async decodeAudioData(){return decoded;} async close(){closed++;} }
  class OfflineAudioContext {
    createBuffer(_,length,rate){return {getChannelData(){return new Float32Array(length);}};}
    createBufferSource(){return {connect(){},start(){}};}
    async startRendering(){return decoded;}
  }
  const mockTrack=()=>({stop(){stopped++;},addEventListener(){},removeEventListener(){}});
  const media={getTracks(){return [mockTrack()];}};
  class MediaStream { constructor(tracks){this._tracks=tracks||[];} getTracks(){return this._tracks;} }
  const context=vm.createContext({Blob,FormData,AbortController,DataView,ArrayBuffer,Float32Array,setTimeout,clearTimeout,setInterval,clearInterval,
    AudioContext,OfflineAudioContext,MediaRecorder:class {},MediaStream, navigator:{mediaDevices:{getUserMedia:async()=>media}},
    document:{getElementById:id=>elements[id],createElement:()=>new Element()},addEventListener(){}});
  vm.runInContext(source,context);
  const controls=context.GopherLLMAudio.init({fetch:fetch || (async()=>({ok:true,json:async()=>({models:[{id:'voxtral',name:'Voxtral'}]})})),insert(text){inserted=text;return true;}});
  return {context,elements,controls,decoded,get closed(){return closed;},get stopped(){return stopped;},get inserted(){return inserted;}};
}
const settle=()=>new Promise(resolve=>setImmediate(resolve));

function liveHarness(fetch) {
  const h = harness(fetch);
  h.nodes = [];
  h.context.AudioContext = class {
    audioWorklet = {addModule: async()=>{}};
    async resume() {}
    async close() {}
    createMediaStreamSource() { return {connect(){},disconnect(){}}; }
  };
  h.context.AudioWorkletNode = class {
    port = {onmessage:null,postMessage:message=>{this.command=message;}};
    constructor() { h.nodes.push(this); }
    connect() {} disconnect() {}
    emit(data) { this.port.onmessage?.({data}); }
  };
  return h;
}

test('live stop drains an in-flight request and partial PCM before finalizing',async()=>{
  let resolveChunk; const requests=[];
  const h=liveHarness(async(url,opts)=>{
    if(url==='/models/audio')return {ok:true,json:async()=>({models:[{id:'v'}]})};
    if(url==='/v1/audio/realtime/sessions')return {ok:true,json:async()=>({id:'session'})};
    requests.push({url,opts});
    if(requests.length===1)return new Promise(resolve=>{resolveChunk=resolve;});
    return {ok:true,json:async()=>({text:'Hello world'})};
  });
  h.elements.audioToggle.click();await settle();await h.elements.audioLive.click();
  const node=h.nodes[0];node.emit(new ArrayBuffer(6400));
  h.elements.audioLive.click();assert.equal(node.command,'stop');
  node.emit(Uint8Array.from([1,2,3,4]).buffer);node.emit(Uint8Array.from([5,6]).buffer);node.emit('stopped');
  assert.equal(requests.length,1);
  resolveChunk({ok:true,json:async()=>({text:'Hello'})});await settle();
  assert.equal(requests[1].url,'/v1/audio/realtime/sessions/session?final=1');
  assert.deepEqual(Array.from(new Uint8Array(requests[1].opts.body)),[1,2,3,4,5,6]);
  assert.equal(h.elements.audioResult.value,'Hello world');
  assert.equal(h.stopped,1);assert.equal(h.elements.audioCancel.hidden,true);
});

test('live stop with an empty worklet buffer still flushes decoder delay',async()=>{
  const requests=[];
  const h=liveHarness(async(url,opts)=>{
    if(url==='/models/audio')return {ok:true,json:async()=>({models:[{id:'v'}]})};
    if(url==='/v1/audio/realtime/sessions')return {ok:true,json:async()=>({id:'empty'})};
    requests.push({url,opts});return {ok:true,json:async()=>({text:'Tail'})};
  });
  h.elements.audioToggle.click();await settle();await h.elements.audioLive.click();
  h.elements.audioLive.click();h.nodes[0].emit('stopped');await settle();
  assert.match(requests[0].url,/final=1$/);assert.equal(requests[0].opts.body.byteLength,0);
  assert.equal(h.elements.audioResult.value,'Tail');
});

test('live cancellation aborts pending audio and ignores late transcripts',async()=>{
  let resolveChunk, signal;
  const h=liveHarness(async(url,opts)=>{
    if(url==='/models/audio')return {ok:true,json:async()=>({models:[{id:'v'}]})};
    if(url==='/v1/audio/realtime/sessions')return {ok:true,json:async()=>({id:'cancel'})};
    if(opts.method==='DELETE')return {ok:true};
    signal=opts.signal;return new Promise(resolve=>{resolveChunk=resolve;});
  });
  h.elements.audioToggle.click();await settle();await h.elements.audioLive.click();
  h.nodes[0].emit(new ArrayBuffer(6400));h.elements.audioCancel.click();
  assert.equal(signal.aborted,true);assert.equal(h.stopped,1);
  resolveChunk({ok:true,json:async()=>({text:'late'})});await settle();
  assert.equal(h.elements.audioResult.value,'');assert.equal(h.elements.audioStatus.textContent,'Cancelled.');
});

test('WAV conversion preserves signed PCM and clamps peaks', async()=>{
  const h=harness();const wav=h.context.GopherLLMAudio.encodeWAV(new Float32Array([-2,-.5,0,.5,2]));const view=new DataView(await wav.arrayBuffer());
  assert.equal(view.getUint32(24,true),16000);assert.equal(view.getUint16(22,true),1);
  assert.deepEqual(Array.from({length:5},(_,i)=>view.getInt16(44+i*2,true)),[-32768,-16384,0,16384,32767]);
  assert.throws(()=>h.context.GopherLLMAudio.encodeWAV([NaN]),/invalid samples/);
});

test('long uploads are rejected and decoding contexts always close',async()=>{
  const h=harness();h.decoded.duration=31;
  await assert.rejects(()=>h.context.GopherLLMAudio.toWAV(new Blob(['audio']),false),/30 seconds/);
  assert.equal(h.closed,1);
  await assert.rejects(()=>h.context.GopherLLMAudio.toWAV(new Blob([]),false),/empty/);
});

test('upload uses chosen model and inserts only after review',async()=>{
  let upload;
  const h=harness(async(url,opts)=>{
    if(url==='/models/audio')return {ok:true,json:async()=>({models:[{id:'chosen',name:'Chosen'}]})};
    upload=opts;return {ok:true,json:async()=>({text:' Hallo Welt '})};
  });
  await h.elements.audioToggle.click();await settle();
  assert.equal(h.elements.audioModel.value,'chosen');
  h.elements.audioFile.files=[new Blob(['audio'])];h.elements.audioFile.listeners.change();await settle();
  assert.equal(upload.body.get('model'),'chosen');assert.equal(upload.body.get('file').type,'audio/wav');
  assert.equal(h.inserted,'');assert.equal(h.elements.audioResult.value,'Hallo Welt');
  h.elements.audioInsert.click();assert.equal(h.inserted,'Hallo Welt');assert.equal(h.elements.audioResult.hidden,true);
});

test('cancellation aborts upload and ignores late response',async()=>{
  let resolve,signal;
  const h=harness(async(url,opts)=>{
    if(url==='/models/audio')return {ok:true,json:async()=>({models:[{id:'v'}]})};
    signal=opts.signal;return new Promise(r=>{resolve=r;});
  });
  h.elements.audioToggle.click();await settle();h.elements.audioFile.files=[new Blob(['audio'])];h.elements.audioFile.listeners.change();await settle();
  h.elements.audioCancel.click();assert.equal(signal.aborted,true);
  resolve({ok:true,json:async()=>({text:'late result'})});await settle();
  assert.equal(h.elements.audioResult.value,'');assert.equal(h.elements.audioStatus.textContent,'Cancelled.');
});

test('microphone permission arriving after cancel releases all tracks',async()=>{
  const h=harness();let grant;
  h.context.navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>{grant=resolve;});
  h.elements.audioToggle.click();await settle();h.elements.audioRecord.click();
  h.elements.audioCancel.click();grant({getTracks(){return [{stop(){h.trackStopped=true;}}];}});await settle();
  assert.equal(h.trackStopped,true);assert.equal(h.elements.audioStatus.textContent,'Cancelled.');
});

test('no model and browser-only mode cannot initiate transcription',async()=>{
  const h=harness(async()=>({ok:true,json:async()=>({models:[]})}));
  h.elements.audioToggle.click();await settle();assert.equal(h.elements.audioUpload.disabled,true);assert.match(h.elements.audioStatus.textContent,/No Voxtral/);
  h.controls.setAvailable(false);assert.equal(h.elements.audioToggle.hidden,true);assert.equal(h.elements.audioPanel.hidden,true);
});

test('stopping a microphone recording transcribes and releases tracks',async()=>{
  const h=harness(async url=>url==='/models/audio'
    ? {ok:true,json:async()=>({models:[{id:'v'}]})}
    : {ok:true,json:async()=>({text:'Recorded speech'})});
  h.context.MediaRecorder=class {
    constructor(){this.listeners={};this.state='inactive';this.mimeType='audio/webm';}
    addEventListener(name,fn){this.listeners[name]=fn;}
    start(){this.state='recording';}
    stop(){this.state='inactive';this.listeners.dataavailable({data:new Blob(['audio'])});this.listeners.stop();}
  };
  h.elements.audioToggle.click();await settle();await h.elements.audioRecord.click();
  assert.equal(h.elements.audioRecord.textContent,'Stop & transcribe');
  await h.elements.audioRecord.click();await settle();
  assert.equal(h.stopped,1);assert.equal(h.elements.audioResult.value,'Recorded speech');
  assert.equal(h.elements.audioCancel.hidden,true);
});

test('system audio capture shares tab/screen audio, drops the video track, and transcribes only the audio track',async()=>{
  const h=harness(async url=>url==='/models/audio'
    ? {ok:true,json:async()=>({models:[{id:'v'}]})}
    : {ok:true,json:async()=>({text:'Shared audio text'})});
  let videoStopped=0,audioStopped=0,displayMediaCall=null,recordedMedia=null;
  const videoTrack={kind:'video',stop(){videoStopped++;}};
  const audioTrack={kind:'audio',stop(){audioStopped++;},addEventListener(){},removeEventListener(){}};
  h.context.navigator.mediaDevices.getDisplayMedia=async constraints=>{
    displayMediaCall=constraints;
    return {getVideoTracks(){return [videoTrack];},getAudioTracks(){return [audioTrack];},getTracks(){return [videoTrack,audioTrack];}};
  };
  h.context.navigator.mediaDevices.enumerateDevices=async()=>[];
  h.context.MediaRecorder=class {
    constructor(media){this.listeners={};this.state='inactive';this.mimeType='audio/webm';recordedMedia=media;}
    addEventListener(name,fn){this.listeners[name]=fn;}
    start(){this.state='recording';}
    stop(){this.state='inactive';this.listeners.dataavailable({data:new Blob(['audio'])});this.listeners.stop();}
  };
  h.elements.audioToggle.click();await settle();
  assert.equal(h.elements.audioInputDevice.options.some(o=>o.value==='__system_audio__'),true);
  h.elements.audioInputDevice.value='__system_audio__';
  await h.elements.audioRecord.click();await settle();
  assert.ok(displayMediaCall);assert.equal(displayMediaCall.video,true);
  assert.equal(videoStopped,1,'the discarded video track must be stopped immediately');
  assert.deepEqual(recordedMedia.getTracks(),[audioTrack]);
  await h.elements.audioRecord.click();await settle();
  assert.equal(audioStopped,1);
  assert.equal(h.elements.audioResult.value,'Shared audio text');
});

test('system audio capture surfaces a clear error when no audio track is shared',async()=>{
  const h=harness(async()=>({ok:true,json:async()=>({models:[{id:'v'}]})}));
  let videoStopped=0;
  const videoTrack={kind:'video',stop(){videoStopped++;}};
  h.context.navigator.mediaDevices.getDisplayMedia=async()=>({
    getVideoTracks(){return [videoTrack];},getAudioTracks(){return [];},getTracks(){return [videoTrack];},
  });
  h.context.navigator.mediaDevices.enumerateDevices=async()=>[];
  h.elements.audioToggle.click();await settle();
  h.elements.audioInputDevice.value='__system_audio__';
  await h.elements.audioRecord.click();await settle();
  // Stopped twice: once immediately after capture, once more by the
  // no-audio-track cleanup path that stops every remaining track -- both
  // calls target the same track and stop() is idempotent on a real one.
  assert.equal(videoStopped,2);
  assert.match(h.elements.audioStatus.textContent,/No audio was shared/);
});
