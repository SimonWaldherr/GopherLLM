const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(__dirname+'/script.js','utf8');
const context=vm.createContext({});
vm.runInContext(source.slice(source.indexOf('function planSettingsSearch('),source.indexOf('(async function initChat()')),context);
const plan=context.planSettingsSearch;
const section=(text,advanced=false,available=true)=>({text,advanced,available,element:text});
const pages=[
 {key:'model',sections:[section('Max output tokens and instructions'),section('Administrator token',false,false)]},
 {key:'generation',advanced:true,sections:[section('Sampling temperature Top P seed',true)]},
 {key:'workspace',sections:[section('Appearance theme'),section('Power commands',true)]}
];
test('simple search identifies advanced results without switching to an invisible tab',()=>{
 const result=plan(pages,'temperature','model',true);
 assert.equal(result.total,0);assert.equal(result.advanced,1);assert.equal(result.selected,'model');
});
test('advanced mode leads directly to the matching section',()=>{
 const result=plan(pages,'temperature','model',false);
 assert.equal(result.total,1);assert.equal(result.advanced,0);assert.equal(result.selected,'generation');
});
test('cross-tab search finds browser preferences and preserves an already matching tab',()=>{
 assert.equal(plan(pages,'theme','model',true).selected,'workspace');
 assert.equal(plan(pages,'theme','workspace',true).selected,'workspace');
});
test('unavailable administrator controls never become search results',()=>{
 for(const simple of [false,true]){
  const result=plan(pages,'administrator','model',simple);
  assert.equal(result.total,0);assert.equal(result.advanced,0);
 }
});
test('search accepts multiple terms regardless of order and case',()=>{
 const result=plan(pages,'  TOKENS   output ','workspace',true);
 assert.equal(result.total,1);assert.equal(result.selected,'model');
 assert.equal(plan(pages,'tokens nonexistent','model',true).total,0);
});
test('advanced sections within a simple tab are accounted for separately',()=>{
 const result=plan(pages,'commands','workspace',true);
 assert.equal(result.total,0);assert.equal(result.advanced,1);
 assert.equal(plan(pages,'commands','workspace',false).total,1);
});
test('empty queries keep current navigation and count only reachable sections',()=>{
 const result=plan(pages,'','workspace',true);
 assert.equal(result.selected,'workspace');assert.equal(result.total,2);
});
