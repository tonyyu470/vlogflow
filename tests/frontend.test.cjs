const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function harness(service) {
  const elements = new Map();
  const element = () => ({addEventListener(){}, classList:{toggle(){},add(){},remove(){}},
    elements:{provider:{addEventListener(){}}}, replaceChildren(){}, append(){}, value:'', dataset:{}});
  const document = {getElementById(id){if(!elements.has(id)) elements.set(id,element());return elements.get(id);},
    querySelectorAll(){return [];}, createElement:element};
  const scope = vm.createContext({document, window:{addEventListener(){}}, FormData, Blob,
    fetch:()=>new Promise(()=>{}), setTimeout:()=>1, clearTimeout(){}, service});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/vlogflow/static/app.js'),'utf8'),scope);
  vm.runInContext(`api=service; renderProject=()=>{}; showJob=()=>{}; listProjects=async()=>{};`,scope);
  return code => vm.runInContext(code,scope);
}
function project(id){return {id,artifacts:{},revision:null,epoch:0};}

test('an upload stays with its originating project after navigation',async()=>{
  const requests=[]; let finishUpload;
  const run=harness(async(url,options)=>{
    requests.push(url);
    if(url.endsWith('/uploads')) return new Promise(resolve=>finishUpload=resolve);
    if(url==='/projects/B') return project('B');
    if(url==='/projects/B/jobs'&&!options) return [];
    return {id:'import-A',status:'queued'};
  });
  run(`project={id:'A',epoch:0};`);
  const upload=run(`importVideoFiles([new Blob(['video'])])`);
  await run(`loadProject('B')`);
  finishUpload({paths:['/staging/A.mp4']}); await upload;
  assert.equal(requests.at(-1),'/projects/A/jobs');
  assert.equal(run('project.id'),'B');
  assert.equal(run('job'),null);
});

test('a slower earlier navigation cannot replace the selected project',async()=>{
  let finishA;
  const run=harness(async url=>{
    if(url==='/projects/A') return new Promise(resolve=>finishA=resolve);
    if(url==='/projects/B') return project('B');
    return [];
  });
  const first=run(`loadProject('A')`);
  await run(`loadProject('B')`);
  finishA(project('A')); await first;
  assert.equal(run('project.id'),'B');
});

test('reference upload retains its original project and transcript',async()=>{
  let finishUpload, submitted;
  const run=harness(async(url,options)=>{
    if(url.endsWith('/uploads')) return new Promise(resolve=>finishUpload=resolve);
    if(url==='/projects/B') return project('B');
    if(url==='/projects/B/jobs') return [];
    submitted={url,body:options.body}; return {id:'voice-A',status:'queued'};
  });
  run(`project={id:'A'};`);
  const upload=run(`importAudioFiles([new Blob(['audio'])],'Original transcript')`);
  await run(`loadProject('B')`);
  finishUpload({paths:['/staging/A.wav']});await upload;
  assert.equal(submitted.url,'/projects/A/jobs');
  assert.equal(submitted.body.transcript,'Original transcript');
  assert.equal(run('job'),null);
});

test('an in-flight script save cannot replace a newly selected project',async()=>{
  let finishSave;
  const run=harness(async(url,options)=>{
    if(url==='/projects/A'&&options?.method==='PATCH') return new Promise(resolve=>finishSave=resolve);
    if(url==='/projects/B') return project('B');
    return [];
  });
  run(`project={id:'A',epoch:0};collectScenes=()=>[];`);
  const save=run('saveScript()');
  const rejected=assert.rejects(save);
  await run(`loadProject('B')`);
  finishSave(project('A'));await rejected;
  assert.equal(run('project.id'),'B');
});

test('a completed job refresh cannot overwrite a later navigation',async()=>{
  let finishA, requestedA;
  const ready = new Promise(resolve=>requestedA=resolve);
  const run=harness(async url=>{
    if(url==='/jobs/job-A') return {id:'job-A',status:'complete'};
    if(url==='/projects/A') {requestedA();return new Promise(resolve=>finishA=resolve);}
    if(url==='/projects/B') return project('B');
    return [];
  });
  run(`project={id:'A',epoch:0};job={id:'job-A',status:'running'};`);
  const polling=run('pollJob()'); await ready;
  await run(`loadProject('B')`);finishA(project('A'));await polling;
  assert.equal(run('project.id'),'B');
  assert.equal(run('job'),null);
});
