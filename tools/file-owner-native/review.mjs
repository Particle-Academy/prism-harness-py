import fs from 'node:fs';
import os from 'node:os';
import assert from 'node:assert/strict';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {loadWindowsContract} from './vendor.mjs';
// Verify both vendored files before importing the contract or compiling C#.
const {validateWindowsContexts}=await loadWindowsContract();
const ROOT=fileURLToPath(new URL('../../',import.meta.url)),SELF=fileURLToPath(import.meta.url);
const [mode,argument,output]=process.argv.slice(2);
const backend=mode?.startsWith('--')?argument:mode;
assert(['ts','py'].includes(backend),'Select one production backend');
const env=Object.fromEntries(Object.entries(process.env).filter(([key])=>/^(PATH|SYSTEMROOT|WINDIR|COMSPEC|PATHEXT|TEMP|TMP|PRISM_FILE_OWNER_PYTHON)$/i.test(key)));
if(backend==='py')env.PYTHONPATH=join(ROOT,'src');
function execute(exe,args,extra={},timeout=60000){
 const child=spawnSync(exe,args,{cwd:ROOT,env:{...env,...extra},encoding:'utf8',windowsHide:true,timeout,maxBuffer:4*1024*1024});
 if(extra.PRISM_FILE_OWNER_NATIVE_REPORT){fs.writeFileSync(extra.PRISM_FILE_OWNER_NATIVE_REPORT+'.stdout.log',child.stdout??'',{flag:'wx'});fs.writeFileSync(extra.PRISM_FILE_OWNER_NATIVE_REPORT+'.stderr.log',child.stderr??'',{flag:'wx'});}
 assert(!child.error,'Native command must execute: '+child.error?.code);
 assert.equal(child.status,0,child.stdout+'\n'+child.stderr);return child.stdout;
}
function proof(report){
 const args=backend==='ts'?['node_modules/vitest/vitest.mjs','run','test/file-owner-native.test.ts']:['-m','pytest','-q','-s','tests/test_file_owner_native.py'];
 const stdout=execute(backend==='ts'?process.execPath:(env.PRISM_FILE_OWNER_PYTHON??'python'),args,{PRISM_FILE_OWNER_NATIVE_REPORT:report});
 const value=JSON.parse(fs.readFileSync(report,'utf8'));
 const metadata=stdout.replace(/\x1b\[[0-9;]*m/g,'').split(/\r?\n/).filter(line=>line.includes('"native_identity_setup"')).map(line=>JSON.parse(line.slice(line.indexOf('{'))));
 assert(metadata.length>0,'Native metadata timing must be measured');
 assert(metadata.every(row=>row.native_identity_setup===true&&row.platform===process.platform&&typeof row.elapsed_ms==='number'&&row.elapsed_ms>=0));
 value.metadata_setup=metadata;fs.writeFileSync(report,JSON.stringify(value));
 assert.equal(value.backend,backend);assert.equal(value.production,true);
 assert.equal(value.identity.platform,process.platform);assert.equal(typeof value.identity.pid_namespace,'string');
 assert.notEqual(value.holder_pid,value.prober_pid);assert.deepEqual(value.identity,value.holder_identity);
 return value;
}
const nativeSource=fs.readFileSync(new URL('./windows-native.cs',import.meta.url),'utf8');
function launch(args,depth,requireNoJob,breakaway){
 const command="$ErrorActionPreference='Stop'\nAdd-Type -TypeDefinition @'\n"+nativeSource+"\n'@\n$data=[Console]::In.ReadToEnd() | ConvertFrom-Json\n[PrismLockNativeReview]::Launch($data.exe,[string[]]$data.args,$data.cwd,[int]$data.depth,[bool]$data.require_no_job,[bool]$data.breakaway)";
 const child=spawnSync('powershell.exe',['-NoLogo','-NoProfile','-NonInteractive','-Command',command],{
  input:JSON.stringify({exe:process.execPath,args,cwd:ROOT,depth,require_no_job:requireNoJob,breakaway}),
  cwd:ROOT,env,encoding:'utf8',windowsHide:true,timeout:75000,maxBuffer:4*1024*1024,
 });
 assert(!child.error,'Owned native setup must execute');assert.equal(child.status,0,child.stderr);
 return JSON.parse(child.stdout.trim());
}
if(mode==='--child'){proof(output);process.exit(0);}
if(mode==='--supervisor'){
 const report=output+'.proof.json',native=launch([SELF,'--child',backend,report],1,false,false);
 assert.equal(native.initial_no_job,false);
 fs.writeFileSync(output,JSON.stringify({name:'inherited-supervisor-job',status:'passed',native,proof:JSON.parse(fs.readFileSync(report))}),{flag:'wx'});
 process.exit(0);
}
const contracts=execute(process.execPath,['--test','--test-reporter=tap','tools/file-owner-native/report-controls.mjs']);
assert.match(contracts,/^# tests 8$/m,'All eight report controls must execute');
const root=fs.mkdtempSync(join(os.tmpdir(),'prism-production-contexts-'));
if(process.platform!=='win32'){
 const value=proof(join(root,'native.json'));console.log(JSON.stringify({backend,platform:process.platform,production:true,proof:value}));
 console.log('Production native owner proof passed: 1 live holder; 1 dead owner; 3 foreign/unknown controls; 8 report controls.');
}else{
 const rows=[],failures=[];
 for(const[name,depth]of[['no-job',0],['ordinary-job',1],['nested-job',2]]){
  try{
   const report=join(root,name+'.json'),native=launch([SELF,'--child',backend,report],depth,depth===0,depth===0);
   if(native.status==='unmeasured'){rows.push({name,status:'unmeasured',reason:native.reason,native});continue;}
   rows.push({name,status:'passed',native,proof:JSON.parse(fs.readFileSync(report,'utf8'))});
  }catch(error){failures.push({name,error:error.message});}
 }
 try{
  const report=join(root,'inherited.json'),supervisor=launch([SELF,'--supervisor',backend,report],1,false,false);
  rows.push({...JSON.parse(fs.readFileSync(report,'utf8')),supervisor});
 }catch(error){failures.push({name:'inherited-supervisor-job',error:error.message});}
 const result={backend,platform:'win32',production:true,rows,failures,container_path:'unmeasured; no container created',fixture:root};
 console.log(JSON.stringify(result));const verdict=validateWindowsContexts(result);
 console.log('Production native owner proof passed: '+verdict.measured_contexts+' measured Windows contexts; no-job '+verdict.no_job+'; 8 report controls.');
}
