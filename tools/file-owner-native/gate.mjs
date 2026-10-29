import assert from'node:assert/strict';
import{spawnSync}from'node:child_process';
import{fileURLToPath}from'node:url';
import{requireNativeVerdict}from'./verdict.mjs';
import{loadWindowsContract}from'./vendor.mjs';
const{validateWindowsContexts}=await loadWindowsContract();
const cwd=fileURLToPath(new URL('../../',import.meta.url)),backend=process.argv[2];
const controls=spawnSync(process.execPath,['--test','--test-reporter=tap','tools/file-owner-native/verdict-controls.mjs'],{cwd,encoding:'utf8',windowsHide:true,timeout:30000});
assert(!controls.error);assert.equal(controls.status,0,controls.stdout+controls.stderr);assert.match(controls.stdout,/^# tests 6$/m);
const result=spawnSync(process.execPath,['tools/file-owner-native/review.mjs',backend],{cwd,env:process.env,encoding:'utf8',windowsHide:true,timeout:240000,maxBuffer:4*1024*1024});
process.stdout.write(result.stdout??'');process.stderr.write(result.stderr??'');
requireNativeVerdict(result,process.platform,backend,validateWindowsContexts);
console.log('Native gate evidence confirmed: production owner proof; 6 verdict controls.');
if(backend==='py'&&process.env.GITHUB_ACTIONS==='true'&&['darwin','linux'].includes(process.platform)){
 const stress=spawnSync(process.env.PRISM_PYTHON_BINARY??'python',['tools/file-owner-native/overlap-stress.py'],{cwd,env:process.env,encoding:'utf8',windowsHide:true,timeout:180000,maxBuffer:4*1024*1024});
 process.stdout.write(stress.stdout??'');process.stderr.write(stress.stderr??'');
 assert(!stress.error);assert.equal(stress.status,0,'Hosted exclusion stress failed');
 assert.match(stress.stdout,/^Hosted owner exclusion passed: 200 iterations; 400 callbacks; 800 events; 0 overlaps; detector control rejected\.$/m);
 const rows=stress.stdout.split(/\r?\n/).filter(line=>line.startsWith('{')).map(JSON.parse);
 assert.equal(rows.length,1);assert.equal(rows[0].status,'PASS');assert.equal(rows[0].iterations,200);assert.equal(rows[0].overlaps,0);
}
