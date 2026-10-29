import {test} from 'node:test';
// Explicit Node test-runner input; excluded from Vitest's filename discovery.
import assert from 'node:assert/strict';
import {validateWindowsContexts,NO_JOB_UNMEASURED} from './windows-context-contract.mjs';
function fixture(){
 const identity={platform:'win32',boot_id:'fixture',pid_namespace:'win32:kernel'};
 const proof={identity,holder_identity:{...identity},live:{result:'held',probes:1},dead:{result:'nonce-bound-reclaimed',probes:1},
  held:['foreign-realm','foreign-realm-released','both-unknown'].map(id=>({id,result:'held-unknown',probes:0,directory_unchanged:true}))};
 const native={status:'passed',membership_checked:true,breakaway_requested:false,initial_no_job:false,verified_owned_job_depth:1,exit:0};
 return{failures:[],rows:[
  {name:'no-job',status:'unmeasured',reason:NO_JOB_UNMEASURED,native:{status:'unmeasured',membership_checked:true,breakaway_requested:true,initial_no_job:false,child_ran:false}},
  ...['ordinary-job','nested-job','inherited-supervisor-job'].map(name=>({name,status:'passed',native:{...native,verified_owned_job_depth:name==='nested-job'?2:1},proof:structuredClone(proof)})),
 ]};
}
test('accepts truthful retained-membership baseline and all three measured contexts',()=>{assert.equal(validateWindowsContexts(fixture()).measured_contexts,3);});
for(const[name,mutate]of[
 ['fake no-Job pass',r=>{r.rows[0].status='passed';r.rows[0].native.status='passed';}],
 ['missing ordinary Job',r=>{r.rows.splice(1,1);}],
 ['missing native membership check',r=>{r.rows[1].native.membership_checked=false;}],
 ['foreign PID probe',r=>{r.rows[2].proof.held[0].probes=1;}],
 ['zero live work',r=>{r.rows[1].proof.live.probes=0;}],
 ['failed actual recovery',r=>{r.rows[3].proof.dead.result='held';}],
 ['unmeasured baseline ran a child',r=>{r.rows[0].native.child_ran=true;}],
])test('rejects '+name,()=>{const value=fixture();mutate(value);assert.throws(()=>validateWindowsContexts(value));});
