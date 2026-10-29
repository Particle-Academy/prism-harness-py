import assert from 'node:assert/strict';
export function requireNativeVerdict(result,platform,backend,validateWindowsContexts){
 assert.equal(result.status,0,'Native producer failed');assert(!result.error);
 assert(['linux','darwin','win32'].includes(platform));
 const lines=result.stdout.split(/\r?\n/);
 const reports=lines.filter(line=>line.startsWith('{')).map(line=>JSON.parse(line));
 const report=reports.find(value=>value.production===true&&value.backend===backend&&value.platform===platform);
 assert(report,'Missing actual production report');
 if(platform==='win32'){
  assert.equal(typeof validateWindowsContexts,'function');validateWindowsContexts(report);
  assert.equal(report.failures.length,0);assert.equal(report.rows.length,4);
  const measured=report.rows.filter(row=>row.status==='passed').length;
  const baseline=report.rows.find(row=>row.name==='no-job');
  if(baseline.status==='passed'){assert.equal(baseline.native.initial_no_job,true);assert.equal(measured,4);}
  else{assert.equal(baseline.status,'unmeasured');assert.equal(baseline.native.initial_no_job,false);assert.equal(baseline.native.child_ran,false);assert.equal(measured,3);}
  const state=baseline.status==='passed'?'verified-absent':'unmeasured-with-membership-evidence';
  assert(lines.includes('Production native owner proof passed: '+measured+' measured Windows contexts; no-job '+state+'; 8 report controls.'));
 }else{
  assert.equal(report.proof.live.probes,1);assert.equal(report.proof.dead.result,'nonce-bound-reclaimed');
  assert.equal(report.proof.dead.probes,1);assert.equal(report.proof.held.length,3);
  assert(report.proof.held.every(row=>row.probes===0&&row.directory_unchanged===true));
  assert(lines.includes('Production native owner proof passed: 1 live holder; 1 dead owner; 3 foreign/unknown controls; 8 report controls.'));
 }
 return report;
}
