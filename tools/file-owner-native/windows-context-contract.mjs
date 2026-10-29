import assert from 'node:assert/strict';

export const NO_JOB_UNMEASURED = 'ambient supervisor Job cannot be left on this host (CREATE_BREAKAWAY_FROM_JOB accepted, membership retained)';

export function validateWindowsContexts(report) {
  assert.deepEqual(report.failures, [], 'Native setup errors cannot become unmeasured identity coverage');
  assert.deepEqual(report.rows.map(row => row.name).sort(), ['no-job', 'ordinary-job', 'nested-job', 'inherited-supervisor-job'].sort(), 'All four context records are required');
  const baseline = report.rows.find(row => row.name === 'no-job');
  for (const name of ['ordinary-job', 'nested-job', 'inherited-supervisor-job']) {
    assert.equal(report.rows.find(row => row.name === name).status, 'passed', 'Every measurable Windows context must pass');
  }
  assert.equal(baseline.native.status, baseline.status);
  assert.equal(baseline.native.membership_checked, true, 'no-Job proof needs an actual membership check');
  assert.equal(baseline.native.breakaway_requested, true);
  if (baseline.status === 'unmeasured') {
    assert.equal(baseline.reason, NO_JOB_UNMEASURED);
    assert.equal(baseline.native.initial_no_job, false, 'Unmeasured baseline must retain observed membership');
    assert.equal(baseline.native.child_ran, false);
    assert.equal(baseline.proof, undefined, 'An unmeasured row carries no recovery coverage');
  } else {
    assert.equal(baseline.status, 'passed');
    assert.equal(baseline.native.initial_no_job, true, 'no-Job pass requires verified absent membership');
  }
  for (const row of report.rows.filter(row => row.status !== 'unmeasured')) {
    assert.equal(row.status, 'passed');
    assert.equal(row.native.status, 'passed');
    assert.equal(row.native.membership_checked, true);
    assert.equal(row.native.exit, 0);
    const expectedDepth = row.name === 'no-job' ? 0 : row.name === 'nested-job' ? 2 : 1;
    assert.equal(row.native.verified_owned_job_depth, expectedDepth);
    if (row.name !== 'no-job') assert.equal(row.native.breakaway_requested, false, 'Measured Jobs must retain the ambient supervisor chain');
    if (row.name === 'inherited-supervisor-job') assert.equal(row.native.initial_no_job, false);
    assert.equal(row.proof.identity.pid_namespace, 'win32:kernel');
    assert.deepEqual(row.proof.identity, row.proof.holder_identity);
    assert.deepEqual(row.proof.live, { result: 'held', probes: 1 });
    assert.equal(row.proof.dead.result, 'nonce-bound-reclaimed');
    assert.equal(row.proof.dead.probes, 1);
    assert.deepEqual(row.proof.held.map(value => value.id), ['foreign-realm', 'foreign-realm-released', 'both-unknown']);
    for (const held of row.proof.held) {
      assert.equal(held.result, 'held-unknown'); assert.equal(held.probes, 0); assert.equal(held.directory_unchanged, true);
    }
  }
  return { measured_contexts: baseline.status === 'passed' ? 4 : 3, no_job: baseline.status === 'passed' ? 'verified-absent' : 'unmeasured-with-membership-evidence' };
}
