import fs from'node:fs';
import assert from'node:assert/strict';
import{createHash}from'node:crypto';
export async function loadWindowsContract(){
 const vendor=JSON.parse(fs.readFileSync(new URL('./vendor.json',import.meta.url),'utf8'));
 assert.deepEqual(Object.keys(vendor.files).sort(),['windows-context-contract.mjs','windows-native.cs']);
 for(const[name,hash]of Object.entries(vendor.files)){
  const source=fs.readFileSync(new URL('./'+name,import.meta.url),'utf8').replaceAll('\r\n','\n').trimEnd()+'\n';
  assert.equal(createHash('sha256').update(source).digest('hex'),hash,'Reviewed native fixture changed: '+name);
 }
 return import('./windows-context-contract.mjs');
}
