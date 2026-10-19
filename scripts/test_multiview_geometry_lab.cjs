const assert = require('node:assert/strict');
const {execFileSync, spawnSync} = require('node:child_process');
const path = require('node:path');
const {calculate, presets} = require('../static/labs/multiview-geometry/geometry_lab.js');
const companion = path.resolve(__dirname, '../content/posts/ai/multiview-geometry/geometry_sanity.py');
const expected = {correct:0, reversed:96, updated:0, stale:24, scaled:0, baseline:48, mismatch:0};
const near = (a, b, label) => assert(Math.abs(a-b) < 1e-9, `${label}: ${a} != ${b}`);
for (const [name, input] of Object.entries(presets)) {
  const result = calculate(input);
  near(result.reprojection_error_px, expected[name], name);
  near(result.vertical_epipolar_error_px, 0, name);
}
near(calculate(presets.mismatch).match_reprojection_error_px,100,'Bad horizontal match');
near(calculate(presets.scaled).depth_relative_error_percent,100,'Invisible scale error');
const mutuallyWrong=calculate({...presets.reversed,match_offset_px:96});
near(mutuallyWrong.match_reprojection_error_px,0,'Wrong matches support wrong pose');
near(mutuallyWrong.reprojection_error_px,96,'Independent observation exposes wrong pose');
const base = calculate();
for (const scale of [.5,1,1.3,2,3]) {
  const result=calculate({baseline_m:.12*scale,depth_z_m:2*scale});
  near(result.reprojection_error_px,0,'Joint scale');
  near(result.depth_scale,scale,'Scale factor');
}
near(calculate({match_offset_px:100,crop_resize:true}).match_reprojection_error_px,50,'Resize offset');
assert(!calculate({baseline_m:.5,depth_z_m:.1}).projected_in_frame);
for (const patch of [{baseline_m:0},{baseline_m:Infinity},{baseline_m:NaN},
  {depth_z_m:.01},{depth_z_m:11},{match_offset_px:201},{match_offset_px:'2'},
  {reverse_pose:1},{crop_resize:'false'},{stale_intrinsics:true},{unknown:1},
  {toString:1},null,[],'bad']) assert.throws(()=>calculate(patch));

// Python backprojects a 3D point and applies a camera translation; JavaScript
// uses the reduced rectified formula. Compare their unrounded results.
const fixtures=Object.values(presets).map(p=>({...p}));
fixtures.push({baseline_m:.1234567,depth_z_m:2.1234567,match_offset_px:.125});
fixtures.push({...presets.reversed,match_offset_px:96});
for(let i=0;i<40;i++) fixtures.push({baseline_m:.02+i*.011,depth_z_m:.3+i*.21,
  reverse_pose:i%3===0,crop_resize:i%2===0,stale_intrinsics:i%4===0,
  match_offset_px:-150+i*7.3});
const program='import json,runpy,sys; f=runpy.run_path(sys.argv[1])["calculate_case"]; print(json.dumps([f(**p) for p in json.load(sys.stdin)],allow_nan=False))';
const python=JSON.parse(execFileSync('python3',['-B','-c',program,companion],
  {input:JSON.stringify(fixtures),encoding:'utf8',timeout:10000}));
function compare(a,b,label='result') {
  if(typeof a==='number') near(a,b,label);
  else if(Array.isArray(a)) {assert.equal(a.length,b.length,label);a.forEach((v,i)=>compare(v,b[i],label+'.'+i));}
  else if(a&&typeof a==='object') {assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort());for(const k of Object.keys(a))compare(a[k],b[k],label+'.'+k);}
  else assert.equal(a,b,label);
}
fixtures.forEach((input,i)=>compare(calculate(input),python[i],`case ${i}`));
// CLI ignores a forged cached result and recomputes the input only.
const replay=JSON.parse(execFileSync('python3',['-B',companion,'--input','-'],
  {input:JSON.stringify({input:presets.scaled,reprojection_error_px:999}),encoding:'utf8',timeout:10000}));
near(replay.reprojection_error_px,0,'CLI replay');
for(const payload of [{input:{depth_z_m:0}},{input:{reverse_pose:1}},{input:{unknown:1}},{}]) {
  const child=spawnSync('python3',['-B',companion,'--input','-'],{input:JSON.stringify(payload),encoding:'utf8',timeout:10000});
  assert.notEqual(child.status,0); assert(!child.stderr.includes('Traceback'),child.stderr);
}
for(const inputFile of ['', '/tmp/nonexistent-geometry-case-9fb25.json']){
  const child=spawnSync('python3',['-B',companion,'--input',inputFile],{encoding:'utf8',timeout:10000});
  assert.notEqual(child.status,0);assert(!child.stderr.includes('Traceback'),child.stderr);
}
console.log(JSON.stringify({presets:7,cross_language_cases:fixtures.length,
  checks:'projection, crop mapping, pose direction, joint scale, epipolar blind spot, bounds, invalid inputs, JSON replay passed'}));
