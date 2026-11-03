// Deterministic application screenshot for the article's image request.
// This is constructed input data, not a recorded agent trajectory.
const fs = require('node:fs');
const path = require('node:path');
const {connect} = require('../../../scripts/lib/browser.cjs');
const [port, output] = process.argv.slice(2);
const html = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Settings example</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#172337;font:18px Arial,sans-serif}
main{width:560px;margin:36px auto;padding:28px;background:white;border:1px solid #cad2df;border-radius:14px}
h1{margin:0 0 28px;font-size:30px}section{display:flex;align-items:center;justify-content:space-between}
h2{margin:0 0 10px;font-size:23px}p{margin:0;color:#40526b}label{display:flex;gap:10px;align-items:center;font-weight:bold}
input{appearance:none;margin:0;width:58px;height:32px;border-radius:20px;background:#147a50;position:relative}
input:after{content:'';position:absolute;right:4px;top:4px;width:24px;height:24px;border-radius:50%;background:white}
</style><main><h1>Settings</h1><section><div><h2>Notifications</h2>
<p>Desktop notifications are enabled.</p></div><label><input type="checkbox" checked aria-label="Notifications">ON</label>
</section></main></html>`;
(async () => {
  fs.mkdirSync(path.dirname(output), {recursive: true});
  const browser = await connect(Number(port));
  try {
    await browser.viewport(640, 360);
    await browser.navigate('data:text/html;charset=utf-8,' + encodeURIComponent(html));
    await browser.evaluate('document.fonts.ready');
    await browser.screenshot(output);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
