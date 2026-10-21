// Screenshot + console-error check for a local page or SVG.
// usage: node scripts/shot.cjs <file> <out.png> [width] [colorScheme]
const path = require('path'), fs = require('fs'), os = require('os');
const { chromium } = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright');
(async () => {
  const [file, out, width = '1000', scheme = 'light'] = process.argv.slice(2);
  let target = path.resolve(file);
  if (target.endsWith('.svg')) {
    const tmp = path.join(os.tmpdir(), 'svgwrap_' + Date.now() + '.html');
    fs.writeFileSync(tmp, '<!doctype html><body style="margin:0;background:#fff"><img src="file://' + target + '" style="display:block;max-width:100%"></body>');
    target = tmp;
  }
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: +width, height: 800 }, colorScheme: scheme });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto('file://' + target, { waitUntil: 'networkidle' });
  await page.waitForTimeout(400);
  await page.screenshot({ path: out, fullPage: true, timeout: 20000 });
  await browser.close();
  if (errors.length) { console.log('CONSOLE ERRORS:'); errors.forEach(e => console.log('  ' + e)); process.exit(1); }
  console.log('OK', out);
})();
