// Screenshot + console-error check for a local page.
// usage: node scripts/shot.cjs <file.html> <out.png> [width] [colorScheme]
const path = require('path');
const { chromium } = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright');
(async () => {
  const [file, out, width = '1000', scheme = 'light'] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: +width, height: 800 }, colorScheme: scheme });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto('file://' + path.resolve(file), { waitUntil: 'networkidle' });
  await page.waitForTimeout(500);
  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
  if (errors.length) { console.log('CONSOLE ERRORS:'); errors.forEach(e => console.log('  ' + e)); process.exit(1); }
  console.log('OK', out);
})();
