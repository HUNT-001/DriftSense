// Real-browser end-to-end check of the SPA against a live API (the deployed stack or
// `python -m service.dev_server --ui ui/dist`). Runs an escalating agent episode, approves rank 1
// on the approval card, waits for success, checks 375 px width, and saves screenshots.
//
//   npm i --no-save playwright-core@1.55.0
//   node scripts/browser-e2e.cjs http://127.0.0.1:8000 /tmp/shots [path-to-chrome-or-edge]
//
// Uses an installed Chrome/Edge (no browser download). Exit 0 = PASS.
const BROWSER =
  process.argv[4] ||
  process.env.BROWSER_PATH ||
  (process.platform === 'win32'
    ? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
    : '/usr/bin/google-chrome');
const { chromium } = require('playwright-core');
const base = process.argv[2] || 'http://127.0.0.1:8000';
const out = process.argv[3] || '.';
(async () => {
  const browser = await chromium.launch({ executablePath: BROWSER, headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  const t0 = Date.now();
  await page.goto(base + '/');
  await page.waitForSelector('#run-episode');
  await page.screenshot({ path: out + '/01_home.png', fullPage: true });
  await page.selectOption('#ep-family', 'dram');
  await page.fill('#ep-seed', '17');
  await page.click('#run-episode');
  await page.waitForSelector('#approver', { timeout: 120000 });
  await page.screenshot({ path: out + '/02_approval.png', fullPage: true });
  await page.fill('#approver', 'judge-smoke');
  await page.getByRole('button', { name: 'Approve rank 1' }).click();
  await page.getByRole('button', { name: 'Confirm' }).click();
  await page.getByText("Succeeded on the operator's choice", { exact: false }).first().waitFor({ timeout: 120000 });
  await page.screenshot({ path: out + '/03_result.png', fullPage: true });
  const rows = await page.locator('li, tr').count();
  console.log(JSON.stringify({ ok: true, seconds: (Date.now() - t0) / 1000, rows, errors }));
  // mobile width
  const m = await browser.newPage({ viewport: { width: 375, height: 800 } });
  await m.goto(base + '/');
  await m.waitForSelector('#run-episode');
  const overflow = await m.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  await m.screenshot({ path: out + '/04_mobile.png', fullPage: true });
  console.log(JSON.stringify({ mobile_horizontal_overflow: overflow }));
  if (overflow) process.exitCode = 1;
  await browser.close();
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
