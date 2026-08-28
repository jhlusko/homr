/**
 * Does review.html actually engrave and highlight, in a DOM?
 *
 * This exists because the engraving shipped broken. It had been "verified" by
 * rendering the MusicXML with Verovio in Node and syntax-checking the page - neither
 * of which can see that `scoreUrl` was a const scoped inside render(), and so was not
 * defined where renderEngraving() called it. Every item showed
 * "Could not engrave this label (scoreUrl is not defined)".
 *
 * So this runs the page's own functions against a real DOM: the same failure would
 * fail this test.
 *
 *   npm i jsdom verovio            # or point NODE_PATH at somewhere that has them
 *   node docs/writeups/test-engraving.js
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = __dirname;
const req = (name) => {
  try { return require(name); } catch (e) {
    console.error(`missing dependency ${name} - run: npm i jsdom verovio`);
    process.exit(2);
  }
};
const { JSDOM } = req('jsdom');
const verovio = req('verovio');

const SETS = ['overfull-single', 'overfull-grandstaff'];

verovio.module.onRuntimeInitialized = async () => {
  const html = fs.readFileSync(path.join(ROOT, 'review.html'), 'utf8');
  const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]).join('\n');
  const dom = new JSDOM('<!doctype html><div id="engraving"></div>', { pretendToBeVisual: true });
  const { window } = dom;
  // jsdom implements neither; the highlight needs a box and the render needs a width.
  // This checks the wiring, not the geometry.
  window.SVGElement.prototype.getBBox = () => ({ x: 10, y: 10, width: 100, height: 60 });
  Object.defineProperty(window.HTMLElement.prototype, 'clientWidth', { get: () => 900 });

  const sandbox = {
    window, document: window.document, verovio, console,
    localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
    fetch: async (url) => {
      const p = path.join(ROOT, url.replace(/^\//, ''));
      if (!fs.existsSync(p)) return { ok: false, status: 404 };
      return {
        ok: true, status: 200,
        text: async () => fs.readFileSync(p, 'utf8'),
        json: async () => JSON.parse(fs.readFileSync(p, 'utf8')),
      };
    },
  };
  const ctx = vm.createContext(sandbox);
  vm.runInContext(js.slice(0, js.indexOf('\nbuildNav();')), ctx);
  sandbox.__tk = new verovio.toolkit();
  // Top-level `let` in a vm script is a lexical binding, not a property of the context
  // object, so these have to be assigned from inside the context.
  vm.runInContext('vrvToolkit = __tk; vrvReady = true; viewMode = "detail";', ctx);

  let checked = 0, failures = 0;
  for (const setName of SETS) {
    const manifest = path.join(ROOT, 'review-data', 'sets', setName, 'manifest.json');
    if (!fs.existsSync(manifest)) { console.log(`  skip ${setName} (not built)`); continue; }
    const items = JSON.parse(fs.readFileSync(manifest, 'utf8'));
    sandbox.__items = items;
    vm.runInContext(`setName = ${JSON.stringify(setName)}; items = __items;`, ctx);
    for (let i = 0; i < items.length; i++) {
      vm.runInContext(`idx = ${i};`, ctx);
      const host = window.document.getElementById('engraving');
      host.innerHTML = '';
      await ctx.renderEngraving();
      checked++;
      const want = (items[i].overfull_bars || []).length;
      const got = host.querySelectorAll('rect.vrv-hl').length;
      const svgs = host.querySelectorAll('svg').length;
      const failed = /Could not engrave/.test(host.innerHTML) || svgs === 0 || got !== want;
      if (failed) {
        failures++;
        console.log(`  FAIL ${items[i].id}: svg=${svgs} highlights=${got}/${want} ` +
                    host.textContent.slice(0, 90));
      }
    }
  }
  console.log(failures
    ? `\n${failures} of ${checked} items failed to engrave`
    : `\nall ${checked} items engrave, with the expected number of highlights`);
  process.exit(failures ? 1 : 0);
};
