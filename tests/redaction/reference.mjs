// JavaScript reference for the level-1 rules: proves the rule file means the same in JavaScript.
// Usage: node reference.mjs <rules.json> <corpus.json>. Prints JSON {failures: [...], checked: n}.
import { readFileSync } from "node:fs";

const rulesDoc = JSON.parse(readFileSync(process.argv[2], "utf8"));
const corpus = JSON.parse(readFileSync(process.argv[3], "utf8"));

const fold = new Map();
for (const e of rulesDoc.fold) {
  for (let cp = e.from[0]; cp <= e.from[1]; cp++) fold.set(cp, e.shift ? e.to + (cp - e.from[0]) : e.to);
}

function luhn(d) {
  let t = 0;
  for (let i = 0; i < d.length; i++) {
    let n = d.charCodeAt(d.length - 1 - i) - 48;
    if (i % 2) { n *= 2; if (n > 9) n -= 9; }
    t += n;
  }
  return t % 10 === 0;
}
const VD = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]];
const VP = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]];
function verhoeff(d) {
  let c = 0;
  for (let i = 0; i < d.length; i++) c = VD[c][VP[i % 8][d.charCodeAt(d.length - 1 - i) - 48]];
  return c === 0;
}
const VALIDATORS = { luhn, verhoeff };

const rules = rulesDoc.rules.map((r) => ({
  label: r.label,
  re: new RegExp(r.pattern, "g"),
  pre: r.prefilter ? new RegExp(r.prefilter) : null,
  validators: (r.validators || []).map((n) => VALIDATORS[n]),
  deny: r.deny_prefix ? new RegExp(r.deny_prefix.pattern) : null,
  window: r.deny_prefix ? r.deny_prefix.window : 0,
}));

const integerKeys = new RegExp(rulesDoc.integer_keys);

function redact(text, key) {
  const context = key ? key.toLowerCase() + ": " : "";
  let shadow = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    shadow += fold.has(cp) ? String.fromCharCode(fold.get(cp)) : ch;
  }
  for (const rule of rules) {
    if (rule.pre && !rule.pre.test(shadow)) continue;
    let out = "", sh = "", last = 0, hit = false;
    rule.re.lastIndex = 0;
    for (const m of shadow.matchAll(rule.re)) {
      const start = m.index, end = start + m[0].length;
      if (start === end) continue;
      const digits = m[0].replace(/[^0-9]/g, "");
      if (!rule.validators.every((v) => v(digits))) continue;
      if (rule.deny) {
        let before = shadow.slice(Math.max(0, start - rule.window), start);
        if (context && start < rule.window) before = (context + before).slice(-rule.window);
        if (rule.deny.test(before.toLowerCase())) continue;
      }
      out += text.slice(last, start) + rule.label;
      sh += shadow.slice(last, start) + rule.label;
      last = end;
      hit = true;
    }
    if (!hit) continue;
    text = out + text.slice(last);
    shadow = sh + shadow.slice(last);
  }
  return text;
}

function walk(value, key) {
  if (typeof value === "string") return redact(value, key);
  if (typeof value === "number") {
    if (!Number.isInteger(value) || Math.abs(value) < 1e9 || !key || !integerKeys.test(key.toLowerCase())) return value;
    const text = String(value);
    const out = redact(text, key);
    return out === text ? value : out;
  }
  if (Array.isArray(value)) return value.map((item) => walk(item, key));
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = walk(v, k);
    return out;
  }
  return value;
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const failures = [];
let checked = 0;
for (const e of corpus.must_redact) {
  checked++;
  const got = redact(e.text);
  if (got !== e.expect) failures.push({ id: e.id, got, want: e.expect });
}
for (const e of corpus.must_keep) {
  checked++;
  const got = redact(e.text);
  if (got !== e.text) failures.push({ id: e.id, got, want: e.text });
}
for (const e of corpus.values_must_redact || []) {
  checked++;
  const got = walk(e.value);
  if (!same(got, e.expect)) failures.push({ id: e.id, got, want: e.expect });
}
for (const e of corpus.values_must_keep || []) {
  checked++;
  const got = walk(e.value);
  if (!same(got, e.value)) failures.push({ id: e.id, got, want: e.value });
}
console.log(JSON.stringify({ failures, checked }));
