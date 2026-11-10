/**
 * The target's alignment, searched once and kept.
 *
 * 🔴 EVERY TEST HERE INJECTS A CacheStorage, because node has none - and that
 * is also the arm that matters most: the module must behave on a browser
 * WITHOUT one (a non-secure context, a private window, a full disk) exactly as
 * it does on a miss. A screen that refuses to start because storage is
 * unavailable would be a cache turned into a dependency.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { alignmentForScreen, canonicalSequence, countRecords, readCachedMsa,
  sequenceHash, writeCachedMsa } from "../shared/input/msa-cache.js";

/** Just enough of CacheStorage for this module's three calls. */
function fakeCaches(over = {}) {
  const store = new Map();
  return {
    store,
    open: over.open ?? (async () => ({
      match: async (url) => (store.has(url)
        ? { text: async () => store.get(url) } : undefined),
      put: over.put ?? (async (url, response) => {
        store.set(url, await response.text());
      }),
    })),
  };
}

// 🔴 THE FIXTURE'S QUERY MUST BE THE SEQUENCE IT IS STORED UNDER. It was
// `MKT` while every test keyed on `MKTAY`, i.e. these tests were caching an
// alignment of a DIFFERENT protein - which is exactly the mistake
// readCachedMsa's query check now refuses, and it failed four of them on the
// first run. The check is right; the fixture was wrong.
const A3M = ">query\nMKTAY\n>hit1\nMKSAY\n>hit2\nMRTAY\n";

test("the key ignores whitespace and case but nothing else", () => {
  assert.equal(canonicalSequence(" mkt ay\nia "), "MKTAYIA");
  assert.equal(sequenceHash("MKTAY"), sequenceHash(" mktay\n"));
  assert.notEqual(sequenceHash("MKTAY"), sequenceHash("MKTAW"));
  // 🔴 and a repeated motif must not alias a longer one: the length is hashed
  assert.notEqual(sequenceHash("AAA"), sequenceHash("AAAAAA"));
});

test("the hash is 16 hex characters and needs no crypto.subtle", () => {
  assert.match(sequenceHash("MKTAYIAKQRQISFVKSHFSRQ"), /^[0-9a-f]{16}$/);
});

test("depth is counted off the text", () => {
  assert.equal(countRecords(A3M), 3);
  assert.equal(countRecords(""), 0);
  assert.equal(countRecords(undefined), 0);
});

test("an a3m written for one sequence comes back for the same sequence", async () => {
  const caches = fakeCaches();
  assert.equal(await writeCachedMsa("MKTAY", A3M, { caches }), true);
  const hit = await readCachedMsa(" mktay ", { caches });
  assert.equal(hit.a3m, A3M);
  assert.equal(hit.depth, 3);
  // ...and not for a different one
  assert.equal(await readCachedMsa("MKTAW", { caches }), null);
});

test("🔴 the search runs once for the target, not once per ligand", async () => {
  const caches = fakeCaches();
  let searches = 0;
  const search = async () => { searches += 1; return A3M; };
  const first = await alignmentForScreen("MKTAY", { search, caches });
  assert.equal(first.source, "search");
  assert.equal(first.depth, 3);
  for (let ligand = 0; ligand < 5; ligand += 1) {
    const again = await alignmentForScreen("MKTAY", { search, caches });
    assert.equal(again.source, "cache");
    assert.equal(again.a3m, A3M);
  }
  assert.equal(searches, 1, "the MMseqs2 search must run exactly once");
});

test("🔴 with no CacheStorage at all it still works, by searching", async () => {
  let searches = 0;
  const search = async () => { searches += 1; return { a3m: A3M }; };
  // caches: null is what a non-secure context looks like to this module
  for (let run = 0; run < 2; run += 1) {
    const result = await alignmentForScreen("MKTAY", { search, caches: null });
    assert.equal(result.source, "search");
    assert.equal(result.depth, 3);
  }
  assert.equal(searches, 2, "without storage every run searches, and none fail");
});

test("🔴 a cache that throws on open is a miss, not a failure", async () => {
  const caches = fakeCaches({ open: async () => { throw new Error("denied"); } });
  assert.equal(await readCachedMsa("MKTAY", { caches }), null);
  assert.equal(await writeCachedMsa("MKTAY", A3M, { caches }), false);
  const result = await alignmentForScreen("MKTAY",
    { search: async () => A3M, caches });
  assert.equal(result.source, "search");
});

test("🔴 a quota error on put does not end the run", async () => {
  const caches = fakeCaches({
    put: async () => { throw new DOMException("quota", "QuotaExceededError"); },
  });
  const result = await alignmentForScreen("MKTAY",
    { search: async () => A3M, caches });
  assert.equal(result.source, "search");
  assert.equal(result.a3m, A3M);
});

test("a search that returns nothing is an error, not an empty alignment", async () => {
  await assert.rejects(
    () => alignmentForScreen("MKTAY", { search: async () => "", caches: fakeCaches() }),
    /returned no alignment/);
});

test("force re-searches even with a hit, and replaces it", async () => {
  const caches = fakeCaches();
  await writeCachedMsa("MKTAY", A3M, { caches });
  const deeper = `${A3M}>hit3\nMKA\n`;
  const result = await alignmentForScreen("MKTAY",
    { search: async () => deeper, caches, force: true });
  assert.equal(result.source, "search");
  assert.equal(result.depth, 4);
  assert.equal((await readCachedMsa("MKTAY", { caches })).depth, 4);
});

test("the status line says where the alignment came from", async () => {
  const caches = fakeCaches();
  const said = [];
  const onStatus = (message) => said.push(message);
  await alignmentForScreen("MKTAY", { search: async () => A3M, caches, onStatus });
  await alignmentForScreen("MKTAY", { search: async () => A3M, caches, onStatus });
  assert.match(said.join(" | "), /Searching for homologues/);
  assert.match(said.join(" | "), /MSA: 3 rows \(kept for the next run\)/);
  assert.match(said.join(" | "), /MSA from cache: 3 rows/);
});

test("🔴 a cached entry whose own query disagrees is a MISS, not a hit", async () => {
  // The search path throws on this; the cache used to accept any entry whose
  // hash matched. A collision, an entry from an older build, or a truncated
  // Response would silently substitute one target's alignment for another -
  // and a wrong MSA is invisible in the output.
  const caches = fakeCaches();
  await writeCachedMsa("MKTAY", A3M, { caches });
  // the stored a3m's query is MKT, not MKTAY: plant it under MKTAY's key
  const key = [...caches.store.keys()][0];
  caches.store.set(key, ">someone-else\nWWWW\n>hit\nWWWA\n");
  assert.equal(await readCachedMsa("MKTAY", { caches }), null);
  // ...and the run then searches rather than using it
  const result = await alignmentForScreen("MKTAY",
    { search: async () => A3M, caches });
  assert.equal(result.source, "search");
});

test("the query comparison drops insertion columns rather than upper-casing them",
  async () => {
    const { firstRecord } = await import("../shared/input/msa-cache.js");
    // lower case and dots/dashes are insertions relative to the query
    assert.equal(firstRecord(">q\nMKtaY..A-\n>h\nMKAAY\n"), "MKYA");
    assert.equal(firstRecord(">q\nMKT\n"), "MKT");
    assert.equal(firstRecord(""), null);
    assert.equal(firstRecord(">q\n\n"), null);
  });

test("🔴 a matching cached query is still a hit", async () => {
  const caches = fakeCaches();
  assert.equal(await writeCachedMsa("MKTAY", A3M, { caches }), true);
  const hit = await readCachedMsa(" mktay ", { caches });
  assert.notEqual(hit, null);
  assert.equal(hit.depth, 3);
});
