/**
 * The target's alignment, kept across ligands AND across reloads.
 *
 * 🔴 WHAT THIS SAVES AND WHAT IT DOES NOT. A screen folds the same protein once
 * per ligand, and an MMseqs2 search is tens of seconds of somebody else's
 * server. Running it once and reusing the a3m removes exactly that - the
 * SEARCH. It does **not** remove the trunk: the ligand's tokens enter the pair
 * and MSA representations, so both passes' trunks run per ligand and no
 * embedding of the protein can be carried between ligands. That claim is
 * repeated here because it is the one a reader of a screening tool most wants
 * to be false, and web/affinity-run.js's header makes it too.
 *
 * 🔴 AND THE EXISTING CACHE IS IN MEMORY ONLY, WHICH IS A RELOAD AWAY FROM A
 * SECOND SEARCH. `planSearchReuse` keys on `JSON.stringify(chains)` in a page
 * variable: right for one session, gone on refresh - and a screen of several
 * hundred ligands is a thing a reader comes back to. This persists the a3m in
 * CacheStorage under a hash of the sequence, so the second visit starts with
 * the alignment it already paid for.
 *
 * 🔴 THE KEY IS THE SEQUENCE, NOT THE JOB. Two screens of one target with
 * different ligand files share the alignment, which is the point; and a key
 * that included anything else would miss. The hash is over the CLEANED
 * sequence, so whitespace and case in the box cannot produce a second entry for
 * one protein.
 *
 * 🔴 AND IT IS A CACHE, SO EVERY PATH HERE TOLERATES FAILURE. CacheStorage is
 * absent in a non-secure context, throws in a private window on some browsers,
 * and can be evicted between the read and the write. A miss and a thrown
 * `open` must be the same thing to the caller - "search it" - or a screen
 * refuses to start on a browser whose storage is full.
 */

const CACHE_NAME = "localfold-msa-v1";
/** Not a real URL: CacheStorage keys on one, and this namespace is ours. */
const keyUrl = (hash) => `https://localfold.invalid/msa/${hash}.a3m`;

/**
 * The sequence as the key sees it: one chain, upper case, no whitespace.
 *
 * 🔴 NOT `cleanSequence`. That function keeps only amino-acid letters, which is
 * right for a sequence box and wrong for a cache key: it would silently map two
 * different inputs onto one key if either contained something it drops. Here
 * the rule is only to remove what cannot change the search - whitespace and
 * case - and anything else is part of the key.
 */
export function canonicalSequence(sequence) {
  return String(sequence ?? "").replace(/\s+/g, "").toUpperCase();
}

/**
 * A stable 64-bit-ish hash of the sequence, as 16 hex characters.
 *
 * 🔴 NOT crypto.subtle, DELIBERATELY. `crypto.subtle` is undefined in a
 * non-secure context - which is exactly where a reader running this off a file
 * server will be - so a SHA-256 key would make the cache unavailable precisely
 * where it is most wanted, and it is a cache key rather than a security
 * boundary. FNV-1a over two offsets, which for a protein sequence has no
 * practical collision risk and needs nothing from the platform.
 */
export function sequenceHash(sequence) {
  const text = canonicalSequence(sequence);
  let a = 0x811c9dc5;
  let b = 0x01000193;
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    a = Math.imul(a ^ code, 0x01000193) >>> 0;
    b = Math.imul(b ^ (code + index), 0x85ebca6b) >>> 0;
  }
  // the length goes in too, so a repeated motif cannot alias a longer one
  a = Math.imul(a ^ text.length, 0x01000193) >>> 0;
  return `${a.toString(16).padStart(8, "0")}${b.toString(16).padStart(8, "0")}`;
}

/** Whether this browser has CacheStorage at all, without throwing. */
function cacheAvailable() {
  try {
    return typeof caches !== "undefined" && caches !== null
      && typeof caches.open === "function";
  } catch {
    return false;
  }
}

/**
 * The cached a3m for this sequence, or null.
 *
 * @param {string} sequence
 * @param {{caches?: CacheStorage}} [options] injected for the tests, which have
 *   no CacheStorage of their own
 * @returns {Promise<{a3m: string, depth: number, hash: string} | null>}
 */
export async function readCachedMsa(sequence, options = {}) {
  const storage = options.caches ?? (cacheAvailable() ? caches : null);
  if (storage === null) return null;
  const hash = sequenceHash(sequence);
  try {
    const cache = await storage.open(CACHE_NAME);
    const hit = await cache.match(keyUrl(hash));
    if (hit === undefined || hit === null) return null;
    const a3m = await hit.text();
    if (typeof a3m !== "string" || a3m.length === 0) return null;
    // 🔴 THE CACHED ALIGNMENT'S OWN QUERY IS CHECKED, as the SEARCH path checks
    // it (`mmseqs2-api.js` throws "MMseqs2 returned an A3M for a different
    // query sequence"). Returning a hit on the strength of a 64-bit
    // non-cryptographic hash alone means any wrong entry - a collision, an
    // entry written by an older build, a truncated Response - silently
    // substitutes one target's alignment for another, and an MSA is the one
    // input whose being wrong is invisible in the output. The a3m carries its
    // own query record; comparing it costs a split.
    const query = firstRecord(a3m);
    if (query !== null && query !== canonicalSequence(sequence)) return null;
    // 🔴 THE DEPTH IS RE-COUNTED, NOT READ FROM A HEADER. A stored count that
    // disagrees with the stored text would set the MSA stack's pipeline keys
    // to a depth the batch does not have, which is a validation error far from
    // here. Counting is cheap and cannot disagree.
    return { a3m, depth: countRecords(a3m), hash };
  } catch {
    // An unreadable cache is a miss. See this file's header.
    return null;
  }
}

/**
 * The first record's sequence, upper-cased with insertions removed - the a3m's
 * own statement of what was searched.
 *
 * 🔴 LOWER-CASE COLUMNS ARE INSERTIONS RELATIVE TO THE QUERY and are not part
 * of it, so they are dropped rather than upper-cased. Keeping them would make
 * every alignment look like a different query and turn every cache hit into a
 * miss - a silent, total loss of the cache rather than a wrong answer.
 */
export function firstRecord(a3m) {
  const lines = String(a3m ?? "").split("\n");
  const start = lines.findIndex((line) => line.startsWith(">"));
  if (start < 0) return null;
  let sequence = "";
  for (let index = start + 1; index < lines.length; index += 1) {
    if (lines[index].startsWith(">")) break;
    sequence += lines[index].trim();
  }
  if (sequence === "") return null;
  return sequence.replace(/[a-z.\-]/g, "").toUpperCase();
}

/** `>` records in an a3m, which is the alignment's depth. */
export function countRecords(a3m) {
  let depth = 0;
  for (const line of String(a3m ?? "").split("\n")) {
    if (line.startsWith(">")) depth += 1;
  }
  return depth;
}

/**
 * Keep this sequence's a3m for the next ligand and the next visit.
 *
 * @returns {Promise<boolean>} whether it was stored; false is not an error
 */
export async function writeCachedMsa(sequence, a3m, options = {}) {
  const storage = options.caches ?? (cacheAvailable() ? caches : null);
  if (storage === null) return false;
  if (typeof a3m !== "string" || a3m.length === 0) return false;
  try {
    const cache = await storage.open(CACHE_NAME);
    await cache.put(keyUrl(sequenceHash(sequence)),
      new Response(a3m, { headers: { "content-type": "text/plain" } }));
    return true;
  } catch {
    // 🔴 A FULL DISK MUST NOT END A SCREEN. `put` rejects on a quota error, and
    // the alignment is still in memory for this run's remaining ligands.
    return false;
  }
}

/**
 * The alignment for a screen: cached, or searched once and then cached.
 *
 * @param {string} sequence
 * @param {{search: (sequence: string) => Promise<{a3m: string}|string>,
 *          onStatus?: (message: string) => void, caches?: CacheStorage,
 *          force?: boolean}} options
 * @returns {Promise<{a3m: string, depth: number, source: "cache"|"search",
 *                    hash: string}>}
 */
export async function alignmentForScreen(sequence, options) {
  const hash = sequenceHash(sequence);
  if (options.force !== true) {
    const hit = await readCachedMsa(sequence, options);
    if (hit !== null) {
      options.onStatus?.(`MSA from cache: ${hit.depth} rows`);
      return { ...hit, source: "cache" };
    }
  }
  options.onStatus?.("Searching for homologues (once for this target)");
  const searched = await options.search(sequence);
  const a3m = typeof searched === "string" ? searched : searched?.a3m;
  if (typeof a3m !== "string" || a3m.length === 0) {
    throw new Error("the MSA search returned no alignment");
  }
  const stored = await writeCachedMsa(sequence, a3m, options);
  const depth = countRecords(a3m);
  options.onStatus?.(`MSA: ${depth} rows`
    + (stored ? " (kept for the next run)" : ""));
  return { a3m, depth, source: "search", hash };
}
