/**
 * RDKit's refinement driver: a persistent `order` array, within-partition
 * stable sorts, and the activeset bookkeeping.
 *
 * 🔴 THIS REPLACES A GLOBAL RE-RANK, AND THE DIFFERENCE IS THE LAST FAULT.
 * This port previously re-sorted every atom each sweep and broke ties by ATOM
 * INDEX. RDKit sorts each partition's SLICE of a carried-forward `order` array
 * with a STABLE sort, so equal atoms keep their PREVIOUS relative order - which
 * is not index order after the first sweep. Measured: the initial partition was
 * right on 74/74 while the post-refinement class order was right on 63, which
 * is exactly that signature.
 *
 * `count[a]` is the size of a's class when a is the class's first atom in
 * `order`, and 0 otherwise - the same dual-purpose array RDKit uses, so
 * `hanoisort`'s habit of rewriting it during the sort is reproduced explicitly
 * here rather than hidden in a sort.
 */
export function createPartitions(n) {
  const p = {
    order: new Int32Array(n), count: new Int32Array(n),
    klass: new Int32Array(n), next: new Int32Array(n),
    changed: new Int32Array(n), touched: new Uint8Array(n), activeset: -1, n,
  };
  for (let i = 0; i < n; i += 1) { p.order[i] = i; }
  if (n > 0) p.count[p.order[0]] = n;
  return p;
}

/** `CreateSinglePartition` then a seed ordering by the base comparator. */
export function seedPartitions(p, compareBase) {
  // 🔴 RDKit STARTS FROM ONE CLASS and lets the first refinement (with
  // `df_useNbrs` false) split it by basecomp. Seeding the order array by the
  // base comparator and letting refinement take it from there is the same
  // thing, and it keeps the stable-sort invariant from the first sweep.
  const slice = Array.from(p.order);
  slice.sort((a, b) => compareBase(a, b) || a - b);
  for (let i = 0; i < slice.length; i += 1) p.order[i] = slice[i];
  p.count.fill(0);
  for (let i = 0; i < slice.length;) {
    let run = 1;
    while (i + run < slice.length
           && compareBase(slice[i], slice[i + run]) === 0) run += 1;
    p.count[slice[i]] = run;
    for (let k = 0; k < run; k += 1) p.klass[slice[i + k]] = i;
    i += run;
  }
  return p;
}

export function activatePartitions(p) {
  p.activeset = -1;
  p.next.fill(-2);
  let i = 0;
  if (p.n === 0) return p;
  do {
    const j = p.order[i];
    if (p.count[j] > 1) {
      p.next[j] = p.activeset;
      p.activeset = j;
      i += p.count[j];
    } else {
      i += 1;
    }
  } while (i < p.n);
  for (let k = 0; k < p.n; k += 1) p.changed[p.order[k]] = 1;
  return p;
}

/** `RefinePartitions`. `compare(a, b)` is the functor; `nbrIds` is per atom. */
export function refinePartitions(p, compare, mode, nbrIds, onPop) {
  let symclass = 0;
  while (p.activeset !== -1) {
    const partition = p.activeset;
    p.activeset = p.next[partition];
    p.next[partition] = -2;

    const len = p.count[partition];
    const offset = p.klass[partition];
    // 🔴 A STABLE SORT OF THE SLICE, with NO index tie-break: equal atoms must
    // keep the order the previous sweep left them in. Array.prototype.sort is
    // stable (ES2019), so passing no fallback comparison is what preserves it.
    const slice = Array.from(p.order.subarray(offset, offset + len));
    // 🔴 THE POP LOG MIRRORS RDKit'S, so the two SCHEDULES can be diffed. The
    // comparator and every key now agree; what was left was the ORDER
    // partitions are refined in, and a sequence of pops is the only view of
    // that.
    onPop?.(partition, offset, len, slice.slice());
    slice.sort((a, b) => compare(a, b));
    for (let k = 0; k < len; k += 1) p.order[offset + k] = slice[k];

    // ...and the sort rewrites `count` with the new sub-class sizes, which is
    // what hanoisort does on the way through.
    for (let k = 0; k < len;) {
      let run = 1;
      while (k + run < len && compare(slice[k], slice[k + run]) === 0) run += 1;
      p.count[slice[k]] = run;
      for (let m = 1; m < run; m += 1) p.count[slice[k + m]] = 0;
      k += run;
    }
    for (let k = 0; k < len; k += 1) p.changed[slice[k]] = 0;

    // 🔴 THE FIRST SUB-CLASS KEEPS THE OLD CLASS NUMBER. The loop starts at
    // `count[start[0]]`, so only the sub-classes after the first are renumbered
    // - to their POSITION in the order array, `offset + i`.
    let index = slice[0];
    for (let i = p.count[index]; i < len; i += 1) {
      index = slice[i];
      if (p.count[index]) symclass = offset + i;
      p.klass[index] = symclass;
      for (const nbr of nbrIds[index]) p.changed[nbr] = 1;
    }

    if (mode) {
      index = slice[0];
      for (let i = p.count[index]; i < len; i += 1) {
        index = slice[i];
        for (const nbr of nbrIds[index]) p.touched[p.klass[nbr]] = 1;
      }
      for (let ii = 0; ii < p.n; ii += 1) {
        if (!p.touched[ii]) continue;
        const npart = p.order[ii];
        if (p.count[npart] > 1 && p.next[npart] === -2) {
          p.next[npart] = p.activeset;
          p.activeset = npart;
        }
        p.touched[ii] = 0;
      }
    }
  }
  return p;
}

/** `BreakTies`: the class's LAST atom takes the class's HIGHEST position. */
export function breakTies(p, compare, mode, nbrIds) {
  for (let i = 0; i < p.n; i += 1) {
    const partition = p.order[i];
    const oldPart = p.klass[partition];
    while (p.count[partition] > 1) {
      const len = p.count[partition];
      const offset = p.klass[partition] + len - 1;
      const index = p.order[offset];
      p.klass[index] = offset;
      p.count[partition] = len - 1;
      p.count[index] = 1;
      if (nbrIds[index].length >= 1) {
        for (const nbr of nbrIds[index]) {
          p.touched[p.klass[nbr]] = 1;
          p.changed[nbr] = 1;
        }
        for (let ii = 0; ii < p.n; ii += 1) {
          if (!p.touched[ii]) continue;
          const npart = p.order[ii];
          if (p.count[npart] > 1 && p.next[npart] === -2) {
            p.next[npart] = p.activeset;
            p.activeset = npart;
          }
          p.touched[ii] = 0;
        }
        refinePartitions(p, compare, mode, nbrIds);
      }
    }
    // 🔴 AND THE LOOP RE-EXAMINES THE SAME POSITION WHEN THE CLASS MOVED.
    // `if (atoms[partition].index != oldPart) i -= 1;` - a split can shift what
    // sits at position i, and skipping it would leave a tie unbroken.
    if (p.klass[partition] !== oldPart) i -= 1;
  }
  return p;
}
