"""
PRIME OFFSET RHYTHM HUNTER
===========================
Hypothesis: p mod 12 predicts the sign of the first working prime offset.
Specifically:
    p mod 12 = 7  → almost always +q (positive offset)
    p mod 12 = 11 → almost always -q (negative offset)

This script:
1. Tests all primes up to LIMIT
2. Records the sign (+/-) and value of the first working prime offset
3. Analyses whether p mod 12 (and other moduli) predicts the sign
4. Looks for any deeper rhythmic patterns

Runtime: ~2-4 hours for LIMIT = 10_000_000_000
"""

import math
from sympy import isprime
from collections import Counter, defaultdict
import time
import json
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
LIMIT         = 10_000_000_000   # 10 billion — adjust if needed
OFFSET_LIMIT  = 1000             # max prime offset to try
PRINT_EVERY   = 5_000_000        # progress heartbeat
SAVE_EVERY    = 50_000_000       # save stats to file this often
RESULTS_FILE  = "rhythm_results.json"
LOG_FILE      = "rhythm_hunt.log"
SEG_SIZE      = 2 ** 19          # ~512 K — fits in L2 cache
# ─────────────────────────────────────────────────────────────────────────────


# ── SIEVE HELPERS ─────────────────────────────────────────────────────────────

def small_primes_up_to(n: int) -> list:
    """Plain Eratosthenes sieve returning all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i, v in enumerate(sieve) if v]


class SegmentedSieveIterator:
    """
    Yields every prime in [2, limit] in ascending order.

    Strategy:
      Phase 1 – base sieve covers [2 .. isqrt(limit)].
                All base primes are yielded directly.
      Phase 2 – segmented sieve covers (isqrt(limit) .. limit].
                The first segment starts at base_ceil+1 (not rounded to
                SEG_SIZE) so there is never a gap between the two phases.
                Subsequent segments advance by SEG_SIZE.
    """

    def __init__(self, limit: int, seg_size: int = SEG_SIZE):
        self.limit    = limit
        self.seg_size = seg_size
        self.base     = small_primes_up_to(int(math.isqrt(limit)))
        self._gen     = self._generate()

    def _generate(self):
        # Phase 1: emit base primes
        for p in self.base:
            if p <= self.limit:
                yield p

        # Phase 2: segmented sweep from (isqrt(limit)+1) to limit
        base_ceil = int(math.isqrt(self.limit))
        low = base_ceil + 1          # start immediately after base region
        # no rounding — that's what caused the original gap

        while low <= self.limit:
            high  = min(low + self.seg_size - 1, self.limit)
            size  = high - low + 1
            sieve = bytearray([1]) * size

            for p in self.base:
                if p * p > high:
                    break
                # smallest multiple of p that is >= low
                start = ((low + p - 1) // p) * p
                if start == p:        # don't cross out p itself
                    start += p
                if start <= high:
                    sieve[start - low :: p] = bytearray(
                        len(sieve[start - low :: p])
                    )

            for i, v in enumerate(sieve):
                if v and low + i > 1:
                    yield low + i

            low += self.seg_size

    def next_prime(self):
        """Returns the next prime, or None when exhausted."""
        return next(self._gen, None)


def get_offsets(limit: int) -> list:
    """Return sorted list of all primes < limit."""
    it = SegmentedSieveIterator(limit - 1)   # -1 so we include limit-1 if prime
    offs = []
    while True:
        p = it.next_prime()
        if p is None:
            break
        offs.append(p)
    return offs


# ── LOGGING / SAVING ──────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path):
        self.f = open(path, 'w', buffering=1, encoding='utf-8')

    def log(self, msg, prefix="INFO"):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{prefix}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")

    def close(self):
        self.f.close()


def save_stats(stats, path):
    out = {}
    for key, val in stats.items():
        if isinstance(val, Counter):
            out[key] = {str(k): v for k, v in val.items()}
        elif isinstance(val, dict):
            out[key] = {
                str(k): ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in val.items()
            }
        else:
            out[key] = val
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


# ── QUICK SELF-TEST ───────────────────────────────────────────────────────────

def _selftest():
    """Verify prime counts against known values before the main run."""
    known = {
        100:       25,
        1_000:    168,
        10_000:  1229,
        100_000: 9592,
        1_000_000: 78498,
    }
    for n, expected in known.items():
        it    = SegmentedSieveIterator(n)
        count = 0
        while True:
            p = it.next_prime()
            if p is None:
                break
            count += 1
        if count != expected:
            raise RuntimeError(
                f"Sieve self-test FAILED for n={n}: "
                f"got {count} primes, expected {expected}"
            )
    print("[SELFTEST] Segmented sieve counts verified up to 1,000,000. OK.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _selftest()

    log = Logger(LOG_FILE)
    log.log("=" * 60)
    log.log("PRIME OFFSET RHYTHM HUNTER")
    log.log("=" * 60)
    log.log(f"Limit:        {LIMIT:,}")
    log.log(f"Offset limit: {OFFSET_LIMIT}")
    log.log(f"Hypothesis:   p mod 12 = 7  -> +offset | p mod 12 = 11 -> -offset")
    log.log("=" * 60)

    offsets = get_offsets(OFFSET_LIMIT)
    log.log(f"Loaded {len(offsets)} prime offsets (2 .. {offsets[-1]})")

    # ── STATS ACCUMULATORS ────────────────────────────────────────────────────

    total      = 0
    start      = time.time()
    last_print = 0
    last_save  = 0

    MODULI = [6, 12, 24, 30]

    sign_by_mod      = {m: defaultdict(Counter) for m in MODULI}
    offset_by_mod12  = defaultdict(lambda: defaultdict(Counter))
    sequence_sample  = []
    MAX_SEQUENCE     = 200_000

    run_counts    = Counter()
    current_run   = 0
    current_sign  = None

    transition_matrix = defaultdict(Counter)
    last_state        = None

    failures = []

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────

    prime_iter = SegmentedSieveIterator(LIMIT)

    while True:
        p = prime_iter.next_prime()
        if p is None:
            break

        total += 1
        d          = 2 * p
        found_k    = None
        found_sign = None

        for k in offsets:
            if isprime(d + k):
                found_k    = k
                found_sign = '+'
                break
            val = d - k
            if val > 1 and isprime(val):
                found_k    = k
                found_sign = '-'
                break

        if found_k is None:
            failures.append(p)
            log.log(f"FAILURE at p={p:,}", prefix="CRIT")
            continue

        # ── ACCUMULATE STATS ──────────────────────────────────────────────────

        for m in MODULI:
            sign_by_mod[m][p % m][found_sign] += 1

        r12 = p % 12
        offset_by_mod12[r12][found_k][found_sign] += 1

        if len(sequence_sample) < MAX_SEQUENCE:
            sequence_sample.append((p, found_k, found_sign))

        if found_sign == current_sign:
            current_run += 1
        else:
            if current_sign is not None:
                run_counts[current_run] += 1
            current_run  = 1
            current_sign = found_sign

        state = (r12, found_sign)
        if last_state is not None:
            transition_matrix[last_state][state] += 1
        last_state = state

        # ── PROGRESS ──────────────────────────────────────────────────────────

        if total - last_print >= PRINT_EVERY:
            elapsed = time.time() - start
            rate    = total / elapsed
            pct     = p / LIMIT * 100
            log.log(
                f"Progress: {pct:5.2f}% | "
                f"Primes: {total:>12,} | "
                f"Rate: {rate:>9,.0f}/s | "
                f"Failures: {len(failures)}"
            )
            log.log("  Mod 12 sign bias (+ means +offset favoured):")
            for r in sorted(sign_by_mod[12]):
                c   = sign_by_mod[12][r]
                tot = c['+'] + c['-']
                if tot:
                    log.log(f"    p%12={r:>2}: +{c['+']} / -{c['-']}  ({c['+']/tot*100:.1f}% positive)")
            last_print = total

        if total - last_save >= SAVE_EVERY:
            save_stats({
                "total_primes":          total,
                "limit":                 LIMIT,
                "failures":              failures[:20],
                "sign_by_mod6":          sign_by_mod[6],
                "sign_by_mod12":         sign_by_mod[12],
                "sign_by_mod24":         sign_by_mod[24],
                "sign_by_mod30":         sign_by_mod[30],
                "run_counts":            run_counts,
                "top_offset_by_mod12":   {str(r): dict(offset_by_mod12[r]) for r in offset_by_mod12},
            }, RESULTS_FILE)
            last_save = total

    # Final run flush
    if current_sign is not None:
        run_counts[current_run] += 1

    elapsed = time.time() - start

    # ── FINAL REPORT ──────────────────────────────────────────────────────────

    log.log("=" * 60)
    log.log("COMPLETE")
    log.log("=" * 60)
    log.log(f"Primes tested: {total:,}")
    log.log(f"Failures:      {len(failures)}")
    log.log(f"Time:          {elapsed:.1f}s  ({elapsed/3600:.2f}h)")

    for mod_name, mod_val in [("MOD 6", 6), ("MOD 12", 12), ("MOD 24", 24)]:
        log.log(f"\n=== {mod_name} SIGN BIAS ===")
        for r in sorted(sign_by_mod[mod_val]):
            c   = sign_by_mod[mod_val][r]
            tot = c['+'] + c['-']
            if tot:
                log.log(f"  p%{mod_val}={r:>2}: +{c['+']} / -{c['-']}  ({c['+']/tot*100:.1f}% positive)")

    log.log("\n=== RUN LENGTH DISTRIBUTION ===")
    log.log("(How many consecutive primes get the same sign in a row?)")
    for length in sorted(run_counts)[:20]:
        log.log(f"  Run of {length:>3}: {run_counts[length]:>10,} times")

    log.log("\n=== MOD 12 TRANSITION MATRIX ===")
    for from_state in sorted(transition_matrix):
        top = transition_matrix[from_state].most_common(3)
        log.log(f"  From {from_state}: {top}")

    log.log("\n=== TOP OFFSET VALUES BY MOD 12 ===")
    for r in sorted(offset_by_mod12):
        top_offsets = sorted(
            offset_by_mod12[r].items(),
            key=lambda x: sum(x[1].values()), reverse=True
        )[:5]
        log.log(f"  p%12={r:>2}: {[(k, dict(v)) for k, v in top_offsets]}")

    save_stats({
        "total_primes":             total,
        "limit":                    LIMIT,
        "failures":                 failures,
        "sign_by_mod6":             sign_by_mod[6],
        "sign_by_mod12":            sign_by_mod[12],
        "sign_by_mod24":            sign_by_mod[24],
        "sign_by_mod30":            sign_by_mod[30],
        "run_counts":               run_counts,
        "sequence_sample_length":   len(sequence_sample),
        "transition_matrix":        {str(k): dict(v) for k, v in transition_matrix.items()},
    }, RESULTS_FILE)

    log.log(f"Results saved to {RESULTS_FILE}")
    log.close()
