"""
PRIME OFFSET RHYTHM HUNTER  —  ULTRA FAST EDITION
=================================================
Optimizations over v1:
  1. Action List precomputation -> Completely skips invalid offset iterations.
  2. gmpy2.is_prime(n, 15) -> Fewer Miller-Rabin tests (safe for 64-bit ints).
  3. primesieve integration -> C++ prime generation (falls back to Python sieve).
  4. Streamlined local worker loop -> Zero tuple unpacking overhead in hot paths.

Setup:
    pip install gmpy2
    pip install primesieve  <-- HIGHLY recommended for 10B limit
"""

import math
import gmpy2
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
import time
import json
from datetime import datetime

try:
    import primesieve
    HAS_PRIMESIEVE = True
except ImportError:
    HAS_PRIMESIEVE = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
LIMIT        = 10_000_000_000
OFFSET_LIMIT = 1000
CHUNK_SIZE   = 50_000       # primes per worker batch
PRINT_EVERY  = 5_000_000
SAVE_EVERY   = 50_000_000
RESULTS_FILE = "rhythm_results.json"
LOG_FILE     = "rhythm_hunt.log"
SEG_SIZE     = 2 ** 19
# ─────────────────────────────────────────────────────────────────────────────

# ── PURE PYTHON SIEVE (FALLBACK) ──────────────────────────────────────────────

def small_primes_up_to(n):
    if n < 2: return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i, v in enumerate(sieve) if v]

class SegmentedSieveIterator:
    def __init__(self, limit, seg_size=SEG_SIZE):
        self.limit    = limit
        self.seg_size = seg_size
        self.base     = small_primes_up_to(int(math.isqrt(limit)))
        self._gen     = self._generate()

    def _generate(self):
        for p in self.base:
            if p <= self.limit:
                yield p
        base_ceil = int(math.isqrt(self.limit))
        low = base_ceil + 1
        while low <= self.limit:
            high  = min(low + self.seg_size - 1, self.limit)
            sieve = bytearray([1]) * (high - low + 1)
            for p in self.base:
                if p * p > high: break
                start = ((low + p - 1) // p) * p
                if start == p: start += p
                if start <= high:
                    sieve[start - low :: p] = bytearray(len(sieve[start - low :: p]))
            for i, v in enumerate(sieve):
                if v and low + i > 1:
                    yield low + i
            low += self.seg_size

    def next_prime(self):
        return next(self._gen, None)

def get_offsets(limit):
    if HAS_PRIMESIEVE:
        return primesieve.primes(limit)
    
    it = SegmentedSieveIterator(limit - 1)
    offs = []
    while True:
        p = it.next_prime()
        if p is None: break
        offs.append(p)
    return offs


# ── ACTION LIST PRE-FILTER ────────────────────────────────────────────────────
# Precomputes exactly WHICH offsets and signs are valid for a given (2p % 210).
# This completely eliminates iterating over invalid composite offsets in the worker.

_FILTER_MOD = 210
_SMALL_DIVS = (2, 3, 5, 7)

def build_action_lists(offsets):
    """
    Returns a list of 210 lists. 
    actions[r] = list of (k, is_plus_boolean) 
    """
    actions = [[] for _ in range(_FILTER_MOD)]
    for r in range(_FILTER_MOD):
        for k in offsets:
            # Check if 2p + k is potentially prime
            if all((r + k) % d != 0 for d in _SMALL_DIVS):
                actions[r].append((k, True))
            
            # Check if 2p - k is potentially prime
            if all((r - k) % d != 0 for d in _SMALL_DIVS):
                actions[r].append((k, False))
    return actions


# ── WORKER ────────────────────────────────────────────────────────────────────

_worker_actions = None

def _worker_init(actions):
    global _worker_actions
    _worker_actions = actions

def _process_chunk(prime_chunk):
    """
    Process a list of primes. Returns list of (p, found_k, found_sign).
    """
    actions = _worker_actions
    isp     = gmpy2.is_prime
    results = []
    
    for p in prime_chunk:
        d = 2 * p
        r210 = d % _FILTER_MOD
        found_k, found_sign = None, None

        # Loop ONLY through offsets guaranteed to bypass the 2,3,5,7 filter
        for k, is_plus in actions[r210]:
            if is_plus:
                if isp(d + k, 15):  # Reduced from 25 to 15 for speed
                    found_k, found_sign = k, '+'
                    break
            else:
                val = d - k
                if val > 1 and isp(val, 15):
                    found_k, found_sign = k, '-'
                    break

        results.append((p, found_k, found_sign))
    return results


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
                str(k): ({str(kk): vv for kk, vv in v.items()}
                         if isinstance(v, dict) else v)
                for k, v in val.items()
            }
        else:
            out[key] = val
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


# ── CHUNK GENERATOR ──────────────────────────────────────────────────────────

def prime_chunks(limit, chunk_size):
    """Yield sorted lists of primes in chunks of chunk_size."""
    chunk = []
    if HAS_PRIMESIEVE:
        it = primesieve.Iterator()
        p = it.next_prime()
        while p <= limit:
            chunk.append(p)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
            p = it.next_prime()
    else:
        it = SegmentedSieveIterator(limit)
        while True:
            p = it.next_prime()
            if p is None: break
            chunk.append(p)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
                
    if chunk: yield chunk


# ── SELF-TEST ─────────────────────────────────────────────────────────────────

def _selftest():
    # Verify action list doesn't skip valid primes
    small_offs = get_offsets(50)
    actions = build_action_lists(small_offs)
    errors = 0
    for p in range(3, 10000, 2):
        if not gmpy2.is_prime(p): continue
        d = 2 * p
        r = d % _FILTER_MOD
        
        # Fast way
        fast_k = None
        for k, is_plus in actions[r]:
            val = d + k if is_plus else d - k
            if val > 1 and gmpy2.is_prime(val, 25):
                fast_k = k
                break
                
        # Naive way (what your v1 effectively did)
        naive_k = None
        for k in small_offs:
            if gmpy2.is_prime(d + k, 25):
                naive_k = k
                break
            if d - k > 1 and gmpy2.is_prime(d - k, 25):
                naive_k = k
                break
                
        if fast_k != naive_k:
            errors += 1
            
    if errors:
        raise RuntimeError(f"Action list logic mismatch: {errors} errors")
    print("[SELFTEST] Action list OK.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _selftest()

    log = Logger(LOG_FILE)
    log.log("=" * 60)
    log.log("PRIME OFFSET RHYTHM HUNTER  —  ULTRA FAST EDITION")
    log.log("=" * 60)
    log.log(f"Limit:        {LIMIT:,}")
    log.log(f"Offset limit: {OFFSET_LIMIT}")
    log.log(f"Chunk size:   {CHUNK_SIZE:,} primes/worker batch")
    log.log(f"Sieve Engine: {'primesieve (C++)' if HAS_PRIMESIEVE else 'Pure Python Segmented Sieve'}")
    cores = cpu_count()
    log.log(f"CPU cores:    {cores}")
    log.log("=" * 60)

    offsets = get_offsets(OFFSET_LIMIT)
    log.log(f"Loaded {len(offsets)} prime offsets (2 .. {offsets[-1]})")

    log.log("Building Action List filter (mod 210)...")
    actions = build_action_lists(offsets)
    log.log("Action List ready.")

    # ── STATS ─────────────────────────────────────────────────────────────────
    MODULI = [6, 12, 24, 30]
    sign_by_mod      = {m: defaultdict(Counter) for m in MODULI}
    offset_by_mod12  = defaultdict(lambda: defaultdict(Counter))
    sequence_sample  = []
    MAX_SEQUENCE     = 200_000
    run_counts       = Counter()
    current_run      = 0
    current_sign     = None
    transition_matrix= defaultdict(Counter)
    last_state       = None
    failures         = []
    total            = 0
    start            = time.time()
    last_print       = 0
    last_save        = 0

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    with Pool(
        processes=cores,
        initializer=_worker_init,
        initargs=(actions,)
    ) as pool:

        for chunk_results in pool.imap(_process_chunk, prime_chunks(LIMIT, CHUNK_SIZE)):
            for p, found_k, found_sign in chunk_results:
                total += 1

                if found_k is None:
                    failures.append(p)
                    log.log(f"FAILURE at p={p:,}", prefix="CRIT")
                    continue

                # ── accumulate stats ──────────────────────────────────────────
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

                # ── progress ──────────────────────────────────────────────────
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
                    last_print = total

                if total - last_save >= SAVE_EVERY:
                    save_stats({
                        "total_primes":        total,
                        "limit":               LIMIT,
                        "failures":            failures[:20],
                        "sign_by_mod6":        sign_by_mod[6],
                        "sign_by_mod12":       sign_by_mod[12],
                        "sign_by_mod24":       sign_by_mod[24],
                        "sign_by_mod30":       sign_by_mod[30],
                        "run_counts":          run_counts,
                        "top_offset_by_mod12": {str(r): dict(offset_by_mod12[r])
                                                for r in offset_by_mod12},
                    }, RESULTS_FILE)
                    last_save = total

    # final run flush
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
                log.log(f"  p%{mod_val}={r:>2}: +{c['+']} / -{c['-']}  ({c['+']/tot*100:.1f}%)")

    log.log("\n=== RUN LENGTH DISTRIBUTION ===")
    for length in sorted(run_counts)[:20]:
        log.log(f"  Run of {length:>3}: {run_counts[length]:>10,} times")

    log.log("\n=== MOD 12 TRANSITION MATRIX ===")
    for from_state in sorted(transition_matrix):
        top = transition_matrix[from_state].most_common(3)
        log.log(f"  From {from_state}: {top}")

    save_stats({
        "total_primes":           total,
        "limit":                  LIMIT,
        "failures":               failures,
        "sign_by_mod6":           sign_by_mod[6],
        "sign_by_mod12":          sign_by_mod[12],
        "sign_by_mod24":          sign_by_mod[24],
        "sign_by_mod30":          sign_by_mod[30],
        "run_counts":             run_counts,
        "sequence_sample_length": len(sequence_sample),
        "transition_matrix":      {str(k): dict(v) for k, v in transition_matrix.items()},
    }, RESULTS_FILE)

    log.log(f"Results saved to {RESULTS_FILE}")
    log.close()
