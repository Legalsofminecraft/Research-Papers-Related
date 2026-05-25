"""
PRIME DOUBLING CONJECTURE TESTER — PRODUCTION ENGINE
+ Progress % tracking (chunk-based)
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import time
import math
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LIMIT        = 10_000_000_000
OFFSET_LIMIT = 1_000
CHUNK_SIZE   = 10_000_000
PRINT_EVERY  = 5_000_000
HARD_THRESH  = 13

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ─────────────────────────────────────────────
# SMALL SIEVE
# ─────────────────────────────────────────────

def small_sieve(n):
    sieve = np.ones(n // 2, dtype=np.bool_)
    root = int(math.isqrt(n))

    for i in range(3, root + 1, 2):
        if sieve[i // 2]:
            sieve[i * i // 2::i] = False

    return np.concatenate(([2], np.where(sieve[1:])[0] * 2 + 3))


# ─────────────────────────────────────────────
# SEGMENTED SIEVE
# ─────────────────────────────────────────────

def segmented_sieve(lo, hi, primes):
    if lo % 2 == 0:
        lo += 1

    size = (hi - lo) // 2 + 1
    sieve = np.ones(size, dtype=np.bool_)

    root = int(math.isqrt(hi))

    for p in primes[1:]:
        if p > root:
            break

        start = max(p * p, ((lo + p - 1) // p) * p)
        if start % 2 == 0:
            start += p

        sieve[(start - lo) // 2 :: p] = False

    return sieve, lo


# ─────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────

def process_chunk(args):
    lo, hi, primes, offsets, tracked, hard_thresh = args

    sieve, base = segmented_sieve(lo, hi, primes)
    primes_chunk = np.where(sieve)[0] * 2 + base

    max_k = offsets[-1]
    ext_hi = 2 * hi + max_k + 100

    ext_sieve, _ = segmented_sieve(3, ext_hi, primes)

    total = 0
    failures = []
    hard_cases = []

    max_offset = 0
    max_prime = 0

    offsets = np.array(offsets, dtype=np.int64)

    for p in primes_chunk:
        total += 1
        d = 2 * p

        found = None

        for k in offsets:
            a = d + k
            b = d - k

            if a > 2 and ext_sieve[(a - 3) // 2]:
                found = k
                break

            if b > 2 and ext_sieve[(b - 3) // 2]:
                found = k
                break

        if found is None:
            failures.append(int(p))
        else:
            if found > max_offset:
                max_offset = found
                max_prime = int(p)

            if found > hard_thresh:
                hard_cases.append((int(p), int(found)))

    return total, failures, hard_cases, max_offset, max_prime


# ─────────────────────────────────────────────
# CHUNKS
# ─────────────────────────────────────────────

def generate_chunks():
    for s in range(3, LIMIT, CHUNK_SIZE):
        yield (s, min(s + CHUNK_SIZE, LIMIT))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    log(f"Starting up to {LIMIT:,}")

    cores = cpu_count()
    log(f"CPU cores: {cores}")

    start_time = time.time()

    sqrt_lim = int(math.isqrt(2 * LIMIT)) + 1
    primes = small_sieve(sqrt_lim)
    offsets = small_sieve(OFFSET_LIMIT)[1:]
    tracked = {3, 5, 7, 11, 13}

    log(f"Primes loaded: {len(primes):,}")
    log(f"Offsets loaded: {len(offsets)}")

    # ─────────────────────────────────────────────
    # PREP CHUNKS + PROGRESS TRACKING
    # ─────────────────────────────────────────────

    chunks_list = list(generate_chunks())
    total_chunks = len(chunks_list)
    done_chunks = 0

    total_primes = 0
    failures = []
    hard_cases = []
    max_off = 0
    max_p = 0

    # ─────────────────────────────────────────────
    # PARALLEL EXECUTION
    # ─────────────────────────────────────────────

    with Pool(cores) as pool:

        for t, f, h, mo, mp in pool.imap_unordered(
            process_chunk,
            [
                (lo, hi, primes, offsets, tracked, HARD_THRESH)
                for lo, hi in chunks_list
            ],
            chunksize=4
        ):

            done_chunks += 1
            total_primes += t
            failures.extend(f)
            hard_cases.extend(h)

            if mo > max_off:
                max_off = mo
                max_p = mp
                log(f"★ NEW MAX OFFSET ±{max_off} at p={max_p:,}")

            # ─────────────────────────────────────────────
            # PROGRESS %
            # ─────────────────────────────────────────────

            progress = (done_chunks / total_chunks) * 100
            elapsed = time.time() - start_time
            rate = total_primes / elapsed if elapsed > 0 else 0
            eta = (elapsed / done_chunks) * (total_chunks - done_chunks) if done_chunks else 0

            log(
                f"{progress:6.2f}% | "
                f"{total_primes:,} primes | "
                f"{rate:,.0f} p/s | "
                f"ETA {eta/60:.1f} min | "
                f"max=±{max_off}"
            )

    # ─────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────

    elapsed = time.time() - start_time

    log("=" * 60)
    log(f"DONE in {elapsed:.2f}s ({elapsed/60:.1f} min)")
    log(f"Total primes: {total_primes:,}")
    log(f"Failures: {len(failures)}")
    log(f"Max offset: ±{max_off} at p={max_p:,}")
