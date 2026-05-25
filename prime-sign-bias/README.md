# Prime Sign Bias — Code and Logs

Source code for the paper **"A Sign Asymmetry in Minimal Goldbach Offsets for Doubled Primes"** (Sarthak Kumar, 2026).  
arXiv: [link once published]

---

## Files

| File | Purpose | Dataset in paper |
|------|---------|-----------------|
| `prime_work.py` | Full offset tracking with max-offset records and parallelised chunk processing | Tables 3 & 4 (max offset envelope, up to 2,449,836,167) |
| `prime_work_rhythm_checker.py` | Sign bias analysis — records σ(p) for each prime and accumulates counts by residue class mod 6, 12, 24, 30 | Tables 1 & 2 (sign bias and offset frequency, first 400M primes) |
| `prime_work_failure_hunter.py` | Existence-only verification — early-exit, memory-stateless, no offset tracking | Footnote 1 (500B existence verification) |

---

## Requirements

```
Python 3.12+
numpy
sympy          # used in prime_work_rhythm_checker.py only
```

Install with:

```bash
pip install numpy sympy
```

---

## How to Run

### 1. Sign bias + offset frequency (primary dataset)

```bash
python prime_work_rhythm_checker.py
```

- Tests all primes up to `LIMIT` (default `10_000_000_000`)
- Writes live progress to console and `rhythm_hunt.log`
- Saves running statistics to `rhythm_results.json` every 50M primes
- **Expected runtime:** ~2–4 hours on the reference hardware

### 2. Max offset tracking

```bash
python prime_work.py
```

- Tracks the maximum prime offset q(p) seen and records new records as they occur
- Parallelised across all available cores via `multiprocessing.Pool`
- `LIMIT` (default `10_000_000_000`) and `OFFSET_LIMIT` (default `1000`) are set at the top of the file
- **Expected runtime:** ~1–3 hours depending on core count

### 3. Existence-only verification (500B run)

```bash
python prime_work_failure_hunter.py
```

- Verifies existence of a valid prime offset for every prime up to `LIMIT` (default `500_000_000_000`)
- No offset tracking or sign recording — pure early-exit pass/fail
- Uses 6 cores by default (`cpu_count() - 2`, capped at 6)
- **Expected runtime:** several hours; scale `CHUNK_SIZE` and core count to your hardware

---

## Hardware Reference

All runs reported in the paper were performed on:

- **CPU:** Intel Core i5 (8 logical cores)
- **RAM:** 16 GB
- **OS:** Linux

---

## Configuration

Each script has a `CONFIG` block at the top. The key parameters are:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `LIMIT` | varies | Upper bound on primes tested |
| `OFFSET_LIMIT` | 1000–10000 | Maximum prime offset q to try before declaring a failure |
| `CHUNK_SIZE` | varies | Size of each sieve segment per worker |
| `PRINT_EVERY` | 5,000,000 | Progress log frequency (in primes) |

---

## Output

`prime_work_rhythm_checker.py` produces the data behind Tables 1 and 2 of the paper. The relevant fields in `rhythm_results.json` are:

- `sign_by_mod6` — sign counts by p mod 6 (Table 1)
- `sign_by_mod12`, `sign_by_mod24`, `sign_by_mod30` — finer residue breakdowns
- The offset frequency table (Table 2) is recovered from `top_offset_by_mod12` aggregated across all residue classes

---

## Notes

- `prime_work_rhythm_checker.py` uses `sympy.isprime` for candidate primality checking. For large-scale runs, replacing this with a segmented sieve lookup (as in the other two scripts) will give a significant speedup.
- The self-test in `prime_work_rhythm_checker.py` verifies prime counts against known values up to 1,000,000 before the main run begins. If it fails, the sieve has a bug.
- Zero failures were found in all runs. The conjecture holds within all tested ranges.
