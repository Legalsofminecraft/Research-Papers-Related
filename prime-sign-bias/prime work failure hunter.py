import numpy as np
from multiprocessing import Pool, cpu_count
import math
import time
from datetime import datetime, timedelta

# ================= CONFIG =================

LIMIT = 5_000_000_000_00
CHUNK_SIZE = 2_800_000
OFFSET_LIMIT = 10000
PRINT_EVERY = 5_000_000
VECTOR_BATCH = 10000
LOG_FILE = "prime_hunt.log"

# ==========================================

SMALL_PRIMES = None
OFFSETS = None


class Logger:
    def __init__(self,path):
        self.f=open(path,'w',buffering=1)

    def log(self,msg,prefix="INFO"):
        ts=datetime.now().strftime("%H:%M:%S")
        line=f"[{ts}] [{prefix}] {msg}"
        print(line,flush=True)
        self.f.write(line+"\n")

    def close(self):
        self.f.close()


def init_worker(sp,offs):
    global SMALL_PRIMES,OFFSETS
    SMALL_PRIMES=sp
    OFFSETS=offs


def small_sieve(limit):

    sieve=np.ones(limit+1,dtype=np.bool_)
    sieve[:2]=False

    root=int(limit**0.5)

    for p in range(2,root+1):
        if sieve[p]:
            sieve[p*p::p]=False

    return np.where(sieve)[0].astype(np.int64)


def segmented_sieve_odd(lo,hi):

    lo_odd=lo|1

    if lo_odd>=hi:
        return np.array([],dtype=np.bool_),lo_odd

    size=((hi-lo_odd)+1)//2

    sieve=np.ones(size,dtype=np.bool_)

    if lo_odd==1:
        sieve[0]=False

    for p in SMALL_PRIMES:

        p=int(p)

        if p==2:
            continue

        if p*p>=hi:
            break

        start=max(
            p*p,
            lo_odd+((p-(lo_odd%p))%p)
        )

        if start%2==0:
            start+=p

        idx=(start-lo_odd)//2

        sieve[idx::p]=False

    return sieve,lo_odd


def process_chunk(chunk):

    lo,hi=chunk

    src_sieve,src_lo=segmented_sieve_odd(lo,hi)

    src_primes=(
        np.where(src_sieve)[0]*2+src_lo
    ).astype(np.int64)

    src_primes=src_primes[src_primes>=3]

    total=len(src_primes)

    if total==0:
        return {
            "hi":hi,
            "total":0,
            "failures":[]
        }

    max_k=int(OFFSETS[-1])

    check_lo=max(
        3,
        (2*lo-max_k)|1
    )

    check_hi=2*(hi-1)+max_k+1

    chk_sieve,chk_lo=segmented_sieve_odd(
        check_lo,
        check_hi
    )

    failures=[]

    for start in range(
        0,
        total,
        VECTOR_BATCH
    ):

        batch=src_primes[
            start:start+VECTOR_BATCH
        ]

        found=np.zeros(
            len(batch),
            dtype=bool
        )

        d=2*batch

        for k in OFFSETS:

            plus=d+k
            minus=d-k

            plus_idx=(
                plus-chk_lo
            )//2

            minus_idx=(
                minus-chk_lo
            )//2

            valid_plus=(
                (plus<check_hi)
            )

            valid_minus=(
                (minus>1)
                &
                (minus>=chk_lo)
            )

            ok=np.zeros(
                len(batch),
                dtype=bool
            )

            ok[valid_plus] |= chk_sieve[
                plus_idx[valid_plus]
            ]

            ok[valid_minus] |= chk_sieve[
                minus_idx[valid_minus]
            ]

            found|=ok

            if np.all(found):
                break

        failures.extend(
            batch[~found].tolist()
        )

    return {
        "hi":hi,
        "total":total,
        "failures":failures
    }


def gen_chunks():

    lo=2

    while lo<LIMIT:

        yield(
            lo,
            min(
                lo+CHUNK_SIZE,
                LIMIT
            )
        )

        lo+=CHUNK_SIZE


if __name__=="__main__":

    log=Logger(LOG_FILE)

    cores=max(
        1,
        min(cpu_count()-2,6)
    )

    sqrt_lim=(
        math.isqrt(
            2*LIMIT+OFFSET_LIMIT
        )+1000
    )

    sp=small_sieve(
        sqrt_lim
    )

    offs=small_sieve(
        OFFSET_LIMIT
    )

    offs=offs[
        offs>=3
    ]

    total=0
    last_print=0
    highest_hi=0
    start=time.time()

    with Pool(
        cores,
        initializer=init_worker,
        initargs=(sp,offs)
    ) as pool:

        for result in pool.imap_unordered(
            process_chunk,
            gen_chunks(),
            chunksize=1
        ):

            total+=result["total"]

            highest_hi=max(
                highest_hi,
                result["hi"]
            )

            if total-last_print>=PRINT_EVERY:

                elapsed=time.time()-start

                rate=total/max(
                    elapsed,
                    1
                )

                pct=(
                    highest_hi/LIMIT
                )*100

                log.log(
                    f"Progress {pct:5.2f}% | "
                    f"Primes {total:,} | "
                    f"Rate {rate:,.0f}/s"
                )

                last_print=total

    elapsed=time.time()-start

    log.log(
        f"BOUNDED CONJECTURE HOLDS "
        f"(offset≤{OFFSET_LIMIT}) "
        f"up to {LIMIT:,}"
    )

    log.log(
        f"Time: {elapsed:.1f}s"
    )

    log.close()
