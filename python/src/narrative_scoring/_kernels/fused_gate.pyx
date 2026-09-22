# cython: language_level=3
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Fused F0 floor + percentile cut + narrative aggregation, one pass, OpenMP.

The numpy implementation in ``spec_pipeline`` expresses steps 3-5 as several
full-matrix passes (`np.nonzero`, `np.argpartition`, a sort-based group-by).
Measured on an 8192 x 1508 block those run at 1.2-1.5 GB/s against a machine
that streams 27.5 GB/s -- they are branchy, data-dependent and single-threaded,
which is exactly what numpy cannot express well and C can.

This kernel does the same arithmetic per row, in registers, with the row
resident in L1 (1508 floats = 6 KB), parallel over rows:

    1. count the survivors of the F0 floor                    (one pass)
    2. per q, quickselect the k-th largest survivor            (O(n_surv))
    3. keep S[i, j] >= max(tau, threshold_q)                   (one pass)
    4. group the kept primitives by narrative, mean|median, and
       scatter into thread-private accumulators

Semantics are identical to the numpy path, including the tie rule: the cut is
`>= threshold`, so several primitives tied at the threshold are all kept even
when that exceeds k. Getting this wrong would silently change the panel, so it
is asserted against a dense reference in the test suite.
"""
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange, parallel
cimport openmp
from libc.math cimport ceil
from libc.stdlib cimport malloc, free
from libcpp.algorithm cimport nth_element

cnp.import_array()


cdef inline float kth_largest(float *buf, int n, int k) noexcept nogil:
    """k-th largest (1-indexed) of buf[0:n], mutating buf.

    std::nth_element is an O(n)-average introselect. A hand-rolled Hoare
    quickselect was wrong here by one position on certain duplicate-free inputs
    (caught by the dense-reference test on real data, row 4595 of a 2008-09
    block), so this defers to the standard library rather than re-deriving it.
    Ascending nth_element puts the k-th LARGEST at index n - k.
    """
    nth_element(buf, buf + (n - k), buf + n)
    return buf[n - k]


cdef inline void sort_pairs_by_key(int *keys, float *vals, int n) noexcept nogil:
    """Insertion sort on the kept set (tiny: typically 1-5, bounded by n_surv)."""
    cdef int i, j, ktmp
    cdef float vtmp
    for i in range(1, n):
        ktmp = keys[i]; vtmp = vals[i]; j = i - 1
        while j >= 0 and keys[j] > ktmp:
            keys[j + 1] = keys[j]; vals[j + 1] = vals[j]; j -= 1
        keys[j + 1] = ktmp; vals[j + 1] = vtmp


cdef inline float median_inplace(float *v, int n) noexcept nogil:
    cdef int i, j
    cdef float t
    for i in range(1, n):                      # runs are tiny; insertion sort
        t = v[i]; j = i - 1
        while j >= 0 and v[j] > t:
            v[j + 1] = v[j]; j -= 1
        v[j + 1] = t
    if n % 2: return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def gate_aggregate_rowwise(
    const float[:, ::1] S,
    double tau,
    const double[::1] qs,
    const int[::1] prim_to_narr,
    int n_narr,
    bint use_median,
    int n_threads,
):
    """Steps 3-5 of the spec for the row-wise percentile axis.

    Returns, per q in ``qs``: (count, total, peak, n_unassigned, n_kept,
    n_floored) with count/total/peak shaped (n_narr,) -- exactly what
    ``_DayAccumulator`` folds in.
    """
    cdef int n_rows = S.shape[0], n_prim = S.shape[1], n_q = qs.shape[0]
    if n_threads < 1:
        n_threads = 1

    # thread-private accumulators, reduced after the parallel region
    cdef cnp.ndarray[cnp.int64_t, ndim=3] t_count = np.zeros((n_threads, n_q, n_narr), dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] t_total = np.zeros((n_threads, n_q, n_narr), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] t_peak = np.full((n_threads, n_q, n_narr), -np.inf, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_unassigned = np.zeros((n_threads, n_q), dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_kept = np.zeros((n_threads, n_q), dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] t_floored = np.zeros(n_threads, dtype=np.int64)

    cdef long long[:, :, ::1] cnt = t_count
    cdef double[:, :, ::1] tot = t_total
    cdef double[:, :, ::1] pk = t_peak
    cdef long long[:, ::1] unass = t_unassigned
    cdef long long[:, ::1] kept_c = t_kept
    cdef long long[::1] floored = t_floored

    cdef int i, j, q, tid, n_surv, n_keep, k, run_len, nid, run_start
    cdef float thr, v
    cdef float *surv
    cdef float *kvals
    cdef int *kkeys
    cdef float *runbuf
    cdef double acc

    with nogil, parallel(num_threads=n_threads):
        surv = <float *> malloc(n_prim * sizeof(float))
        kvals = <float *> malloc(n_prim * sizeof(float))
        kkeys = <int *> malloc(n_prim * sizeof(int))
        runbuf = <float *> malloc(n_prim * sizeof(float))
        tid = openmp.omp_get_thread_num()

        for i in prange(n_rows, schedule='static'):
            # --- step 3: F0 floor, count survivors -------------------------
            n_surv = 0
            for j in range(n_prim):
                v = S[i, j]
                if v >= tau:
                    surv[n_surv] = v
                    n_surv = n_surv + 1
            floored[tid] += (n_prim - n_surv)

            if n_surv == 0:                       # stays fully unassigned
                for q in range(n_q):
                    unass[tid, q] += 1
                continue

            for q in range(n_q):
                # --- step 4: k-th largest survivor -------------------------
                # Parenthesised so no reassociation is even possible: the
                # numerator is exact for integral n_surv and q.
                k = <int> ceil((n_surv * (100.0 - qs[q])) / 100.0)
                if k < 1: k = 1
                if k > n_surv: k = n_surv
                for j in range(n_surv):
                    kvals[j] = surv[j]            # quickselect mutates
                thr = kth_largest(kvals, n_surv, k)

                # Two separate tests, matching numpy's `above & (S >= thr)`
                # EXACTLY, including the comparison precision: the F0 floor is
                # compared in double (tau is a Python float, so numpy upcasts S
                # to float64), while the threshold is a float32 value drawn from
                # the data and compared in float32. Collapsing these into one
                # float32 test flips scores within an ulp of tau.
                # Ties at the threshold are all kept, which is what `>=` does.
                n_keep = 0
                for j in range(n_prim):
                    v = S[i, j]
                    if v >= tau and v >= thr:
                        kkeys[n_keep] = prim_to_narr[j]
                        kvals[n_keep] = v
                        n_keep = n_keep + 1
                if n_keep == 0:
                    unass[tid, q] += 1
                    continue
                kept_c[tid, q] += n_keep

                # --- step 5: primitive -> narrative, per headline ----------
                sort_pairs_by_key(kkeys, kvals, n_keep)
                run_start = 0
                while run_start < n_keep:
                    nid = kkeys[run_start]
                    run_len = 1
                    while run_start + run_len < n_keep and kkeys[run_start + run_len] == nid:
                        run_len = run_len + 1
                    if use_median:
                        for j in range(run_len):
                            runbuf[j] = kvals[run_start + j]
                        v = median_inplace(runbuf, run_len)
                    else:
                        acc = 0.0
                        for j in range(run_len):
                            acc = acc + kvals[run_start + j]
                        v = <float> (acc / run_len)
                    cnt[tid, q, nid] += 1
                    tot[tid, q, nid] += v
                    if v > pk[tid, q, nid]:
                        pk[tid, q, nid] = v
                    run_start = run_start + run_len

        free(surv); free(kvals); free(kkeys); free(runbuf)

    out = []
    for q in range(n_q):
        peak = t_peak[:, q, :].max(axis=0)
        out.append((
            t_count[:, q, :].sum(axis=0),
            t_total[:, q, :].sum(axis=0),
            peak,
            int(t_unassigned[:, q].sum()),
            int(t_kept[:, q].sum()),
            int(t_floored.sum()),
        ))
    return out
