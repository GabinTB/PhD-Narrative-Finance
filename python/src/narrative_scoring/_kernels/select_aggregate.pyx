# cython: language_level=3
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Fused per-row selection + aggregation for the canonical scorer (selection.py, aggregation.py).

Per row of the (n_head, n_prim) float32 score block, with the row resident in
L1 and rows spread over OpenMP threads:

    1. k-th largest of the FULL row via std::nth_element (O(P), no full sort);
       k = n_candidates = ceil((1-q) * P) (config.n_candidates_for);
    2. retained iff s >= max(kth, tau)      -- q-candidates INTERSECT tau;
    3. optional jump cut: sort the retained (~16) scores descending, cut after
       the first largest positive gap, only when count > jump_min;
    4. per-primitive accumulators (optional) and, after grouping the retained
       primitives by narrative with mean|median, per-narrative accumulators:
       count, sum, sum of squares, max -- all in double;
    5. the F0 trim threshold of the row (n_trim-th largest), so the null
       draws can be subsampled without a second pass over the block.

Every comparison is float32 against float32, exactly as the numpy reference;
means are accumulated in double. Ties at the k-th boundary are all kept
(">="), matching the reference. Thread-private accumulators are reduced at
the end, so the result does not depend on the thread count.
"""
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange, parallel
cimport openmp
from libc.stdlib cimport malloc, free
from libcpp.algorithm cimport nth_element

cnp.import_array()


cdef inline float kth_largest(float *buf, int n, int k) noexcept nogil:
    """k-th largest (1-indexed) of buf[0:n]; mutates buf. Ascending nth_element -> index n-k."""
    nth_element(buf, buf + (n - k), buf + n)
    return buf[n - k]


cdef inline void sort_desc_pairs(float *vals, int *keys, int n) noexcept nogil:
    """Insertion sort by value descending, stable (tiny n)."""
    cdef int i, j, ktmp
    cdef float vtmp
    for i in range(1, n):
        vtmp = vals[i]; ktmp = keys[i]; j = i - 1
        while j >= 0 and vals[j] < vtmp:
            vals[j + 1] = vals[j]; keys[j + 1] = keys[j]; j -= 1
        vals[j + 1] = vtmp; keys[j + 1] = ktmp


cdef inline void sort_by_key_pairs(int *keys, float *vals, int n) noexcept nogil:
    """Insertion sort by key ascending, stable; ties keep value order (tiny n)."""
    cdef int i, j, ktmp
    cdef float vtmp
    for i in range(1, n):
        ktmp = keys[i]; vtmp = vals[i]; j = i - 1
        while j >= 0 and keys[j] > ktmp:
            keys[j + 1] = keys[j]; vals[j + 1] = vals[j]; j -= 1
        keys[j + 1] = ktmp; vals[j + 1] = vtmp


cdef inline double median_sorted_run(float *v, int n) noexcept nogil:
    """Median of a run already sorted ascending; the average of two in double."""
    if n % 2:
        return <double> v[n // 2]
    return 0.5 * (<double> v[n // 2 - 1] + <double> v[n // 2])


cdef inline void sort_asc(float *v, int n) noexcept nogil:
    cdef int i, j
    cdef float t
    for i in range(1, n):
        t = v[i]; j = i - 1
        while j >= 0 and v[j] > t:
            v[j + 1] = v[j]; j -= 1
        v[j + 1] = t


def select_aggregate_rowwise(
    const float[:, ::1] S,
    float tau,
    int n_candidates,
    int jump_min_candidates,
    const int[::1] prim_to_narr,
    int n_narr,
    bint use_median,
    bint want_primitive,
    int n_threads,
    int n_trim=0,
):
    """Steps 1-5 above. ``jump_min_candidates < 0`` disables the jump cut;
    ``n_trim == 0`` skips the trim threshold (returned as +inf).

    Returns a dict with the reduced day accumulators for narratives
    (``narr_count/total/sumsq/peak``), for primitives when ``want_primitive``
    (``prim_*``, else None), the per-row ``trim_threshold`` (float32) and the
    block counts ``n_unassigned``, ``n_candidates``, ``n_f0_survivors``,
    ``n_retained_pre_jump``, ``n_retained``, ``n_jump_trimmed``,
    ``jump_gap_sum``.
    """
    cdef int n_rows = S.shape[0], n_prim = S.shape[1]
    if n_candidates < 1 or n_candidates > n_prim:
        raise ValueError("n_candidates must be in [1, n_prim]")
    if n_trim < 0 or n_trim > n_prim:
        raise ValueError("n_trim must be in [0, n_prim]")
    if n_threads < 1:
        n_threads = 1
    cdef int n_prim_acc = n_prim if want_primitive else 1

    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_ncnt = np.zeros((n_threads, n_narr), dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_ntot = np.zeros((n_threads, n_narr), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_nsq = np.zeros((n_threads, n_narr), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_npk = np.full((n_threads, n_narr), -np.inf, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_pcnt = np.zeros((n_threads, n_prim_acc), dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_ptot = np.zeros((n_threads, n_prim_acc), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_psq = np.zeros((n_threads, n_prim_acc), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_ppk = np.full((n_threads, n_prim_acc), -np.inf, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_stats = np.zeros((n_threads, 6), dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] t_gap = np.zeros(n_threads, dtype=np.float64)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] trim_out = np.full(n_rows, np.inf, dtype=np.float32)

    cdef long long[:, ::1] ncnt = t_ncnt
    cdef double[:, ::1] ntot = t_ntot
    cdef double[:, ::1] nsq = t_nsq
    cdef double[:, ::1] npk = t_npk
    cdef long long[:, ::1] pcnt = t_pcnt
    cdef double[:, ::1] ptot = t_ptot
    cdef double[:, ::1] psq = t_psq
    cdef double[:, ::1] ppk = t_ppk
    # unassigned, candidates, f0_survivors, retained_pre_jump, retained, jump_trimmed
    cdef long long[:, ::1] stats = t_stats
    cdef double[::1] gap_sum = t_gap
    cdef float[::1] trim_thr = trim_out

    cdef int i, j, tid, n_keep, n_cand, n_f0, run_start, run_len, nid, cut
    cdef float kth, thr, v
    cdef double acc, dv, gap, best_gap      # gaps in double, as the numpy reference
    cdef float *buf
    cdef float *kvals
    cdef int *kkeys

    with nogil, parallel(num_threads=n_threads):
        buf = <float *> malloc(n_prim * sizeof(float))
        kvals = <float *> malloc(n_prim * sizeof(float))
        kkeys = <int *> malloc(n_prim * sizeof(int))
        tid = openmp.omp_get_thread_num()

        for i in prange(n_rows, schedule='static'):
            # --- 1. candidate threshold from the full row ------------------
            for j in range(n_prim):
                buf[j] = S[i, j]
            kth = kth_largest(buf, n_prim, n_candidates)
            thr = kth if kth > tau else tau
            # --- 5. F0 trim threshold (n_trim-th largest) -------------------
            if n_trim > 0:
                trim_thr[i] = kth_largest(buf, n_prim, n_trim)

            # --- 2. retained = candidates AND >= tau ------------------------
            n_keep = 0
            n_cand = 0
            n_f0 = 0
            for j in range(n_prim):
                v = S[i, j]
                if v >= kth:
                    n_cand = n_cand + 1
                if v >= tau:
                    n_f0 = n_f0 + 1
                if v >= thr:
                    kkeys[n_keep] = j
                    kvals[n_keep] = v
                    n_keep = n_keep + 1
            stats[tid, 1] += n_cand
            stats[tid, 2] += n_f0
            stats[tid, 3] += n_keep
            if n_keep == 0:
                stats[tid, 0] += 1
                continue

            # --- 3. optional jump cut ---------------------------------------
            if jump_min_candidates >= 0 and n_keep > jump_min_candidates:
                sort_desc_pairs(kvals, kkeys, n_keep)
                best_gap = 0.0
                cut = n_keep
                for j in range(n_keep - 1):
                    gap = <double> kvals[j] - <double> kvals[j + 1]
                    if gap > best_gap:            # strict: first occurrence wins
                        best_gap = gap
                        cut = j + 1
                if best_gap > 0.0 and cut < n_keep:
                    n_keep = cut
                    stats[tid, 5] += 1
                    gap_sum[tid] += best_gap
            stats[tid, 4] += n_keep

            # --- 4a. primitive-grain accumulators ---------------------------
            if want_primitive:
                for j in range(n_keep):
                    dv = <double> kvals[j]
                    pcnt[tid, kkeys[j]] += 1
                    ptot[tid, kkeys[j]] += dv
                    psq[tid, kkeys[j]] += dv * dv
                    if dv > ppk[tid, kkeys[j]]:
                        ppk[tid, kkeys[j]] = dv

            # --- 4b. primitive -> narrative, then narrative accumulators -----
            for j in range(n_keep):
                kkeys[j] = prim_to_narr[kkeys[j]]
            sort_by_key_pairs(kkeys, kvals, n_keep)
            run_start = 0
            while run_start < n_keep:
                nid = kkeys[run_start]
                run_len = 1
                while run_start + run_len < n_keep and kkeys[run_start + run_len] == nid:
                    run_len = run_len + 1
                if use_median:
                    sort_asc(kvals + run_start, run_len)
                    acc = median_sorted_run(kvals + run_start, run_len)
                else:
                    acc = 0.0
                    for j in range(run_len):
                        acc = acc + <double> kvals[run_start + j]
                    acc = acc / run_len
                ncnt[tid, nid] += 1
                ntot[tid, nid] += acc
                nsq[tid, nid] += acc * acc
                if acc > npk[tid, nid]:
                    npk[tid, nid] = acc
                run_start = run_start + run_len

        free(buf); free(kvals); free(kkeys)

    out = {
        "narr_count": t_ncnt.sum(axis=0), "narr_total": t_ntot.sum(axis=0),
        "narr_sumsq": t_nsq.sum(axis=0), "narr_peak": t_npk.max(axis=0),
        "trim_threshold": trim_out,
        "n_unassigned": int(t_stats[:, 0].sum()), "n_candidates": int(t_stats[:, 1].sum()),
        "n_f0_survivors": int(t_stats[:, 2].sum()),
        "n_retained_pre_jump": int(t_stats[:, 3].sum()),
        "n_retained": int(t_stats[:, 4].sum()), "n_jump_trimmed": int(t_stats[:, 5].sum()),
        "jump_gap_sum": float(t_gap.sum()),
    }
    if want_primitive:
        out.update({
            "prim_count": t_pcnt.sum(axis=0), "prim_total": t_ptot.sum(axis=0),
            "prim_sumsq": t_psq.sum(axis=0), "prim_peak": t_ppk.max(axis=0),
        })
    else:
        out.update({"prim_count": None, "prim_total": None, "prim_sumsq": None, "prim_peak": None})
    return out
