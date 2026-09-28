# cython: language_level=3
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Fused per-row selection + aggregation for the canonical scorer (selection.py, aggregation.py).

Per row of the (n_head, n_prim) float32 score block, with the row resident in
L1 and rows spread over OpenMP threads:

    0. optional bipolar pole mask (selection.apply_pole_mask), when pairs are given:
       the row is first selected UNMASKED (steps 1-3, for the diagnostic only),
       then per pair each pole's top-3 mean (running insertion over its
       contiguous column block, summed in double in descending order) decides
       which pole is kept (tie: pole a); the other pole is set to -inf IN S
       ITSELF, so the caller's null draws see the masked row; the changed
       (headline, narrative) retention is counted against the unmasked set;
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
from libc.math cimport INFINITY
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


cdef inline void sort_int(int *v, int n) noexcept nogil:
    cdef int i, j, t
    for i in range(1, n):
        t = v[i]; j = i - 1
        while j >= 0 and v[j] > t:
            v[j + 1] = v[j]; j -= 1
        v[j + 1] = t


cdef inline double pole_score(const float *row, int start, int length) noexcept nogil:
    """Mean of the top-3 (all when fewer) of row[start:start+length], summed in double
    from the largest down: selection.pole_scores, bit for bit."""
    cdef float t0 = -INFINITY, t1 = -INFINITY, t2 = -INFINITY, v
    cdef int j, m = length if length < 3 else 3
    cdef double acc
    for j in range(start, start + length):
        v = row[j]
        if v > t0:
            t2 = t1; t1 = t0; t0 = v
        elif v > t1:
            t2 = t1; t1 = v
        elif v > t2:
            t2 = v
    acc = <double> t0
    if m > 1:
        acc = acc + <double> t1
    if m > 2:
        acc = acc + <double> t2
    return acc / m


cdef inline int select_row(const float *row, float *buf, int n_prim, int n_candidates,
                           float tau, int jump_min, float *kvals, int *kkeys,
                           long long *st, double *gap_sum) noexcept nogil:
    """Steps 1-3 for one row into (kvals, kkeys); returns the retained count. ``buf`` is
    left holding a permutation of the row (for the trim threshold). ``st`` (candidates,
    f0 survivors, retained pre-jump, jump trimmed) and ``gap_sum`` are incremented."""
    cdef int j, n_keep = 0, n_cand = 0, n_f0 = 0, cut
    cdef float kth, thr, v
    cdef double gap, best_gap
    for j in range(n_prim):
        buf[j] = row[j]
    kth = kth_largest(buf, n_prim, n_candidates)
    thr = kth if kth > tau else tau
    for j in range(n_prim):
        v = row[j]
        if v >= kth:
            n_cand = n_cand + 1
        if v >= tau:
            n_f0 = n_f0 + 1
        if v >= thr:
            kkeys[n_keep] = j
            kvals[n_keep] = v
            n_keep = n_keep + 1
    st[0] += n_cand
    st[1] += n_f0
    st[2] += n_keep
    if n_keep > 0 and jump_min >= 0 and n_keep > jump_min:
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
            st[3] += 1
            gap_sum[0] += best_gap
    return n_keep


def select_aggregate_rowwise(
    float[:, ::1] S,
    float tau,
    int n_candidates,
    int jump_min_candidates,
    const int[::1] prim_to_narr,
    int n_narr,
    bint use_median,
    bint want_primitive,
    int n_threads,
    int n_trim=0,
    pair_a_start=None,
    pair_a_len=None,
    pair_b_start=None,
    pair_b_len=None,
):
    """Steps 0-5 above. ``jump_min_candidates < 0`` disables the jump cut;
    ``n_trim == 0`` skips the trim threshold (returned as +inf); no pair arrays (or
    empty ones) disables the pole mask. With the mask, ``S`` is modified in place.

    Returns a dict with the reduced day accumulators for narratives
    (``narr_count/total/sumsq/peak``), for primitives when ``want_primitive``
    (``prim_*``, else None), the per-row ``trim_threshold`` (float32) and
    ``n_masked`` (int64, masked scores per row), and the block counts
    ``n_unassigned``, ``n_candidates``, ``n_f0_survivors``, ``n_retained_pre_jump``,
    ``n_retained``, ``n_jump_trimmed``, ``jump_gap_sum``, ``n_pole_masked``,
    ``n_mask_changed_retention``.
    """
    cdef int n_rows = S.shape[0], n_prim = S.shape[1]
    if n_candidates < 1 or n_candidates > n_prim:
        raise ValueError("n_candidates must be in [1, n_prim]")
    if n_trim < 0 or n_trim > n_prim:
        raise ValueError("n_trim must be in [0, n_prim]")
    if n_threads < 1:
        n_threads = 1
    cdef int n_prim_acc = n_prim if want_primitive else 1

    empty = np.empty(0, dtype=np.int32)
    arrs = [np.ascontiguousarray(a if a is not None else empty, dtype=np.int32)
            for a in (pair_a_start, pair_a_len, pair_b_start, pair_b_len)]
    if len({a.size for a in arrs}) != 1:
        raise ValueError("the four pair arrays must have the same length")
    cdef int n_pairs = arrs[0].size
    cdef const int[::1] pa0 = arrs[0]
    cdef const int[::1] pla = arrs[1]
    cdef const int[::1] pb0 = arrs[2]
    cdef const int[::1] plb = arrs[3]
    if arrs[0].size and ((arrs[1] < 1).any() or (arrs[3] < 1).any() or (arrs[0] < 0).any()
                         or (arrs[2] < 0).any() or (arrs[0] + arrs[1] > n_prim).any()
                         or (arrs[2] + arrs[3] > n_prim).any()):
        raise ValueError("a pole block lies outside the row")

    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_ncnt = np.zeros((n_threads, n_narr), dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_ntot = np.zeros((n_threads, n_narr), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_nsq = np.zeros((n_threads, n_narr), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_npk = np.full((n_threads, n_narr), -np.inf, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_pcnt = np.zeros((n_threads, n_prim_acc), dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_ptot = np.zeros((n_threads, n_prim_acc), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_psq = np.zeros((n_threads, n_prim_acc), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] t_ppk = np.full((n_threads, n_prim_acc), -np.inf, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_stats = np.zeros((n_threads, 8), dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] t_scratch = np.zeros((n_threads, 4), dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] t_gap = np.zeros(n_threads, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] t_gap_scratch = np.zeros(n_threads, dtype=np.float64)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] trim_out = np.full(n_rows, np.inf, dtype=np.float32)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] masked_out = np.zeros(n_rows, dtype=np.int64)

    cdef long long[:, ::1] ncnt = t_ncnt
    cdef double[:, ::1] ntot = t_ntot
    cdef double[:, ::1] nsq = t_nsq
    cdef double[:, ::1] npk = t_npk
    cdef long long[:, ::1] pcnt = t_pcnt
    cdef double[:, ::1] ptot = t_ptot
    cdef double[:, ::1] psq = t_psq
    cdef double[:, ::1] ppk = t_ppk
    # candidates, f0_survivors, retained_pre_jump, jump_trimmed, unassigned, retained,
    # pole_masked, mask_changed_retention
    cdef long long[:, ::1] stats = t_stats
    cdef long long[:, ::1] scratch = t_scratch        # the unmasked pass's discarded counts
    cdef double[::1] gap_sum = t_gap
    cdef double[::1] gap_scratch = t_gap_scratch
    cdef float[::1] trim_thr = trim_out
    cdef long long[::1] n_masked = masked_out

    cdef int i, j, p, tid, n_keep, n_u, n_d, run_start, run_len, nid, a0, la, b0, lb, lo, n_lose
    cdef double acc, dv
    cdef float *row
    cdef float *buf
    cdef float *kvals
    cdef int *kkeys
    cdef float *uvals
    cdef int *ukeys
    cdef int *mkeys
    cdef int *dnid

    with nogil, parallel(num_threads=n_threads):
        buf = <float *> malloc(n_prim * sizeof(float))
        kvals = <float *> malloc(n_prim * sizeof(float))
        kkeys = <int *> malloc(n_prim * sizeof(int))
        uvals = <float *> malloc(n_prim * sizeof(float))
        ukeys = <int *> malloc(n_prim * sizeof(int))
        mkeys = <int *> malloc(n_prim * sizeof(int))
        dnid = <int *> malloc(2 * n_prim * sizeof(int))
        tid = openmp.omp_get_thread_num()

        for i in prange(n_rows, schedule='static'):
            row = &S[i, 0]
            # --- 0. pole mask (unmasked selection first, for the diagnostic) --
            n_u = 0
            if n_pairs > 0:
                n_u = select_row(row, buf, n_prim, n_candidates, tau, jump_min_candidates,
                                 uvals, ukeys, &scratch[tid, 0], &gap_scratch[tid])
                for p in range(n_pairs):
                    a0 = pa0[p]; la = pla[p]; b0 = pb0[p]; lb = plb[p]
                    if pole_score(row, a0, la) >= pole_score(row, b0, lb):   # tie: pole a
                        lo = b0; n_lose = lb
                    else:
                        lo = a0; n_lose = la
                    for j in range(lo, lo + n_lose):
                        row[j] = -INFINITY
                    n_masked[i] += n_lose
                stats[tid, 6] += n_masked[i]

            # --- 1-3. candidates, tau, optional jump cut ----------------------
            n_keep = select_row(row, buf, n_prim, n_candidates, tau, jump_min_candidates,
                                kvals, kkeys, &stats[tid, 0], &gap_sum[tid])
            # --- 5. F0 trim threshold (n_trim-th largest), on the masked row ---
            if n_trim > 0:
                trim_thr[i] = kth_largest(buf, n_prim, n_trim)

            # --- 0b. (headline, narrative) pairs whose retained set changed ----
            if n_pairs > 0:
                for j in range(n_keep):
                    mkeys[j] = kkeys[j]
                sort_int(mkeys, n_keep)
                sort_int(ukeys, n_u)
                n_d = 0
                j = 0
                p = 0
                while j < n_keep or p < n_u:
                    if p >= n_u or (j < n_keep and mkeys[j] < ukeys[p]):
                        dnid[n_d] = prim_to_narr[mkeys[j]]; n_d = n_d + 1; j = j + 1
                    elif j >= n_keep or ukeys[p] < mkeys[j]:
                        dnid[n_d] = prim_to_narr[ukeys[p]]; n_d = n_d + 1; p = p + 1
                    else:
                        j = j + 1; p = p + 1
                if n_d > 0:
                    sort_int(dnid, n_d)
                    stats[tid, 7] += 1
                    for j in range(1, n_d):
                        if dnid[j] != dnid[j - 1]:
                            stats[tid, 7] += 1

            if n_keep == 0:
                stats[tid, 4] += 1
                continue
            stats[tid, 5] += n_keep

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

        free(buf); free(kvals); free(kkeys); free(uvals); free(ukeys); free(mkeys); free(dnid)

    out = {
        "narr_count": t_ncnt.sum(axis=0), "narr_total": t_ntot.sum(axis=0),
        "narr_sumsq": t_nsq.sum(axis=0), "narr_peak": t_npk.max(axis=0),
        "trim_threshold": trim_out, "n_masked": masked_out,
        "n_candidates": int(t_stats[:, 0].sum()), "n_f0_survivors": int(t_stats[:, 1].sum()),
        "n_retained_pre_jump": int(t_stats[:, 2].sum()),
        "n_jump_trimmed": int(t_stats[:, 3].sum()),
        "n_unassigned": int(t_stats[:, 4].sum()), "n_retained": int(t_stats[:, 5].sum()),
        "jump_gap_sum": float(t_gap.sum()),
        "n_pole_masked": int(t_stats[:, 6].sum()),
        "n_mask_changed_retention": int(t_stats[:, 7].sum()),
    }
    if want_primitive:
        out.update({
            "prim_count": t_pcnt.sum(axis=0), "prim_total": t_ptot.sum(axis=0),
            "prim_sumsq": t_psq.sum(axis=0), "prim_peak": t_ppk.max(axis=0),
        })
    else:
        out.update({"prim_count": None, "prim_total": None, "prim_sumsq": None, "prim_peak": None})
    return out
