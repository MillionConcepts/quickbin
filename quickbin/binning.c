#include "binning.h"

#define PYARRAY_AS_DOUBLES(PYARG) ((double *) PyArray_DATA(PYARG))
#define PYARRAY_AS_LONGS(PYARG) ((long *) PyArray_DATA(PYARG))

static inline void
assign_countsum(double *count, double *sum, long index, double val) {
    count[index] += 1;
    sum[index] += val;
}

static inline void
populate_meanarr(
    const long size, const double *count, const double *sum, double *mean
) {
    for (long ix = 0; ix < size; ix++) {
        if (count[ix] == 0) mean[ix] = NAN;
        else mean[ix] = sum[ix] / count[ix];
    }
}

static inline double
stdev(const double count, const double sum, const double sqr) {
    return sqrt((sqr * count - sum * sum) / (count * count));
}

static inline void
populate_stdarr(
    const long size, const double *count, const double *sum,
    const double *sqr, double *std
) {
    for (long ix = 0; ix < size; ix++) {
        if (count[ix] == 0) std[ix] = NAN;
        else std[ix] = stdev(count[ix], sum[ix], sqr[ix]);
    }
}

static inline int
doublecomp(const void *a, const void *b) {
    double *aval = (double *) a, *bval = (double *) b;
    if (*aval > *bval) return 1;
    if (*bval > *aval) return -1;
    return 0;
}

static int
arg_as_double(const char *binfunc, PyObject *const *args, Py_ssize_t n,
              double *dp)
{
    double d = PyFloat_AsDouble(args[n]);
    if (d == -1.0 && PyErr_Occurred()) {
        // Doing "raise new_exception(...) from old_exception" in the
        // C API is way more trouble than it's worth.  See discussion
        // here: https://stackoverflow.com/questions/51030659
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s: could not convert arg %zd (%S) to C double",
                     binfunc, n, (PyObject *)Py_TYPE(args[n]));
        return -1;
    }
    *dp = d;
    return 0;
}

static int
arg_as_long(const char *binfunc, PyObject *const *args, Py_ssize_t n,
            long *lp)
{
    long l = PyLong_AsLong(args[n]);
    if (l == -1 && PyErr_Occurred()) {
        // see arg_as_double for why we're discarding the original error
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s: could not convert arg %zd (%S) to C long",
                     binfunc, n, (PyObject *)Py_TYPE(args[n]));
        return -1;
    }
    *lp = l;
    return 0;
}

static int
arg_as_array(const char *binfunc, PyObject *const *args, Py_ssize_t n,
             npy_intp insize, bool none_ok, PyArrayObject **p_array,
             const npy_intp ref_itemsize, const char *ref_dtype_name)
{
    *p_array = NULL;
    if (Py_IsNone(args[n])) {
        if (none_ok) {
            return 0;
        }
        PyErr_Format(PyExc_TypeError, "%s: arg %zd may not be None",
                     binfunc, n);
        return -1;
    }
    PyArrayObject *array = (PyArrayObject *)PyArray_FROM_O(args[n]);
    // PyArray_FROM_O creates a strong reference to the object. We do not
    // actually want to create a strong reference to the object here.
    if (!array) {
        // see arg_as_double for why we're discarding the original error
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s: could not convert arg %zd (%S) to ndarray",
                     binfunc, n, (PyObject *)Py_TYPE(args[n]));
        return -1;
    }
    Py_DECREF(args[n]);
    if (PyArray_NDIM(array) != 1) {
        PyErr_Format(PyExc_TypeError,
                     "%s: arg %zd must be a 1-dimensional array",
                     binfunc, n);
        return -1;
    }
    if (insize >= 0 && PyArray_SIZE(array) != insize) {
        PyErr_Format(PyExc_TypeError,
                     "%s: arg %zd must have %zd elements (it has %zd)",
                     binfunc, n, insize, PyArray_SIZE(array));
        return -1;
    }

    if (ref_dtype_name != NULL) {
        const char *dtype_name = PyArray_DESCR(array)->typeobj->tp_name;
        if (strcmp(dtype_name, ref_dtype_name) != 0) {
            PyErr_Format(
                    PyExc_TypeError,
                    "%s: array %zd must be of type %s; got %s",
                    binfunc, n, ref_dtype_name, dtype_name);
            return -1;
        }
    }
    npy_intp itemsize = PyArray_ITEMSIZE(array);
    if (ref_itemsize != itemsize) {
        PyErr_Format(PyExc_TypeError,
                     "%s: array %zd must have %zd-byte elements; got %zd",
                     binfunc, n, ref_itemsize, itemsize);
        return -1;
    }
    *p_array = array;
    return 0;
}

static int
double_array_bounds(PyArrayObject *arr, double bounds[static 2]) {
    double maxval, minval;
    PyObject *maxscalar = PyArray_Max(arr, 0, NULL);
    PyObject *minscalar = PyArray_Min(arr, 0, NULL);
    if (maxscalar == NULL || minscalar == NULL) {
        return -1;
    }
    PyArray_ScalarAsCtype(maxscalar, &maxval);
    PyArray_ScalarAsCtype(minscalar, &minval);
    Py_DECREF(maxscalar);
    Py_DECREF(minscalar);
    bounds[0] = minval;
    bounds[1] = maxval;
    return 0;
}

static int
check_bounds (
    const char *binfunc,
    PyArrayObject *iarg,
    PyArrayObject *jarg,
    double ibounds[static 2],
    double jbounds[static 2]
) {
    double iminmax[2], jminmax[2];
    if (
        double_array_bounds(iarg, iminmax) == -1
        || double_array_bounds(jarg, jminmax) == -1
   ) {
        PyErr_Format(
            PyExc_RuntimeError, "%s: could not find input array min/max.",
            binfunc
        );
        return -1;
    }
    // the Python handlers set these values to NaN when no bounds were
    // specified by the user. In this case we simply set the bounds to the
    // min/max of the coordinate arrays plus a little slop to keep the largest
    // values in the rightmost bin.
    if (
        isnan(ibounds[0])
        || isnan(ibounds[1])
        || isnan(jbounds[0])
        || isnan(jbounds[1])
    ) {
        // TODO: It would be better to not just use the magic number 5e-15 here,
        //  but rather base it on the resolution of the data type.
        ibounds[0] = iminmax[0];
        ibounds[1] = iminmax[1] + 5e-15;
        jbounds[0] = jminmax[0];
        jbounds[1] = jminmax[1] + 5e-15;
        return 0;
    }
    // otherwise, check to make sure people didn't specify bounds inside the
    // min/max of the input coordinates. We use the values of the x and y
    // coordinate arrays to select indices in the output arrays, and we aren't
    // willing to do bounds checking in the inner loop, so bounds within the
    // the x/y coordinate ranges are memory-unsafe.
    if (
        ibounds[0] > iminmax[0]
        || ibounds[1] < iminmax[1]
        || jbounds[0] > jminmax[0]
        || jbounds[1] < jminmax[1]
    ) {
        // TODO: this error message could be better
        PyErr_Format(PyExc_ValueError, "%s: specified bounds are too small.",
                     binfunc);
        return -1;
    }
    return 0;
}

// yes, this function has 11 arguments. i'm very sorry.
static int
unpack_binfunc_args(
    const char *binfunc,
    PyObject *const *args,
    Py_ssize_t n_args,
    Py_ssize_t n_inputs,
    Py_ssize_t n_outputs,
    Py_ssize_t n_required_outputs,
    Iterface *iter,
    Histspace *space,
    long *ni,
    long *nj,
    PyArrayObject **outputs
) {
    // All binfuncs take arguments in this order:
    // i, j,[, v], output 1[, output 2, ...], xmin, xmax, ymin, ymax, ni, nj
    // Outputs are identified by position
    // Unwanted outputs will be passed as None
    // The first 'n_required_outputs' outputs may not be None
    // (even if the Python-level caller doesn't want 'em, we need 'em
    // for scratch space)
    assert(n_inputs == 2 || n_inputs == 3);
    assert(n_required_outputs >= 1);
    assert(n_outputs >= n_required_outputs);

    if (n_args != 6 + n_inputs + n_outputs) {
        PyErr_Format(PyExc_TypeError, "%s: expected %zd args, got %zd",
                     binfunc, 6 + n_inputs + n_outputs, n_args);
        return -1;
    }

    PyArrayObject *iarg, *jarg, *varg;
    if (arg_as_array(binfunc, args, 0, -1, false, &iarg, 8, "numpy.float64"))
        return -1;
    if (arg_as_array(binfunc, args, 1, PyArray_SIZE(iarg), false, &jarg, 8,
                     "numpy.float64"))
        return -1;
    if (n_inputs == 3) {
        if (arg_as_array(binfunc, args, 2, PyArray_SIZE(iarg), false, &varg,
                         8, NULL))
            return -1;
    } else {
        varg = NULL;
    }

    double imin, imax, jmin, jmax;
    if (   arg_as_double(binfunc, args, n_inputs + n_outputs + 0, &imin)
        || arg_as_double(binfunc, args, n_inputs + n_outputs + 1, &imax)
        || arg_as_double(binfunc, args, n_inputs + n_outputs + 2, &jmin)
        || arg_as_double(binfunc, args, n_inputs + n_outputs + 3, &jmax)
        || arg_as_long  (binfunc, args, n_inputs + n_outputs + 4, ni)
        || arg_as_long  (binfunc, args, n_inputs + n_outputs + 5, nj)) {
        return -1;
    }
    // output arrays are processed last because we need to know ni and
    // nj to know how big they should be
    // even if none of the outputs are _required_, at least one of them
    // should be present, otherwise why bother calling at all?
    npy_intp output_size = *ni * *nj;
    bool have_an_output = false;
    for (Py_ssize_t i = 0; i < n_outputs; i++) {
        if (arg_as_array(binfunc, args, n_inputs + i,
                         output_size, i >= n_required_outputs,
                         &outputs[i], 8, NULL)) {
            return -1;
        }
        if (outputs[i]) {
            have_an_output = true;
        }
    }
    if (!have_an_output) {
        PyErr_SetString(
            PyExc_TypeError, "at least one output array should be present"
        );
        return -1;
    }
    double ibounds[2] = {imin, imax};
    double jbounds[2] = {jmin, jmax};
    if (check_bounds(binfunc, iarg, jarg, ibounds, jbounds))
        return -1;
    PyArrayObject *arrs[3] = {iarg, jarg, varg };
    if (!init_iterface(iter, arrs, n_inputs)) {
        return -1;
    }
    init_histspace(space, ibounds, jbounds, *ni, *nj);
    return 0;
}

PyObject*
binned_count(PyObject *self, PyObject *const *args, Py_ssize_t n_args)
{
    long ni, nj;
    Iterface iter;
    Histspace space;
    PyArrayObject *countarg;
    if (unpack_binfunc_args(__func__, args, n_args, 2, 1, 1,
                            &iter, &space, &ni, &nj, &countarg)) {
        return NULL;
    }

    long *count = PYARRAY_AS_LONGS(countarg);
    FOR_NDITER_COUNT (&iter, &space, indices) {
        if (indices[0] >= 0) count[indices[1] + nj * indices[0]] += 1;
    }

    Py_RETURN_NONE;
}

PyObject*
binned_sum(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    long ni, nj;
    Iterface iter;
    Histspace space;
    PyArrayObject *sumarg;
    if (unpack_binfunc_args(__func__, args, n_args, 3, 1, 1,
                            &iter, &space, &ni, &nj, &sumarg)) {
        return NULL;
    }
    double *sum = PYARRAY_AS_DOUBLES(sumarg);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        if (indices[0] >= 0) sum[indices[1] + indices[0] * nj] += val;
    }
    Py_RETURN_NONE;
}

PyObject*
binned_countvals(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    long ni, nj;
    Iterface iter;
    Histspace space;
    PyArrayObject *outputs[3];
    if (unpack_binfunc_args(__func__, args, n_args, 3, 3, 2,
                            &iter, &space, &ni, &nj, outputs)) {
        return NULL;
    }

    double *count = PYARRAY_AS_DOUBLES(outputs[0]);
    double *sum = PYARRAY_AS_DOUBLES(outputs[1]);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * nj, val);
    }
    if (outputs[2]) {
        populate_meanarr(ni * nj, count, sum, PYARRAY_AS_DOUBLES(outputs[2]));
    }
    Py_RETURN_NONE;
}

PyObject*
binned_std(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    long ni, nj;
    Iterface iter;
    Histspace space;
    PyArrayObject *outputs[4];
    if (unpack_binfunc_args(__func__, args, n_args, 3, 4, 3,
                            &iter, &space, &ni, &nj, outputs)) {
        return NULL;
    }

    // NOTE: no point making the caller construct an ndarray for the sum of
    // squares (who would want it?)
    double *sqr = calloc(sizeof *sqr, ni * nj);
    if (!sqr) {
        PyErr_NoMemory();
        return NULL;
    }
    double *count = PYARRAY_AS_DOUBLES(outputs[0]);
    double *sum = PYARRAY_AS_DOUBLES(outputs[1]);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * nj, val);
        sqr[indices[1] + nj * indices[0]] += (val * val);
    }

    populate_stdarr(ni * nj, count, sum, sqr, PYARRAY_AS_DOUBLES(outputs[2]));
    if (outputs[3]) {
        populate_meanarr(ni * nj, count, sum, PYARRAY_AS_DOUBLES(outputs[3]));
    }

    free(sqr);
    Py_RETURN_NONE;
}

PyObject*
binned_minmax(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    long ni, nj;
    Iterface iter;
    Histspace space;
    PyArrayObject *outputs[2];
    if (unpack_binfunc_args(__func__, args, n_args, 3, 2, 0,
                            &iter, &space, &ni, &nj, outputs)) {
        return NULL;
    }
    double *min = outputs[0] ? PYARRAY_AS_DOUBLES(outputs[0]) : NULL;
    double *max = outputs[1] ? PYARRAY_AS_DOUBLES(outputs[1]) : NULL;
    double val;

    for (long ix = 0; ix < ni * nj; ix++) {
        if (max) max[ix] = -INFINITY;
        if (min) min[ix] = INFINITY;
    }

    FOR_NDITER (&iter, &space, indices, &val) {
        if (max &&
            max[indices[1] + nj * indices[0]] < val) {
            max[indices[1] + nj * indices[0]] = val;
        }
        if (min &&
            min[indices[1] + nj * indices[0]] > val) {
            min[indices[1] + nj * indices[0]] = val;
        }
    }

    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with INFINITY / -INFINITY;
    //  just have a special case up top
    for (long ix = 0; ix < ni * nj; ix++) {
        if (max && max[ix] == -INFINITY) max[ix] = NAN;
        if (min && min[ix] == INFINITY) min[ix] = NAN;
    }
    Py_RETURN_NONE;
}

PyObject*
binned_median(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    // TODO: there may be unnecessary copies happening here
    long ni, nj;
    Iterface iter;
    Histspace space;
    PyArrayObject *medarg;
    if (unpack_binfunc_args(__func__, args, n_args, 3, 1, 1,
                            &iter, &space, &ni, &nj, &medarg)) {
        return NULL;
    }
    ObjectCleanupInfo **toclean = malloc(sizeof(ObjectCleanupInfo*) * 20);
    if (toclean == NULL) {
        PyErr_SetString(PyErr_NoMemory(), "could not init cleanup routine");
        return NULL;
    }
    for (int i = 0; i < 20; i++) {
        toclean[i] = malloc(sizeof(ObjectCleanupInfo));
        if (toclean[i] == NULL) {
            for (int j = 0; j < i; j++) free(toclean[i]);
            PyErr_SetString(PyErr_NoMemory(), "could not init cleanup routine");
            return NULL;
        }
    }
    void *tofree[3] = {NULL, NULL, NULL};
    int nclean = 0, nfree = 3, cs = 20;
    // if we get here these assignments have been validated
    PyArrayObject *iarg = (PyArrayObject *) args[0];
    PyArrayObject *varg = (PyArrayObject *) args[2];
    PyObject *numpy = PyImport_ImportModule("numpy");
    ABORT_IF_NULL(numpy, "numpy", tofree, nfree, toclean, nclean, cs);
    nclean = prep_cleanup(toclean, numpy, false, nclean);
    PyObject *unique = GETATTR(numpy, "unique");
    ABORT_IF_NULL(unique, "np.unique", tofree, nfree, toclean, nclean, cs);
    nclean = prep_cleanup(toclean, unique, false, nclean);
    long arrsize = PyArray_SIZE(iarg);
    // idig and jdig are the bin indices of each value in our input i and j
    // arrays respectively. this is a cheaty version of a digitize-type
    // operation that works only because we always have regular bins.
    PyArrayObject *idig_arr = init_ndarray1d(arrsize, NPY_LONG, 0);
    ABORT_IF_NULL(idig_arr, "idig array", tofree, nfree, toclean, nclean, cs);
    nclean = prep_cleanup(toclean, (PyObject *) idig_arr, true, nclean);
    PyArrayObject *jdig_arr = init_ndarray1d(arrsize, NPY_LONG, 0);
    ABORT_IF_NULL(jdig_arr, "jdig array", tofree, nfree, toclean, nclean, cs);
    nclean = prep_cleanup(toclean, (PyObject *) jdig_arr, true, nclean);
    long *idig = (long *) PyArray_DATA(idig_arr);
    long *jdig = (long *) PyArray_DATA(jdig_arr);
    for (long ix = 0; ix < arrsize; ix++) {
        npy_intp itersize = *iter.sizep;
        long indices[2];
        hist_index(&iter, &space, indices);
        idig[ix] = indices[0];
        jdig[ix] = indices[1];
        itersize--;
        stride(&iter);
    }
    NpyIter_Deallocate(iter.iter);
    PyArrayObject *idig_sortarr = (PyArrayObject *) NP_ARGSORT(idig_arr);
    ABORT_IF_NULL(idig_sortarr, "idig sort", tofree, nfree, toclean, nclean,
                  cs);
    nclean = prep_cleanup(toclean, (PyObject *) idig_sortarr, true, nclean);
    long *idig_sort = (long *) PyArray_DATA(idig_sortarr);
    PyArrayObject *idig_uniqarr = (PyArrayObject *) PYCALL_1(unique, idig_arr);
    ABORT_IF_NULL(idig_uniqarr, "idig uniq", tofree, nfree, toclean,
                  nclean, cs);
    nclean = prep_cleanup(toclean, (PyObject *) idig_uniqarr, true, nclean);
    long ni_unique = PyArray_SIZE(idig_uniqarr);
    long *idig_uniq = (long *) PyArray_DATA(idig_uniqarr);
    double *vals = (double *) PyArray_DATA(varg);
    long i_sort_ix = 0;
    double* median = PYARRAY_AS_DOUBLES(medarg);
    for (long xix = 0; xix < ni_unique; xix++) {
        long ibin = idig_uniq[xix];
        // TODO: is it actually more efficient to loop over the array once
        //  to count the bins, allocate ibin_indices of the actually-required
        //  size, and then loop over it again?
        long *ibin_indices = calloc(sizeof *ibin_indices, arrsize);
        ABORT_IF_NULL(ibin_indices, ibin_indices (xix), tofree, nfree,
                      toclean, nclean, cs);
        tofree[0] = ibin_indices;
        long ibin_elcount = 0;
        for(;;) {
            ibin_indices[ibin_elcount] = idig_sort[i_sort_ix];
            ibin_elcount += 1;
            if (i_sort_ix + 1 >= arrsize) break;
            i_sort_ix += 1;
            if (idig[idig_sort[i_sort_ix]] != ibin) break;
        }
        if (ibin_elcount == 0) {
            free_pointer_array(tofree, nfree);
            continue;
        }
        long *match_buckets = malloc(sizeof *match_buckets * nj * ibin_elcount);
        ABORT_IF_NULL(match_buckets, match buckets (xix), tofree, nfree,
                      toclean, nclean, cs);
        tofree[1] = match_buckets;
        long *match_count = calloc(sizeof *match_count, nj);
        ABORT_IF_NULL(match_count, match array(xix), tofree, nfree, toclean,
                      nclean, cs);
        tofree[2] = match_count;
        for (long j = 0; j < ibin_elcount; j++) {
            long jbin = jdig[ibin_indices[j]];
            match_buckets[
                jbin * ibin_elcount + match_count[jbin]
            ] = ibin_indices[j];
            match_count[jbin] += 1;
        }
        for (long jbin = 0; jbin < nj; jbin++) {
            long binsize = match_count[jbin];
            if (binsize == 0) continue;
            double *binvals = malloc(sizeof *binvals * binsize);
            ABORT_IF_NULL(binvals, bins (jbin, xix), tofree, nfree, toclean,
                          nclean, cs);
            for (long ix_ix = 0; ix_ix < binsize; ix_ix++) {
                binvals[ix_ix] = vals[match_buckets[jbin * ibin_elcount + ix_ix]];
            }
            qsort(binvals, binsize, sizeof(double), doublecomp);
            double bin_median;
            if (binsize % 2 == 1) bin_median = binvals[binsize / 2];
            else bin_median = (
                  binvals[binsize / 2] + binvals[binsize / 2 - 1]
              ) / 2;
            median[jbin + space.nj * ibin] = bin_median;
            free(binvals);
        }
        free_pointer_array(tofree, 3);
    }
    do_object_cleanup(tofree, nfree, toclean, nclean, cs);
    Py_RETURN_NONE;
}
