#include "binning.h"
#include "iterators.h"

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
    for (long i = 0; i < size; i++) {
        if (count[i] == 0) mean[i] = NAN;
        else mean[i] = sum[i] / count[i];
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
    for (long i = 0; i < size; i++) {
        if (count[i] == 0) std[i] = NAN;
        else std[i] = stdev(count[i], sum[i], sqr[i]);
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
             npy_intp insize, bool none_ok, PyArrayObject **p_array)
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
    if (!array) {
        // see arg_as_double for why we're discarding the original error
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s: could not convert arg %zd (%S) to ndarray",
                     binfunc, n, (PyObject *)Py_TYPE(args[n]));
        return -1;
    }


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

    // TODO also check the element type

    *p_array = array;
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
    long *nx,
    long *ny,
    PyArrayObject **outputs
) {
    // All binfuncs take arguments in this order:
    // x, y[, v], output 1[, output 2, ...], xmin, xmax, ymin, ymax, nx, ny
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

    PyArrayObject *xarg, *yarg, *varg;
    if (arg_as_array(binfunc, args, 0, -1, false, &xarg))
        return -1;
    if (arg_as_array(binfunc, args, 1, PyArray_SIZE(xarg), false, &yarg))
        return -1;
    if (n_inputs == 3) {
        if (arg_as_array(binfunc, args, 2, PyArray_SIZE(xarg), false, &varg))
            return -1;
    } else {
        varg = NULL;
    }

    double xmin, xmax, ymin, ymax;
    if (   arg_as_double(binfunc, args, n_inputs + n_outputs + 0, &xmin)
        || arg_as_double(binfunc, args, n_inputs + n_outputs + 1, &xmax)
        || arg_as_double(binfunc, args, n_inputs + n_outputs + 2, &ymin)
        || arg_as_double(binfunc, args, n_inputs + n_outputs + 3, &ymax)
        || arg_as_long  (binfunc, args, n_inputs + n_outputs + 4, nx)
        || arg_as_long  (binfunc, args, n_inputs + n_outputs + 4, ny)) {
        return -1;
    }

    // output arrays are processed last because we need to know nx and
    // ny to know how big they should be
    // even if none of the outputs are _required_, at least one of them
    // should be present, otherwise why bother calling at all?
    npy_intp output_size = *nx * *ny;
    bool have_an_output = false;
    for (Py_ssize_t i = 0; i < n_outputs; i++) {
        if (arg_as_array(binfunc, args, n_inputs + i,
                         output_size, i >= n_required_outputs,
                         &outputs[i])) {
            return -1;
        }
        if (outputs[i]) {
            have_an_output = true;
        }
    }
    if (!have_an_output) {
        PYRAISE(ValueError, "at least one output array should be present");
    }

    double xbounds[2] = {xmin, xmax};
    double ybounds[2] = {ymin, ymax};
    PyArrayObject *arrs[3] = { xarg, yarg, varg };
    if (!init_iterface(iter, arrs, n_inputs)) {
        PYRAISE(PyExc_RuntimeError, "Binning setup failed.");
    }
    init_histspace(space, xbounds, ybounds, *nx, *ny);
    return 0;
}

PyObject*
binned_count(PyObject *self, PyObject *const *args, Py_ssize_t n_args)
{
    long nx, ny;
    Iterface iter;
    Histspace space;
    PyArrayObject *countarg;
    if (unpack_binfunc_args(__func__, args, n_args, 2, 1, 1,
                            &iter, &space, &nx, &ny, &countarg)) {
        return NULL;
    }

    long *count = PYARRAY_AS_LONGS(countarg);
    FOR_NDITER_COUNT (&iter, &space, indices) {
        if (indices[0] >= 0) count[indices[1] + ny * indices[0]] += 1;
    }

    Py_RETURN_NONE;
}

PyObject*
binned_sum(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    long nx, ny;
    Iterface iter;
    Histspace space;
    PyArrayObject *sumarg;
    if (unpack_binfunc_args(__func__, args, n_args, 3, 1, 1,
                            &iter, &space, &nx, &ny, &sumarg)) {
        return NULL;
    }

    double *sum = PYARRAY_AS_DOUBLES(sumarg);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        if (indices[0] >= 0) sum[indices[1] + ny * indices[0]] += val;
    }
    Py_RETURN_NONE;
}

PyObject*
binned_countvals(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    long nx, ny;
    Iterface iter;
    Histspace space;
    PyArrayObject *outputs[3];
    if (unpack_binfunc_args(__func__, args, n_args, 3, 3, 2,
                            &iter, &space, &nx, &ny, outputs)) {
        return NULL;
    }

    double *count = PYARRAY_AS_DOUBLES(outputs[0]);
    double *sum = PYARRAY_AS_DOUBLES(outputs[1]);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * ny, val);
    }
    if (outputs[2]) {
        populate_meanarr(nx * ny, count, sum, PYARRAY_AS_DOUBLES(outputs[2]));
    }
    Py_RETURN_NONE;
}

PyObject*
binned_std(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    long nx, ny;
    Iterface iter;
    Histspace space;
    PyArrayObject *outputs[4];
    if (unpack_binfunc_args(__func__, args, n_args, 3, 4, 3,
                            &iter, &space, &nx, &ny, outputs)) {
        return NULL;
    }

    // NOTE: no point making the caller construct an ndarray for the sum of
    // squares (who would want it?)
    double *sqr = calloc(sizeof *sqr, nx * ny);
    if (!sqr) {
        PyErr_NoMemory();
        return NULL;
    }
    double *count = PYARRAY_AS_DOUBLES(outputs[0]);
    double *sum = PYARRAY_AS_DOUBLES(outputs[1]);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * ny, val);
        sqr[indices[1] + ny * indices[0]] += (val * val);
    }

    populate_stdarr(nx * ny, count, sum, sqr, PYARRAY_AS_DOUBLES(outputs[2]));
    if (outputs[3]) {
        populate_meanarr(nx * ny, count, sum, PYARRAY_AS_DOUBLES(outputs[3]));
    }

    free(sqr);
    Py_RETURN_NONE;
}

PyObject*
binned_minmax(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    long nx, ny;
    Iterface iter;
    Histspace space;
    PyArrayObject *outputs[2];
    if (unpack_binfunc_args(__func__, args, n_args, 3, 2, 0,
                            &iter, &space, &nx, &ny, outputs)) {
        return NULL;
    }
    double *min = outputs[0] ? PYARRAY_AS_DOUBLES(outputs[0]) : NULL;
    double *max = outputs[1] ? PYARRAY_AS_DOUBLES(outputs[1]) : NULL;
    double val;

    for (long i = 0; i < nx * ny; i++) {
        if (max) max[i] = -INFINITY;
        if (min) min[i] = INFINITY;
    }

    FOR_NDITER (&iter, &space, indices, &val) {
        if (max &&
            max[indices[1] + ny * indices[0]] < val) {
            max[indices[1] + ny * indices[0]] = val;
        }
        if (min &&
            min[indices[1] + ny * indices[0]] > val) {
            min[indices[1] + ny * indices[0]] = val;
        }
    }

    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with INFINITY / -INFINITY;
    //  just have a special case up top
    for (long i = 0; i < nx * ny; i++) {
        if (max && max[i] == -INFINITY) max[i] = NAN;
        if (min && min[i] == INFINITY) min[i] = NAN;
    }
    Py_RETURN_NONE;
}

PyObject*
binned_median(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    // TODO: there may be unnecessary copies happening here
    long nx, ny;
    Iterface iter;
    Histspace space;
    PyArrayObject *medarg;
    if (unpack_binfunc_args(__func__, args, n_args, 3, 1, 1,
                            &iter, &space, &nx, &ny, &medarg)) {
        return NULL;
    }
    // if we get here these assignments have been validated
    PyArrayObject *xarg = (PyArrayObject *)args[0];
    PyArrayObject *varg = (PyArrayObject *)args[2];

    PyObject *numpy = PyImport_ImportModule("numpy");
    PyObject *unique = GETATTR(numpy, "unique");
    long arrsize = PyArray_SIZE(xarg);
    // xdig and ydig are the bin indices of each value in our input x and y
    // arrays respectively. this is a cheaty version of a digitize-type
    // operation that works only because we always have regular bins.
    PyArrayObject *xdig_arr = init_ndarray1d(arrsize, NPY_LONG, 0);
    PyArrayObject *ydig_arr = init_ndarray1d(arrsize, NPY_LONG, 0);
    long *xdig = (long *) PyArray_DATA(xdig_arr);
    long *ydig = (long *) PyArray_DATA(ydig_arr);
    for (long i = 0; i < arrsize; i++) {
        npy_intp itersize = *iter.sizep;
        long indices[2];
        hist_index(&iter, &space, indices);
        xdig[i] = indices[0];
        ydig[i] = indices[1];
        itersize--;
        stride(&iter);
    }
    NpyIter_Deallocate(iter.iter);
    PyArrayObject *xdig_sortarr = (PyArrayObject *) NP_ARGSORT(xdig_arr);
    // TODO: ensure that these remain NULL when cast to PyArrayObject in
    //  Pythonland failure cases
    if (xdig_sortarr == NULL) return NULL;
    long *xdig_sort = (long *) PyArray_DATA(xdig_sortarr);
    PyArrayObject *xdig_uniqarr = (PyArrayObject *) PYCALL_1(unique, xdig_arr);
    // TODO: laboriously decrement various references in these failure cases
    if (xdig_uniqarr == NULL) return NULL;
    long nx_uniq = PyArray_SIZE(xdig_uniqarr);
    long *xdig_uniq = (long *) PyArray_DATA(xdig_uniqarr);
    DECREF_ALL(unique, numpy);
    double *vals = (double *) PyArray_DATA(varg);
    long x_sort_ix = 0;
    double* median = PYARRAY_AS_DOUBLES(medarg);
    for (long xix = 0; xix < nx_uniq; xix++) {
        long xbin = xdig_uniq[xix];
        // TODO: is it actually more efficient to loop over the array once
        //  to count the bins, allocate xbin_indices of the actually-required
        //  size, and then loop over it again?
        long *xbin_indices = calloc(sizeof *xbin_indices, arrsize);
        long xbin_elcount = 0;
        for(;;) {
            xbin_indices[xbin_elcount] = xdig_sort[x_sort_ix];
            xbin_elcount += 1;
            if (x_sort_ix + 1 >= arrsize) break;
            x_sort_ix += 1;
            if (xdig[xdig_sort[x_sort_ix]] != xbin) break;
        }
        if (xbin_elcount == 0) {
            free(xbin_indices);
            continue;
        }
        long *match_buckets = malloc(sizeof *match_buckets * ny * xbin_elcount);
        long *match_count = calloc(sizeof *match_count, ny);
        for (long j = 0; j < xbin_elcount; j++) {
            long ybin = ydig[xbin_indices[j]];
            match_buckets[ybin * xbin_elcount + match_count[ybin]] = xbin_indices[j];
            match_count[ybin] += 1;
        }
        for (long ybin = 0; ybin < ny; ybin++) {
            long binsize = match_count[ybin];
            if (binsize == 0) continue;
            double *binvals = malloc(sizeof *binvals * binsize);
            for (long ix_ix = 0; ix_ix < binsize; ix_ix++) {
                binvals[ix_ix] = vals[match_buckets[ybin * xbin_elcount + ix_ix]];
            }
            qsort(binvals, binsize, sizeof(double), doublecomp);
            double bin_median;
            if (binsize % 2 == 1) bin_median = binvals[binsize / 2];
            else bin_median = (
                  binvals[binsize / 2] + binvals[binsize / 2 - 1]
              ) / 2;
            median[ybin + space.ny * xbin] = bin_median;
            free(binvals);
        }
        FREE_ALL(match_buckets, match_count, xbin_indices);
    }
    DESTROY_ALL_NDARRAYS(xdig_uniqarr, xdig_sortarr, ydig_arr, xdig_arr);
    Py_RETURN_NONE;
}
