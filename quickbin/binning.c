#include "binning.h"

static bool
check_arrs(PyArrayObject *arrays[static 2], long n_arrays) {
    npy_intp insize = PyArray_SIZE(arrays[0]);
    for (long i = 0; i < n_arrays; i++) {
        if (arrays[i] == NULL) {
            PyErr_SetString(PyExc_TypeError, "Couldn't parse an array");
            return false;
        }
        if (PyArray_NDIM(arrays[i]) != 1) {
            PyErr_SetString(PyExc_TypeError, "Arrays must be of dimension 1");
            return false;
        }
        if (PyArray_SIZE(arrays[i]) != insize) {
            PyErr_SetString(PyExc_TypeError, "Arrays must be of the same size");
            return false;
        }
    }
    return true;
}


#define PYARRAY_AS_DOUBLES(PYARG)                                   \
(double *) PyArray_DATA((PyArrayObject *) PyArray_FROM_O(PYARG))

#define PYARRAY_AS_LONGS(PYARG)                                     \
(long *) PyArray_DATA((PyArrayObject *) PyArray_FROM_O(PYARG))


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

int
prep_binning(
    const long nx, const long ny,
    const double xmin, const double xmax,
    const double ymin, const double ymax,
    PyObject *xarg, PyObject *yarg, PyObject *varg,
    PyArrayObject *arrs[static 3], const int narrs,
    Iterface *iter, Histspace *space
) {
    if (check_arrs(arrs, narrs) == false) { return 0; };
    arrs[0] = (PyArrayObject *) PyArray_FROM_O(xarg);
    arrs[1] = (PyArrayObject *) PyArray_FROM_O(yarg);
    if (narrs == 3) {
        arrs[2] = (PyArrayObject *) PyArray_FROM_O(varg);
    }
    double xbounds[2] = {xmin, xmax};
    double ybounds[2] = {ymin, ymax};
    if (!init_iterface(iter, arrs, narrs)) {
        PYRAISE(ValueError, "Binning setup failed.");
    }
    init_histspace(space, xbounds, ybounds, nx, ny);
    return 1;
}

PyObject*
binned_count(PyObject *self, PyObject *args)
{
    long nx, ny;
    double xmin, xmax, ymin, ymax;
    PyObject *xarg, *yarg, *countarg;
    if (!PyArg_ParseTuple(args, "OOOddddlll",
        &xarg, &yarg, &countarg, &xmin, &xmax,
        &ymin, &ymax, &nx, &ny)
    ) { return NULL; } // PyArg_ParseTuple has set an exception
    Iterface iter;
    Histspace space;
    PyArrayObject *arrs[3];
    int status = prep_binning(
        nx, ny, xmin, xmax, ymin, ymax, xarg, yarg, NULL, arrs,
        2, &iter, &space
    );
    if (status == 0) { return NULL; }
    long *count = PYARRAY_AS_LONGS(countarg);
    FOR_NDITER_COUNT (&iter, &space, indices) {
        if (indices[0] >= 0) count[indices[1] + ny * indices[0]] += 1;
    }
    return Py_None;
}

PyObject*
binned_sum(PyObject *self, PyObject *args) {
    long nx, ny;
    double xmin, xmax, ymin, ymax, val;
    PyObject *xarg, *yarg, *varg, *sumarg;
    // PyArg_ParseTuple sets an exception on failure
    if (!PyArg_ParseTuple(args, "OOOOddddlll",
      &xarg, &yarg, &varg, &sumarg,
      &xmin, &xmax, &ymin, &ymax, &nx, &ny)) { return NULL; }
    Iterface iter;
    Histspace space;
    PyArrayObject *arrs[3];
    int status = prep_binning(
        nx, ny, xmin, xmax, ymin, ymax, xarg, yarg, varg, arrs, 3, &iter, &space
    );
    if (status == 0) { return NULL; }
    double *sum = PYARRAY_AS_DOUBLES(sumarg);
    FOR_NDITER (&iter, &space, indices, &val) {
        if (indices[0] >= 0) sum[indices[1] + ny * indices[0]] += val;
    }
    return Py_None;
}

PyObject*
binned_countvals(PyObject *self, PyObject *args) {
    long nx, ny;
    double xmin, xmax, ymin, ymax, val;
    PyObject *xarg, *yarg, *varg, *countarg, *sumarg, *meanarg;
    unsigned int opmask;
    // PyArg_ParseTuple sets an exception on failure
    if (!PyArg_ParseTuple(args, "OOOOOddddllli",
      &xarg, &yarg, &varg, &countarg, &sumarg, &meanarg,
      &xmin, &xmax, &ymin, &ymax,
      &nx, &ny, &opmask)) { return NULL; }
    Iterface iter;
    Histspace space;
    PyArrayObject *arrs[3];
    int status = prep_binning(
        nx, ny, xmin, xmax, ymin, ymax, xarg, yarg, varg, arrs, 3, &iter, &space
    );
    if (status == 0) { return NULL; }
    double *count = PYARRAY_AS_DOUBLES(countarg);
    double *sum = PYARRAY_AS_DOUBLES(sumarg);
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * ny, val);
    }
    if (opmask & GH_MEAN) {
        populate_meanarr(nx * ny, count, sum, PYARRAY_AS_DOUBLES(meanarg));
    }
    return Py_None;
}

PyObject*
binned_std(PyObject *self, PyObject *args) {
    long nx, ny;
    double xmin, xmax, ymin, ymax, val;
    PyObject *xarg, *yarg, *varg, *countarg, *sumarg, *meanarg, *stdarg;
    unsigned int opmask;
    // PyArg_ParseTuple sets an exception on failure
    if (!PyArg_ParseTuple(args, "OOOOOOOddddllli",
      &xarg, &yarg, &varg, &countarg, &sumarg, &meanarg, &stdarg,
      &xmin, &xmax, &ymin, &ymax,
      &nx, &ny, &opmask)) { return NULL; }
    Iterface iter;
    Histspace space;
    PyArrayObject *arrs[3];
    int status = prep_binning(
        nx, ny, xmin, xmax, ymin, ymax, xarg, yarg, varg, arrs, 3, &iter, &space
    );
    if (status == 0) return NULL;
    // NOTE: no point making the caller construct an ndarray for the sum of
    // squares (who would want it?)
    double *sqr = calloc(sizeof *sqr, nx * ny);
    double *sum = PYARRAY_AS_DOUBLES(sumarg);
    double *count = PYARRAY_AS_DOUBLES(countarg);
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * ny, val);
        sqr[indices[1] + ny * indices[0]] += (val * val);
    }
    populate_stdarr(nx * ny, count, sum, sqr, PYARRAY_AS_DOUBLES(stdarg));
    free(sqr);
    return Py_None;
}

PyObject*
binned_minmax(PyObject *self, PyObject *args) {
    long nx, ny;
    double xmin, xmax, ymin, ymax, val;
    PyObject *xarg, *yarg, *varg, *minarg, *maxarg;
    // PyArg_ParseTuple sets an exception on failure
    if (!PyArg_ParseTuple(args, "OOOOOddddlll",
      &xarg, &yarg, &varg, &minarg, &maxarg,
      &xmin, &xmax, &ymin, &ymax, &nx, &ny)) { return NULL; }
    Iterface iter;
    Histspace space;
    PyArrayObject *arrs[3];
    int status = prep_binning(
        nx, ny, xmin, xmax, ymin, ymax, xarg, yarg, varg, arrs, 3, &iter, &space
    );
    if (status == 0) { return NULL; }
    double* min = PYARRAY_AS_DOUBLES(minarg);
    double* max = PYARRAY_AS_DOUBLES(maxarg);
    for (long i = 0; i < nx * ny; i++) {
        max[i] = -INFINITY;
        min[i] = INFINITY;
    }
    FOR_NDITER (&iter, &space, indices, &val) {
        if (max[indices[1] + ny * indices[0]] < val) {
            max[indices[1] + ny * indices[0]] = val;
        }
        if (min[indices[1] + ny * indices[0]] > val) {
            min[indices[1] + ny * indices[0]] = val;
        }
    }
    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with INFINITY / -INFINITY;
    //  just have a special case up top
    for (long i = 0; i < nx * ny; i++) {
        if (min[i] == INFINITY) min[i] = NAN;
        if (max[i] == -INFINITY) max[i] = NAN;
    }
    return Py_None;
}

// this feels _painfully_ repetitive with binned_max()
PyObject*
binned_min(PyObject *self, PyObject *args) {
    long nx, ny;
    double xmin, xmax, ymin, ymax, val;
    PyObject *xarg, *yarg, *varg, *minarg;
    // PyArg_ParseTuple sets an exception on failure
    if (!PyArg_ParseTuple(args, "OOOOddddlll",
      &xarg, &yarg, &varg, &minarg,
      &xmin, &xmax, &ymin, &ymax, &nx, &ny)) { return NULL; }
    Iterface iter;
    Histspace space;
    PyArrayObject *arrs[3];
    int status = prep_binning(
        nx, ny, xmin, xmax, ymin, ymax, xarg, yarg, varg, arrs, 3, &iter, &space
    );
    if (status == 0) { return NULL; }
    double* min = PYARRAY_AS_DOUBLES(minarg);
    FOR_NDITER (&iter, &space, indices, &val) {
        if (min[indices[1] + ny * indices[0]] > val) {
            min[indices[1] + ny * indices[0]] = val;
        }
    }
    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with INFINITY;
    //  just have a special case up top
    for (long i = 0; i < nx * ny; i++) {
        if (min[i] == INFINITY) min[i] = NAN;
    }
    return Py_None;
}

PyObject*
binned_max(PyObject *self, PyObject *args) {
    long nx, ny;
    double xmin, xmax, ymin, ymax, val;
    PyObject *xarg, *yarg, *varg, *maxarg;
    // PyArg_ParseTuple sets an exception on failure
    if (!PyArg_ParseTuple(args, "OOOOddddlll",
      &xarg, &yarg, &varg, &maxarg,
      &xmin, &xmax, &ymin, &ymax, &nx, &ny)) { return NULL; }
    Iterface iter;
    Histspace space;
    PyArrayObject *arrs[3];
    int status = prep_binning(
        nx, ny, xmin, xmax, ymin, ymax, xarg, yarg, varg, arrs, 3, &iter, &space
    );
    if (status == 0) { return NULL; }
    double* max = PYARRAY_AS_DOUBLES(maxarg);
    FOR_NDITER (&iter, &space, indices, &val) {
        if (max[indices[1] + ny * indices[0]] < val) {
            max[indices[1] + ny * indices[0]] = val;
        }
    }
    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with -INFINITY;
    //  just have a special case up top
    for (long i = 0; i < nx * ny; i++) {
        if (max[i] == -INFINITY) max[i] = NAN;
    }
    return Py_None;
}

PyObject*
binned_median(PyObject *self, PyObject *args) {
    // TODO: there may be unnecessary copies happening here
    long nx, ny;
    double xmin, xmax, ymin, ymax;
    PyObject *xarg, *yarg, *varg, *medarg;
    // PyArg_ParseTuple sets an exception on failure
    if (!PyArg_ParseTuple(args, "OOOOddddlll",
      &xarg, &yarg, &varg, &medarg,
      &xmin, &xmax, &ymin, &ymax, &nx, &ny)) { return NULL; }
    Iterface iter;
    Histspace space;
    PyArrayObject *arrs[3];
    int status = prep_binning(
        nx, ny, xmin, xmax, ymin, ymax, xarg, yarg, varg, arrs, 3, &iter, &space
    );
    if (status == 0) { return NULL; }
    PyObject *numpy = PyImport_ImportModule("numpy");
    PyObject *unique = GETATTR(numpy, "unique");
    long arrsize = PyArray_SIZE(arrs[0]);
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
    double *vals = (double *) PyArray_DATA(arrs[2]);
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
    return Py_None;
}
