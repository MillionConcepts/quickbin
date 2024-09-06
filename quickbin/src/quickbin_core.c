#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

// TODO: is there a less gross way to load this global
PyObject *op_enum;

static void
load_op_enum() {
    PyObject *definitions = PyImport_ImportModule("quickbin.definitions");
    op_enum = PyObject_GetAttrString(definitions, "Ops");
}

static inline long
opval(char *name) {
    return PyLong_AsLong(
        PyObject_GetAttrString(PyObject_GetAttrString(op_enum, name), "value")
    );
}

static inline bool
opneeds(long opmask, char *name) {
    return (opmask & opval(name)) != 0;
}

typedef struct
Iterface {
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char **data;
    npy_intp *stride;
    npy_intp *sizep;
    npy_intp size;
    int n;
} Iterface;


static bool
make_iterface(Iterface *iter, PyArrayObject *arrays[], int n_arrays) {
    PyArray_Descr* dtypes[n_arrays];
    npy_uint32 op_flags[n_arrays];
    for (int i = 0; i < n_arrays; i++) {
        dtypes[i] = PyArray_DESCR(arrays[i]);
        op_flags[i] = NPY_ITER_READONLY;
    }
    iter->iter = NpyIter_AdvancedNew(
        n_arrays, arrays, NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
        NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, dtypes, -1, NULL,
        NULL, 0
    );
    if (! iter->iter) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't construct iterator");
        return false;
    }
    iter->iternext = NpyIter_GetIterNext(iter->iter, NULL);
    if (! iter->iternext) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't construct iteration");
        NpyIter_Deallocate(iter->iter);
        return false;
    }
    iter->data = NpyIter_GetDataPtrArray(iter->iter);
    iter->stride = NpyIter_GetInnerStrideArray(iter->iter);
    iter->sizep = NpyIter_GetInnerLoopSizePtr(iter->iter);
    iter->n = n_arrays;
    iter->size = 0;
    return true;
}

static void
stride(Iterface *iter) {
    for (int i = 0; i < iter->n; i++) iter->data[i] += iter->stride[i];
}

typedef struct
Histspace {
    double xscl;
    double yscl;
    double xmin;
    double ymin;
    long nx;
    long ny;
} Histspace;


static void
make_histspace(
    Histspace *space, const double xbounds[2], const double ybounds[2],
    const long nx, const long ny
) {
    space->xscl = (double) nx / (xbounds[1] - xbounds[0]);
    space->yscl = (double) ny / (ybounds[1] - ybounds[0]);
    space->xmin = xbounds[0];
    space->ymin = ybounds[0];
    space->nx = nx;
    space->ny = ny;
}

static void
hist_index(const Iterface *iter, const Histspace *space, long indices[static 2]) {
    double tx = *(double *) iter->data[0];
    double ty = *(double *) iter->data[1];
    // TODO, maybe: make the bounds check modal instead of simply enforcing
    //  bounds range up top

    // --- DEAD BOUNDS CHECK CODE ---
    //    int inbounds = (
    //        tx >= (*space).xmin && tx < (*space).xmax
    //        && ty >= (*space).ymin && ty < (*space).ymax
    //    );
    //    long ix = -1;
    //    long iy = -1;
    // -- END DEAD BOUNDS CHECK CODE --

    long ix, iy;
//    if (inbounds) {  // DEAD
    ix = (tx - space->xmin) * space->xscl;
    iy = (ty - space->ymin) * space->yscl;
    if (ix == space->nx) ix -= 1;
    if (iy == space->ny) iy -= 1;
//    }  // DEAD
    indices[0] = ix;
    indices[1] = iy;
}

static void
free_all(int n, void **ptrs) {
    for (int i = 0; i < n; i++) free(ptrs[i]);
}

#define FREE_ALL(...) free_all ( \
    sizeof((void *[]){__VA_ARGS__}) / sizeof(void *),   \
    (void *[]){__VA_ARGS__}                             \
)

static void
decref_all(int n, void **ptrs) {
    for (int i = 0; i < n; i++) Py_DECREF((PyObject *) ptrs[i]);
}

#define DECREF_ALL(...) decref_all (                     \
   sizeof((void *[]){__VA_ARGS__}) / sizeof(void *),     \
   (void *[]){__VA_ARGS__}                               \
)

static void
decref_arrays(long n_arrays, PyArrayObject** arrays) {
    for (long i = 0; i < n_arrays; i++) Py_DECREF(arrays[i]);
}

static char
check_arrs(PyArrayObject *arrays[], long n_arrays) {
    npy_intp insize = PyArray_SIZE(arrays[0]);
    for (long i = 0; i < n_arrays; i++) {
        if (arrays[i] == NULL) {
            PyErr_SetString(PyExc_TypeError, "Couldn't parse an array");
            return 0;
        }
        if (PyArray_NDIM(arrays[i]) != 1) {
            PyErr_SetString(PyExc_TypeError, "Arrays must be of dimension 1");
            return 0;
        }
        if (PyArray_SIZE(arrays[i]) != insize) {
            PyErr_SetString(PyExc_TypeError, "Arrays must be of the same size");
            return 0;
        }
    }
    return 1;
}

static inline int
doublecomp(const void *a, const void *b) {
    double *aval = (double *) a, *bval = (double *) b;
    if (*aval > *bval) return 1;
    if (*bval > *aval) return -1;
    return 0;
}

#define ValueError PyExc_ValueError
#define TypeError PyExc_TypeError

#define pyraise(exc_class, msg) \
    PyErr_SetString(exc_class, msg); \
return NULL;

static inline void
assign_countsum(double *count, double *sum, long index, double val) {
    count[index] += 1;
    sum[index] += val;
}

#define np_argsort(arr) \
    PyArray_ArgSort((PyArrayObject *) arr, 0, NPY_QUICKSORT)

#define np_unique(arr) \
    PyObject_CallFunctionObjArgs(unique, arr, NULL);

static void
setitem_for_output(void *whatever, PyObject *outdict, char *objname) {
    PyObject *py_obj = (PyObject *) whatever;
    PyDict_SetItemString(outdict, objname, py_obj);
    Py_SET_REFCNT(py_obj, 1);
}

static void
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

static void
populate_stdarr(
    const long size, const double *count, const double *sum,
    const double *sqr, double *std
) {
    for (long i = 0; i < size; i++) {
        if (count[i] == 0) std[i] = NAN;
        else std[i] = stdev(count[i], sum[i], sqr[i]);
    }
}

static PyArrayObject*
init_ndarray2d(
    npy_intp *dims, npy_intp dtype, npy_intp fill
) {
    PyArrayObject *arr2d = (PyArrayObject *) PyArray_SimpleNew(2, dims, dtype);
    PyArray_FILLWBYTE(arr2d, fill);
    return arr2d;
}

static PyArrayObject*
init_ndarray1d(npy_intp size, npy_intp dtype, npy_intp fill) {
    PyArrayObject *arr1d =
        (PyArrayObject *) PyArray_SimpleNew(1, (npy_intp []){size}, dtype);
    PyArray_FILLWBYTE(arr1d, fill);
    return arr1d;
}

static inline bool
for_nditer_step(long indices[static 2], Iterface *iter, const Histspace *space, double *val) {
    while (iter->size == 0) {
        // A little kludge:
        // if indices[] == { -1, -1 , -1}, then we are before the very first
        // iteration and we should *not* call iternext.
        // NOTE: it is possible for *iter->sizep to be zero, hence the while loop.
        if (indices[0] == -1 && indices[1] == -1) {
            indices[1] = 0;
        } else if (!iter->iternext(iter->iter)) {
            NpyIter_Deallocate(iter->iter);
            return false;
        }
        iter->size = *iter->sizep;
    }
    hist_index(iter, space, indices);
    *val = *(double *) iter->data[2];
    iter->size -= 1;
    stride(iter);
    return true;
}

#define FOR_NDITER(ITER, SPACE, IXS, VAL)   \
    for (long IXS[2] = {-1, -1};       \
    for_nditer_step(IXS, ITER, SPACE, VAL); \
    )


// TODO: these are tedious special-case versions of the preceding
//  function/macro pair intended for counting. there is probably
//  a cleaner way to do this.

static inline bool
for_nditer_step_count(long indices[static 2], Iterface *iter, const Histspace *space) {
    while (iter->size == 0) {
        if (indices[0] == -1 && indices[1] == -1) {
            indices[1] = 0;
        } else if (!iter->iternext(iter->iter)) {
            NpyIter_Deallocate(iter->iter);
            return false;
        }
        iter->size = *iter->sizep;
    }
    hist_index(iter, space, indices);
    iter->size -= 1;
    stride(iter);
    return true;
}

#define FOR_NDITER_COUNT(ITER, SPACE, IXS)   \
    for (long IXS[2] = {-1, -1};       \
    for_nditer_step_count(IXS, ITER, SPACE); \
    )


static PyObject*
binned_count(
    PyArrayObject *arrs[2],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    long _ignored
) {
    Iterface iter;
    Histspace space;
    if (! make_iterface(&iter, arrs, 2)) return NULL;
    make_histspace(&space, xbounds, ybounds, nx, ny);
    long dims[2] = {nx, ny};
    PyArrayObject *countarr = init_ndarray2d(dims, NPY_LONG, 0);
    long *count = (long *) PyArray_DATA(countarr);
    FOR_NDITER_COUNT (&iter, &space, indices) {
        if (indices[0] >= 0) count[indices[1] + ny * indices[0]] += 1;
    }
    return (PyObject *) countarr;
}

static PyObject*
binned_sum(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    long _ignored
) {
    Iterface iter;
    Histspace space;
    if (! make_iterface(&iter, arrs, 3)) return NULL;
    make_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *sumarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *sum = (double *)PyArray_DATA(sumarr);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        if (indices[0] >= 0) sum[indices[1] + ny * indices[0]] += val;
    }
    return (PyObject *) sumarr;
}

static PyObject*
binned_countvals(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    long opmask
) {
    Iterface iter;
    Histspace space;
    if (! make_iterface(&iter, arrs, 3)) return NULL;
    make_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *countarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    PyArrayObject *sumarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *count = (double *)PyArray_DATA(countarr);
    double *sum = (double *)PyArray_DATA(sumarr);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * ny, val);
    }
    PyObject *outdict = PyDict_New();
    if (opneeds(opmask, "mean")) {
        PyArrayObject *meanarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
        double *mean = (double *) PyArray_DATA(meanarr);
        populate_meanarr(nx * ny, count, sum, mean);
        setitem_for_output(meanarr, outdict, "mean");
    }
    if (opneeds(opmask, "sum")) setitem_for_output(sumarr, outdict, "sum");
    else Py_SET_REFCNT(sumarr, 0);
    if (opneeds(opmask, "count")) setitem_for_output(countarr, outdict, "count");
    else Py_SET_REFCNT(countarr, 0);
    return outdict;
}

static PyObject*
binned_std(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    long opmask
) {
    Iterface iter;
    Histspace space;
    if (! make_iterface(&iter, arrs, 3)) return NULL;
    make_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *countarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    PyArrayObject *sumarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *count = (double *)PyArray_DATA(countarr);
    double *sum = (double *)PyArray_DATA(sumarr);
    // NOTE: no point constructing an ndarray for the sum of squares; we never
    //  return it to the caller (who would want it?)
    double *sqr = calloc(sizeof *sqr, nx * ny);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * ny, val);
        sqr[indices[1] + ny * indices[0]] += (val * val);
    }
    PyArrayObject *stdarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *std = (double *)PyArray_DATA(stdarr);
    populate_stdarr(nx * ny, count, sum, sqr, std);
    free(sqr);
    PyObject *outdict = PyDict_New();
    setitem_for_output(stdarr, outdict, "std");
    if (opneeds(opmask, "mean")) {
        PyArrayObject *meanarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
        double *mean = (double *) PyArray_DATA(meanarr);
        populate_meanarr(nx * ny, count, sum, mean);
        setitem_for_output(meanarr, outdict, "mean");
    }
    if (opneeds(opmask, "sum")) setitem_for_output(sumarr, outdict, "sum");
    else Py_SET_REFCNT(sumarr, 0);
    if (opneeds(opmask, "count")) setitem_for_output(countarr, outdict, "count");
    else Py_SET_REFCNT(countarr, 0);
    return outdict;
}

static PyObject*
binned_minmax(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    long _ignored
    // this feels _painfully_ repetitive with binned_min() and binned_max()
) {
    Iterface iter;
    Histspace space;
    if (! make_iterface(&iter, arrs, 3)) return NULL;
    make_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *maxarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *max = (double *) PyArray_DATA(maxarr);
    PyArrayObject *minarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *min = (double *) PyArray_DATA(minarr);
    for (long i = 0; i < nx * ny; i++) {
        max[i] = -INFINITY;
        min[i] = INFINITY;
    }
    double val;
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
    PyObject *outdict = PyDict_New();
    setitem_for_output(minarr, outdict, "min");
    setitem_for_output(maxarr, outdict, "max");
    return outdict;
}

static PyObject*
binned_min(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    long _ignored
    // this feels _painfully_ repetitive with binned_max()
) {
    Iterface iter;
    Histspace space;
    if (! make_iterface(&iter, arrs, 3)) return NULL;
    make_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *minarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *min = (double *) PyArray_DATA(minarr);
    for (long i = 0; i < nx * ny; i++) min[i] = INFINITY;
    double val;
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
    return (PyObject *) minarr;
}

static PyObject*
binned_max(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    long _ignored
    // this feels _painfully_ repetitive with binned_min()
) {
    Iterface iter;
    Histspace space;
    if (! make_iterface(&iter, arrs, 3)) return NULL;
    make_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *maxarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *max = (double *) PyArray_DATA(maxarr);
    for (long i = 0; i < nx * ny; i++) max[i] = -INFINITY;
    double val;
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
    return (PyObject *) maxarr;

}

static PyObject*
binned_median(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    long _ignored
) {
    // TODO: there may be unnecessary copies happening here
    PyObject *numpy = PyImport_ImportModule("numpy");
    PyObject *unique = PyObject_GetAttrString(numpy, "unique");
    Iterface iter;
    Histspace space;
    make_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *axes[2] = {arrs[0], arrs[1]};
    make_iterface(&iter, axes, 2);
    long arrsize = PyArray_SIZE(arrs[0]);
    // xdig and ydig are the bin indices of each value in our input x and y
    // arrays respectively. this is a cheaty version of a digitize-type
    // operation that works only because we always have regular bins.
    PyArrayObject *xdig_arr = init_ndarray1d(arrsize, NPY_LONG, 0);
    PyArrayObject *ydig_arr = init_ndarray1d(arrsize, NPY_LONG, 0);
    long *xdig = (long *)PyArray_DATA(xdig_arr);
    long *ydig = (long *)PyArray_DATA(ydig_arr);
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
    PyArrayObject *xdig_sortarr = (PyArrayObject *) np_argsort(xdig_arr);
    // TODO: ensure that these remain NULL when cast to PyArrayObject in
    //  Pythonland failure cases
    if (xdig_sortarr == NULL) return NULL;
    long *xdig_sort = (long *)PyArray_DATA(xdig_sortarr);
    PyArrayObject *xdig_uniqarr = (PyArrayObject *) np_unique(xdig_arr);
    // TODO: laboriously decrement various references in these failure cases
    if (xdig_uniqarr == NULL) return NULL;
    long nx_uniq = PyArray_SIZE(xdig_uniqarr);
    long *xdig_uniq = (long *) PyArray_DATA(xdig_uniqarr);
    DECREF_ALL(unique, numpy);
    double *vals = (double *) PyArray_DATA(arrs[2]);
    PyArrayObject *medarr = init_ndarray1d(nx * ny, NPY_DOUBLE, NAN);
    double *median = (double *) PyArray_DATA(medarr);
    long x_sort_ix = 0;
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
    DECREF_ALL(xdig_uniqarr, xdig_sortarr, ydig_arr, xdig_arr);
    return (PyObject *) medarr;
}

static bool
check_opmask(long opmask) {
    if ((opmask <= 0) || (opmask > 255)) {
        pyraise(ValueError, "op bitmask out of range")
    }
    if (opneeds(opmask, "median") && (opmask != opval("median"))) {
        pyraise(ValueError, "median can only be computed alone");
    }
    if (
        (opneeds(opmask, "max") || opneeds(opmask, "min"))
        && !(
            opmask == opval("max")
            || opmask == opval("min")
            || opmask == opval("min") + opval("max")
        )
    ) {
        pyraise(ValueError, "min/max cannot be computed with non-min/max stats")
    }
    return true;
}

static PyObject*
genhist(PyObject *self, PyObject *args) {
    load_op_enum();
    long nx, ny, opmask;
    double xmin, xmax, ymin, ymax;
    PyObject *x_arg, *y_arg, *val_arg;
    if (
        !PyArg_ParseTuple(args, "OOOddddlll",
        &x_arg, &y_arg, &val_arg, &xmin, &xmax,
        &ymin, &ymax, &nx, &ny, &opmask)
    ) {
        pyraise(TypeError, "Bad argument list")
    }
    // TODO: can't actually pass None! PyArg_ParseTuple won't treat None as 'O'.
    //  you need some special handling. so we have to pack it into a tuple or something.
//    if (Py_IsNone(val_arg) && opmask != opval("count")) {
//        pyraise(TypeError, "vals may only be None for 'count'")
//    }
    if (check_opmask(opmask) != true) return NULL;
    PyObject* (*binfunc)(PyArrayObject**, double*, double*, long, long, long);
    if (opneeds(opmask, "std")) binfunc = binned_std;
    else if (
        (opneeds(opmask, "mean"))
        || (opneeds(opmask, "count") & opneeds(opmask, "sum"))
    ) binfunc = binned_countvals;
    else if (opneeds(opmask, "sum")) binfunc = binned_sum;
    else if (opneeds(opmask, "count")) binfunc = binned_count;
    else if (opneeds(opmask, "min") && opneeds(opmask, "max")) binfunc = binned_minmax;
    else if (opneeds(opmask, "min")) binfunc = binned_min;
    else if (opneeds(opmask, "max")) binfunc = binned_max;
    else if (opneeds(opmask, "median")) binfunc = binned_median;
    else {
        pyraise(ValueError, "Unclassified bad op specification")
    }
    PyArrayObject *arrays[3];
    arrays[0] = (PyArrayObject *) PyArray_FROM_O(x_arg);
    arrays[1] = (PyArrayObject *) PyArray_FROM_O(y_arg);
    char ok;
    long n_arrs;
    if (opmask == 2) n_arrs = 2; else n_arrs = 3;
    if (n_arrs == 3) {
        arrays[2] = (PyArrayObject *) PyArray_FROM_O(val_arg);
        ok = check_arrs(arrays, n_arrs);
    }
    else ok = check_arrs(arrays, n_arrs);
    if (ok == 0) {
        // TODO: are these decrefs necessary in this branch?
        if (n_arrs == 2) DECREF_ALL(arrays[0], arrays[1]);
        else DECREF_ALL(arrays[0], arrays[1], arrays[2]);
        return NULL;
    }
    double xbounds[2] = {xmin, xmax};
    double ybounds[2] = {ymin, ymax};
    PyObject *binned_arr = binfunc(arrays, xbounds, ybounds, nx, ny, opmask);
    decref_arrays(n_arrs, arrays);
    if (binned_arr == NULL) {
        if (PyErr_Occurred() == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unclassified error in binning");
        }
        return NULL;
    }
    return binned_arr;
}

static PyMethodDef
QuickbinMethods[] = {
    {
        "genhist",
        (PyCFunction) genhist,
        METH_VARARGS,
        "Generalized histogram function."
    },
    {NULL, NULL, 0, NULL}
};

static struct
PyModuleDef quickbin_core_mod = {
    PyModuleDef_HEAD_INIT,
    "_quickbin_core",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
    or -1 if the module keeps state in global variables. */
    QuickbinMethods
};

PyMODINIT_FUNC PyInit__quickbin_core(void) {
    import_array()
    return PyModule_Create(&quickbin_core_mod);
}

// dummy main()
int
main(int argc, char *argv[]) {
    return 0;
}
