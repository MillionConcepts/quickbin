#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

// Python CAPI shorthand

#define GETATTR PyObject_GetAttrString

#define ValueError PyExc_ValueError
#define TypeError PyExc_TypeError

// impl note: returns 0 instead of NULL so it can be used in functions
// that return pointers, functions that return integers, and functions
// that return booleans
#define PYRAISE(exc_class, msg) do {            \
        PyErr_SetString(exc_class, msg);        \
        return 0;                               \
    } while (0)

#define NP_ARGSORT(ARR) \
    PyArray_ArgSort((PyArrayObject *) ARR, 0, NPY_QUICKSORT)

#define PYCALL_1(FUNC, ARG) \
    PyObject_CallFunctionObjArgs(FUNC, (PyObject *) ARG, NULL);

// type of individual binning functions
typedef PyObject* (*BINFUNC)
    (PyArrayObject**, const double*, const double*, const long, const long, const unsigned int);

// Operations that can be performed by genhist.
#define GH_COUNT        1u
#define GH_SUM          2u
#define GH_MEAN         4u
#define GH_STD          8u
#define GH_MEDIAN      16u
#define GH_MIN         32u
#define GH_MAX         64u

#define GH_ALL        ( GH_COUNT | GH_SUM | GH_MEAN | GH_STD \
                      | GH_MEDIAN | GH_MIN | GH_MAX )

// Mapping between the GH_* bits and strings.
struct genhist_op_name {
    unsigned short opbit;
    char name[12];
};
static const struct genhist_op_name GENHIST_OP_NAMES[] = {
    { GH_COUNT,         "count"  },
    { GH_SUM,           "sum"    },
    { GH_MEAN,          "mean"   },
    { GH_STD,           "std"    },
    { GH_MEDIAN,        "median" },
    { GH_MIN,           "min"    },
    { GH_MAX,           "max"    },
    { 0,                ""       },
};

// Expose the GH_ values to Python as a mapping from names to bits.
static bool
make_ops_mapping(PyObject *module) {
    PyObject *opsdict = PyDict_New();
    if (!opsdict)
        return false;

    for (const struct genhist_op_name *op = GENHIST_OP_NAMES; op->opbit; op++) {
        PyObject *val = PyLong_FromLong(op->opbit);
        if (!val) {
            Py_DECREF(opsdict);
            return false;
        }
        if (PyDict_SetItemString(opsdict, op->name, val)) {
            Py_DECREF(val);
            Py_DECREF(opsdict);
            return false;
        }
        // PyDict_SetItemString *does not* take ownership of val
        Py_DECREF(val);
    }

    PyObject *opsmap = PyDictProxy_New(opsdict);
    Py_DECREF(opsdict);
    if (PyModule_AddObject(module, "OPS", opsmap)) {
        Py_DECREF(opsmap);
        return false;
    }

    // PyModule_AddObject takes our ref to opsmap on success
    return true;
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
init_iterface(Iterface *iter, PyArrayObject *arrays[2], int n_arrays) {
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

static inline void
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
init_histspace(
    Histspace *space,
    const double xbounds[static 2], const double ybounds[static 2],
    const long nx, const long ny
) {
    space->xscl = (double) nx / (xbounds[1] - xbounds[0]);
    space->yscl = (double) ny / (ybounds[1] - ybounds[0]);
    space->xmin = xbounds[0];
    space->ymin = ybounds[0];
    space->nx = nx;
    space->ny = ny;
}

static inline void
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

static inline void
free_all(int n, void **ptrs) {
    for (int i = 0; i < n; i++) free(ptrs[i]);
}

#define FREE_ALL(...) free_all ( \
    sizeof((void *[]){__VA_ARGS__}) / sizeof(void *),   \
    (void *[]){__VA_ARGS__}                             \
)

static inline void
decref_all(int n, void **ptrs) {
    for (int i = 0; i < n; i++) Py_DECREF((PyObject *) ptrs[i]);
}

#define DECREF_ALL(...) decref_all (                     \
   sizeof((void *[]){__VA_ARGS__}) / sizeof(void *),     \
   (void *[]){__VA_ARGS__}                               \
)


static inline void
destroy_ndarray(PyArrayObject *arr) {
    free(PyArray_DATA(arr));
    Py_SET_REFCNT(arr, 0);
}

static inline void
destroy_all_ndarrays(int n, void **ptrs) {
    for (int i = 0; i < n; i++) destroy_ndarray((PyArrayObject *) ptrs[i]);
}

#define DESTROY_ALL_NDARRAYS(...) destroy_all_ndarrays (     \
   sizeof((void *[]){__VA_ARGS__}) / sizeof(void *),     \
   (void *[]){__VA_ARGS__}                               \
)

static inline void
decref_arrays(long n_arrays, PyArrayObject** arrays) {
    for (long i = 0; i < n_arrays; i++) Py_DECREF(arrays[i]);
}

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

static inline int
doublecomp(const void *a, const void *b) {
    double *aval = (double *) a, *bval = (double *) b;
    if (*aval > *bval) return 1;
    if (*bval > *aval) return -1;
    return 0;
}
static inline void
assign_countsum(double *count, double *sum, long index, double val) {
    count[index] += 1;
    sum[index] += val;
}

// this helper function should be used only on a 'whatever' that has been
// created by the caller and which it intends to make no further use of
// before returning it to Pythonland. It sets whatever's pythons refcount
// to 1 to ensure that only the reference to it held by outdict 'matters',
// but this only works if no one else yet knows about the object.
static inline void
set_output_item(void *whatever, PyObject *outdict, char *objname) {
    PyObject *py_obj = (PyObject *) whatever;
    PyDict_SetItemString(outdict, objname, py_obj);
    Py_SET_REFCNT(py_obj, 1);
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

static PyArrayObject*
init_ndarray2d(
    const npy_intp *dims, const npy_intp dtype, const npy_intp fill
) {
    PyArrayObject *arr2d = (PyArrayObject *) PyArray_SimpleNew(2, dims, dtype);
    PyArray_FILLWBYTE(arr2d, fill);
    return arr2d;
}

static PyArrayObject*
init_ndarray1d(const npy_intp size, const npy_intp dtype, const npy_intp fill) {
    PyArrayObject *arr1d =
        (PyArrayObject *) PyArray_SimpleNew(1, (npy_intp []){size}, dtype);
    PyArray_FILLWBYTE(arr1d, fill);
    return arr1d;
}

static inline bool
for_nditer_step(
    long indices[static 2],
    Iterface *iter,
    const Histspace *space,
    double *val
) {
    while (iter->size == 0) {
        // A little kludge:
        // if indices[] == { -1, -1 , -1}, then we are before the very first
        // iteration and we should *not* call iternext.
        // NOTE: it is possible for *iter->sizep to be zero, hence the
        // while loop.
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
for_nditer_step_count(
    long indices[static 2],
    Iterface *iter,
    const Histspace *space
) {
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
    PyArrayObject *arrs[static 2],
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny,
    const unsigned int _ignored
) {
    Iterface iter;
    Histspace space;
    if (!init_iterface(&iter, arrs, 2)) return NULL;
    init_histspace(&space, xbounds, ybounds, nx, ny);
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
    PyArrayObject *arrs[static 3],
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny,
    const unsigned int _ignored
) {
    Iterface iter;
    Histspace space;
    if (!init_iterface(&iter, arrs, 3)) return NULL;
    init_histspace(&space, xbounds, ybounds, nx, ny);
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
    PyArrayObject *arrs[static 3],
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny,
    const unsigned int opmask
) {
    // We can do less work if we are only asked for count or for sum.
    if (opmask == GH_COUNT) {
        return binned_count(arrs, xbounds, ybounds, nx, ny, opmask);
    } else if (opmask == GH_SUM) {
        return binned_sum(arrs, xbounds, ybounds, nx, ny, opmask);
    }

    Iterface iter;
    Histspace space;
    if (!init_iterface(&iter, arrs, 3)) return NULL;
    init_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *countarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    PyArrayObject *sumarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
    double *count = (double *)PyArray_DATA(countarr);
    double *sum = (double *)PyArray_DATA(sumarr);
    double val;
    FOR_NDITER (&iter, &space, indices, &val) {
        assign_countsum(count, sum, indices[1] + indices[0] * ny, val);
    }
    PyObject *outdict = PyDict_New();
    if (opmask & GH_MEAN) {
        PyArrayObject *meanarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
        populate_meanarr(nx * ny, count, sum, (double *) PyArray_DATA(meanarr));
        set_output_item(meanarr, outdict, "mean");
    }
    if (opmask & GH_SUM) set_output_item(sumarr, outdict, "sum");
    else destroy_ndarray(sumarr);
    if (opmask & GH_COUNT) set_output_item(countarr, outdict, "count");
    else destroy_ndarray(countarr);
    return outdict;
}

static PyObject*
binned_std(
    PyArrayObject *arrs[static 3],
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny,
    const unsigned int opmask
) {
    Iterface iter;
    Histspace space;
    if (!init_iterface(&iter, arrs, 3)) return NULL;
    init_histspace(&space, xbounds, ybounds, nx, ny);
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
    set_output_item(stdarr, outdict, "std");
    if (opmask & GH_MEAN) {
        PyArrayObject *meanarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 0);
        populate_meanarr(nx * ny, count, sum, (double *) PyArray_DATA(meanarr));
        set_output_item(meanarr, outdict, "mean");
    }
    if (opmask & GH_SUM) set_output_item(sumarr, outdict, "sum");
    else destroy_ndarray(sumarr);
    if (opmask & GH_COUNT) set_output_item(countarr, outdict, "count");
    else destroy_ndarray(countarr);
    return outdict;
}

static PyObject*
binned_minmax(
    PyArrayObject *arrs[static 3],
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny,
    const unsigned int _ignored
    // this feels _painfully_ repetitive with binned_min() and binned_max()
) {
    Iterface iter;
    Histspace space;
    if (!init_iterface(&iter, arrs, 3)) return NULL;
    init_histspace(&space, xbounds, ybounds, nx, ny);
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
    set_output_item(minarr, outdict, "min");
    set_output_item(maxarr, outdict, "max");
    return outdict;
}

static PyObject*
binned_min(
    PyArrayObject *arrs[static 3],
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny,
    const unsigned int _ignored
    // this feels _painfully_ repetitive with binned_max()
) {
    Iterface iter;
    Histspace space;
    if (!init_iterface(&iter, arrs, 3)) return NULL;
    init_histspace(&space, xbounds, ybounds, nx, ny);
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
    PyArrayObject *arrs[static 3],
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny,
    const unsigned int _ignored
    // this feels _painfully_ repetitive with binned_min()
) {
    Iterface iter;
    Histspace space;
    if (!init_iterface(&iter, arrs, 3)) return NULL;
    init_histspace(&space, xbounds, ybounds, nx, ny);
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
    PyArrayObject *arrs[static 3],
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny,
    const unsigned int _ignored
) {
    // TODO: there may be unnecessary copies happening here
    PyObject *numpy = PyImport_ImportModule("numpy");
    PyObject *unique = GETATTR(numpy, "unique");
    Iterface iter;
    Histspace space;
    init_histspace(&space, xbounds, ybounds, nx, ny);
    PyArrayObject *axes[2] = {arrs[0], arrs[1]};
    init_iterface(&iter, axes, 2);
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
    PyArrayObject *xdig_sortarr = (PyArrayObject *) NP_ARGSORT(xdig_arr);
    // TODO: ensure that these remain NULL when cast to PyArrayObject in
    //  Pythonland failure cases
    if (xdig_sortarr == NULL) return NULL;
    long *xdig_sort = (long *)PyArray_DATA(xdig_sortarr);
    PyArrayObject *xdig_uniqarr = (PyArrayObject *) PYCALL_1(unique, xdig_arr);
    // TODO: laboriously decrement various references in these failure cases
    if (xdig_uniqarr == NULL) return NULL;
    long nx_uniq = PyArray_SIZE(xdig_uniqarr);
    long *xdig_uniq = (long *) PyArray_DATA(xdig_uniqarr);
    DECREF_ALL(unique, numpy);
    double *vals = (double *) PyArray_DATA(arrs[2]);
    PyArrayObject *medarr = init_ndarray1d(nx * ny, NPY_DOUBLE, 16);
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
    DESTROY_ALL_NDARRAYS(xdig_uniqarr, xdig_sortarr, ydig_arr, xdig_arr);
    return (PyObject *) medarr;
}

static bool
check_opmask(
    const long int raw_opmask,
    unsigned int *p_opmask,
    BINFUNC *p_binfunc
) {
    if (raw_opmask <= 0 || raw_opmask > GH_ALL) {
        PYRAISE(ValueError, "op bitmask out of range");
    }

    // if we get here, this conversion will not truncate
    unsigned int opmask = (unsigned int) raw_opmask;

    if (opmask & GH_MEDIAN) {
        if (opmask != GH_MEDIAN) {
            PYRAISE(ValueError, "median can only be computed alone");
        }
        *p_opmask = opmask;
        *p_binfunc = binned_median;
        return true;
    }

    if (opmask & (GH_MIN | GH_MAX)) {
        if (opmask & ~(GH_MIN | GH_MAX)) {
            PYRAISE(ValueError,
                    "min/max cannot be computed with non-min/max stats");
        }
        *p_opmask = opmask;
        if (opmask == GH_MIN)      *p_binfunc = binned_min;
        else if (opmask == GH_MAX) *p_binfunc = binned_max;
        else /* GH_MIN | GH_MAX */ *p_binfunc = binned_minmax;
        return true;
    }

    *p_opmask = opmask;
    if (opmask & GH_STD)
        *p_binfunc = binned_std;
    else
        *p_binfunc = binned_countvals;
    return true;
}


static PyObject*
genhist(PyObject *self, PyObject *args) {
    long nx, ny, long_mask;
    double xmin, xmax, ymin, ymax;
    PyObject *x_y_val_tuple;
    if (
        !PyArg_ParseTuple(args, "Oddddlll",
        &x_y_val_tuple, &xmin, &xmax,
        &ymin, &ymax, &nx, &ny, &long_mask)
    ) {
        return NULL; // PyArg_ParseTuple has set an exception
    }

    unsigned int opmask = 0;
    BINFUNC binfunc = 0;
    if (!check_opmask(long_mask, &opmask, &binfunc)) {
        return NULL; // check_opmask has set an exception
    }

    // NOTE: doing this silly-looking thing because PyArg_ParseTuple
    //  will not interpret python None as 'O'
    PyObject *x_arg = PyList_GetItem(x_y_val_tuple, 0);
    PyObject *y_arg = PyList_GetItem(x_y_val_tuple, 1);
    PyObject *val_arg = PyList_GetItem(x_y_val_tuple, 2);
    if (Py_IsNone(val_arg) && opmask != GH_COUNT) {
        PYRAISE(TypeError, "vals may only be None when only computing count");
    }

    PyArrayObject *arrays[3];
    arrays[0] = (PyArrayObject *) PyArray_FROM_O(x_arg);
    arrays[1] = (PyArrayObject *) PyArray_FROM_O(y_arg);
    long n_arrs = (opmask == GH_COUNT) ? 2 : 3;
    // TODO: are these decrefs necessary in these failure branches?
    if (n_arrs == 3) {
        arrays[2] = (PyArrayObject *) PyArray_FROM_O(val_arg);
        if (check_arrs(arrays, n_arrs) == false){
            DECREF_ALL(arrays[0], arrays[1], arrays[2]);
            return NULL;
        }
    }
    else if (check_arrs(arrays, 2) == false) {
        DECREF_ALL(arrays[0], arrays[1]);
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
    import_array();
    PyObject *m = PyModule_Create(&quickbin_core_mod);
    if (!m)
        return NULL;
    if (!make_ops_mapping(m)) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
