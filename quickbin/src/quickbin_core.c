#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <float.h>
#include <signal.h>

typedef struct ReturnSpec {
    int count;
    int sum;
    int mean;
    int std;
    int median;
    int min;
    int max;
} ReturnSpec;

static ReturnSpec init_returnspec(int opmask, ReturnSpec *spec) {
    (*spec).count = (opmask & 2) > 0;
    (*spec).sum = (opmask & 4) > 0;
    (*spec).mean = (opmask & 8) > 0;
    (*spec).std = (opmask & 16) > 0;
    (*spec).median = (opmask & 32) > 0;
    (*spec).min = (opmask & 64) > 0;
    (*spec).max = (opmask & 128) > 0;
    return (*spec);
}

typedef struct Iterface {
    char **data;
    npy_intp *stride;
    npy_intp *size;
    int initialized;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    int n;
} Iterface;

static void stride(Iterface iter) {
    for (int i = 0; i < iter.n; i++) iter.data[i] += iter.stride[i];
}

static Iterface make_iterface(PyArrayObject *arrays[], int n_arrays) {
    PyArray_Descr* dtypes[n_arrays];
    npy_uint32 op_flags[n_arrays];
    for (int i = 0; i < n_arrays; i++) {
        dtypes[i] = PyArray_DESCR(arrays[i]);
        op_flags[i] = NPY_ITER_READONLY;
    }
    Iterface iterface;
    iterface.initialized = 0;
    iterface.iter = NpyIter_AdvancedNew(
        n_arrays, arrays, NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
        NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, dtypes, -1, NULL,
        NULL, 0
    );
    if (iterface.iter == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't construct iterator");
        return iterface;
    }
    iterface.iternext = NpyIter_GetIterNext(iterface.iter, NULL);
    if (iterface.iternext == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't construct iteration");
        NpyIter_Deallocate(iterface.iter);
        return iterface;
    }
    iterface.data = NpyIter_GetDataPtrArray(iterface.iter);
    iterface.stride = NpyIter_GetInnerStrideArray(iterface.iter);
    iterface.size = NpyIter_GetInnerLoopSizePtr(iterface.iter);
    iterface.n = n_arrays;
    iterface.initialized = 1;
    return iterface;
}

typedef struct Histspace {
    double xscl;
    double yscl;
    double xmin;
    double ymin;
    double xmax;
    double ymax;
    long nx;
    long ny;
} Histspace;

static Histspace make_histspace(
    const double xbounds[2], const double ybounds[2], long nx, long ny
) {
    Histspace space;
    space.xscl = (double) nx / (xbounds[1] - xbounds[0]);
    space.yscl = (double) ny / (ybounds[1] - ybounds[0]);
    space.xmin = xbounds[0];
    space.xmax = xbounds[1];
    space.ymin = ybounds[0];
    space.ymax = ybounds[1];
    space.nx = nx;
    space.ny = ny;
    return space;
}

static void hist_index(
    const Iterface *iter, const Histspace *space, long *indices
) {
    double tx = *(double *) (*iter).data[0];
    double ty = *(double *) (*iter).data[1];
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
    ix = (tx - (*space).xmin) * (*space).xscl;
    iy = (ty - (*space).ymin) * (*space).yscl;
    if (ix == (*space).nx) ix -= 1;
    if (iy == (*space).ny) iy -= 1;
//    }  // DEAD
    indices[0] = ix;
    indices[1] = iy;
}

#define START_VARARG_ITERATION \
va_list args; \
va_start(args, n); \
for (int i = 0; i < n; i++) {

#define END_VARARG_ITERATION \
}                   \
va_end(args);

#define NDITER_START \
do { \
    npy_intp size = *iter.size; \
    while (size--) {       \
        long indices[2]; \
        hist_index(&iter, &space, indices);

#define NDITER_END \
        stride(iter); \
    } \
} while (iter.iternext(iter.iter)); \
NpyIter_Deallocate(iter.iter);

static void free_all(int n, ...) {
    START_VARARG_ITERATION
    free(va_arg(args, void*));
    END_VARARG_ITERATION
}

static void decref_all(int n, ...)
{
    START_VARARG_ITERATION
    Py_DECREF(va_arg(args, PyObject*));
    END_VARARG_ITERATION
}

static void decref_arrays(long n_arrays, PyArrayObject** arrays) {
    for (long i = 0; i < n_arrays; i++) Py_DECREF(arrays[i]);
}

static char check_arrs(PyArrayObject *arrays[], long n_arrays) {
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

int longcomp(const void *a, const void *b) {
    long *aval = (long *) a, *bval = (long *) b;
    if (*aval > *bval) return 1;
    if (*bval > *aval) return -1;
    return 0;
}

#define ValueError PyExc_ValueError
#define TypeError PyExc_TypeError

#define np_array PyArray_SimpleNewFromData

#define pyraise(exc_class, msg) \
PyErr_SetString(exc_class, msg); \
return NULL;

#define ASSIGN_COUNTSUM \
(*count)[indices[1] + ny * indices[0]] += 1; \
(*sum)[indices[1] + ny * indices[0]] += tw;

#define np_argsort(arr) \
PyArray_ArgSort((PyArrayObject *) arr, 0, NPY_QUICKSORT)   \

#define np_unique(arr) \
PyObject_CallFunctionObjArgs(unique, arr, NULL);


void free_wrap(void *capsule){
    void * obj = PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
    free(obj);
};

static void bind_array_destructor(PyObject *obj, void *mem, char *name) {
    PyObject *capsule = PyCapsule_New(
        mem, name, (PyCapsule_Destructor)&free_wrap
    );
    PyArray_SetBaseObject((PyArrayObject *) obj, capsule);
}


static void insert_output_array(
    double *c_arr, long shape[1], PyObject *outdict, char *objname
) {
    PyObject *py_arr = np_array(1, shape, NPY_DOUBLE, c_arr);
    bind_array_destructor(py_arr, (void *) c_arr, objname);
    PyDict_SetItemString(outdict, objname, py_arr);
    Py_SET_REFCNT(py_arr, 1);
}

static void np_to_arr(PyObject *np_obj, void *c_array) {
    PyArray_AsCArray(
        &np_obj,
        c_array,
        PyArray_DIMS((PyArrayObject *) np_obj),
        PyArray_NDIM((PyArrayObject *) np_obj),
        PyArray_DescrFromType(PyArray_TYPE((PyArrayObject *) np_obj))
    );
}

// TODO: why is this copy required to avoid getting the wrong shape?
#define arr_to_np_double(arr, shape) \
PyArray_Copy(                        \
    (PyArrayObject *) np_array(1, shape, NPY_DOUBLE, arr) \
)

void calculate_mean(
    long nx, long ny, const double *count, const double *val, double* mean
) {
    for (long i = 0; i < nx * ny; i++) {
        if (count[i] == 0) mean[i] = NAN;
        else mean[i] = val[i] / count[i];
    }
}

// TODO: as above, why is this copy required to avoid getting the wrong shape?
#define arr_to_np_long(arr, shape) \
PyArray_Copy( \
    (PyArrayObject *) np_array(1, shape, NPY_LONG, arr) \
)

static PyArrayObject *init_ndarray2d(
    npy_intp *dims, npy_intp dtype, npy_intp fill
) {
    PyArrayObject *arr2d = (PyArrayObject *) PyArray_SimpleNew(2, dims, dtype);
    PyArray_FILLWBYTE(arr2d, fill);
    return arr2d;
}

static PyObject* binned_count(
    PyArrayObject *arrs[2],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    ReturnSpec _ignored
) {
    Iterface iter = make_iterface(arrs, 2);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    long dims[2] = {nx, ny};
    PyArrayObject *countarr = init_ndarray2d(dims, NPY_LONG, 0);
    long *count = (long *)PyArray_DATA(countarr);
    NDITER_START
        if (indices[0] >= 0) count[indices[1] + ny * indices[0]] += 1;
    NDITER_END
    return (PyObject *) countarr;
}

static PyObject* binned_sum(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    ReturnSpec _ignored
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*sum)[nx * ny] = malloc(sizeof *sum);
    for (long i = 0; i < nx * ny; i++) (*sum)[i] = 0;
    NDITER_START
        if (indices[0] >= 0) {
            double tw = *(double *) iter.data[2];
            (*sum)[indices[1] + ny * indices[0]] += tw;
        }
    NDITER_END
    npy_intp outlen[1] = {nx * ny};
    PyObject *sumarr = np_array(1, outlen, NPY_DOUBLE, sum);
    bind_array_destructor(sumarr, sum, "sum");
    return sumarr;
}

static PyObject* binned_countvals(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    ReturnSpec spec
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*count)[nx * ny] = malloc(sizeof *count);
    double (*sum)[nx * ny] = malloc(sizeof *sum);
    for (long i = 0; i < nx * ny; i++) {
        (*count)[i] = 0;
        (*sum)[i] = 0;
    }
    NDITER_START
        double tw = *(double *) iter.data[2];
        ASSIGN_COUNTSUM
    NDITER_END
    npy_intp outshape[1] = {nx * ny};
    PyObject *output = PyDict_New();
    if (spec.mean == 1) {
        double (*mean)[nx * ny] = malloc(sizeof *mean);
        calculate_mean(nx, ny, *count, *sum, *mean);
        insert_output_array(*mean, outshape, output, "mean");
    }
    if (spec.sum == 1) {
        insert_output_array(*sum, outshape, output, "sum");
    } else free(sum);
    if (spec.count == 1) {
        insert_output_array(*count, outshape, output, "count");
    } else free(count);
    return output;
}

static PyObject* binned_std(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    ReturnSpec spec
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*count)[nx * ny] = malloc(sizeof *count);
    double (*sum)[nx * ny] = malloc(sizeof *sum);
    double (*sqr)[nx * ny] = malloc(sizeof *sqr);
    for (long i = 0; i < nx * ny; i++) {
        (*count)[i] = 0;
        (*sum)[i] = 0;
        (*sqr)[i] = 0;
    }
    NDITER_START
        double tw = *(double *) iter.data[2];
        ASSIGN_COUNTSUM
        (*sqr)[indices[1] + ny * indices[0]] += (tw * tw);
    NDITER_END
    double (*std)[nx * ny] = malloc(sizeof *std);
    for (long i = 0; i < nx * ny; i++) {
        if ((*count)[i] == 0) (*std)[i] = NAN;
        else {
            (*std)[i] = sqrt(
                ((*sqr)[i] * (*count)[i] - ((*sum)[i] * (*sum)[i]))
                / ((*count)[i] * (*count)[i])
            );
        }
    }
    free(sqr);
    npy_intp outshape[1] = {nx * ny};
    PyObject *output = PyDict_New();
    insert_output_array(*std, outshape, output, "std");
    if (spec.mean == 1) {
        double (*mean)[nx * ny] = malloc(sizeof *mean);
        calculate_mean(nx, ny, *count, *sum, *mean);
        insert_output_array(*mean, outshape, output, "mean");
    }
    if (spec.sum == 1) {
        insert_output_array(*sum, outshape, output, "sum");
    } else free(sum);
    if (spec.count == 1) {
        insert_output_array(*count, outshape, output, "count");
    } else free(count);
    return output;
}

static PyObject* binned_minmax(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    ReturnSpec _ignored
    // this feels _painfully_ repetitive with binned_min() and binned_max()
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*min)[nx * ny] = malloc(sizeof *min);
    double (*max)[nx * ny] = malloc(sizeof *max);
    for (long i = 0; i < nx * ny; i++) {
        (*min)[i] = DBL_MAX;
        (*max)[i] = DBL_MIN;
    }
    NDITER_START
        double tw = *(double *) iter.data[2];
        if ((*min)[indices[1] + ny * indices[0]] > tw) {
            (*min)[indices[1] + ny * indices[0]] = tw;
        }
        if ((*max)[indices[1] + ny * indices[0]] < tw) {
            (*max)[indices[1] + ny * indices[0]] = tw;
        }
    NDITER_END
    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with DBL_MAX / DBL_MIN;
    //  just have a special case up top
    for (long i = 0; i < nx * ny; i++) {
        if ((*min)[i] == DBL_MAX) (*min)[i] = NAN;
        if ((*max)[i] == DBL_MIN) (*max)[i] = NAN;
    }
    npy_intp outshape[1] = {nx * ny};
    PyObject *output = PyDict_New();
    insert_output_array(*min, outshape, output, "min");
    insert_output_array(*max, outshape, output, "max");
    return output;
}

static PyObject* binned_min(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    ReturnSpec _ignored
    // this feels _painfully_ repetitive with binned_max()
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*min)[nx * ny] = malloc(sizeof *min);
    for (long i = 0; i < nx * ny; i++) (*min)[i] = DBL_MAX;
    NDITER_START
        double tw = *(double *) iter.data[2];
        if ((*min)[indices[1] + ny * indices[0]] > tw) {
            (*min)[indices[1] + ny * indices[0]] = tw;
        }
    NDITER_END
    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with DBL_MAX;
    //  just have a special case up top
    for (long i = 0; i < nx * ny; i++) {
        if ((*min)[i] == DBL_MAX) (*min)[i] = NAN;
    }
    npy_intp outshape[1] = {nx * ny};
    PyObject *minarr = np_array(1, outshape, NPY_DOUBLE, min);
    bind_array_destructor(minarr, (void *) min, "min");
    return minarr;
}

static PyObject* binned_max(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    ReturnSpec _ignored
    // this feels _painfully_ repetitive with binned_min()
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*max)[nx * ny] = malloc(sizeof *max);
    for (long i = 0; i < nx * ny; i++) (*max)[i] = DBL_MIN;
    NDITER_START
        double tw = *(double *) iter.data[2];
        if ((*max)[indices[1] + ny * indices[0]] < tw) {
            (*max)[indices[1] + ny * indices[0]] = tw;
        }
    NDITER_END
    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with DBL_MIN
    for (long i = 0; i < nx * ny; i++) {
        if ((*max)[i] == DBL_MIN) (*max)[i] = NAN;
    }
    npy_intp outlen[1] = {nx * ny};
    PyObject *maxarr = np_array(1, outlen, NPY_DOUBLE, max);
    bind_array_destructor(maxarr, (void *) max, "max");
    return maxarr;
}

static PyObject* binned_median(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny,
    ReturnSpec _ignored
) {
    // TODO: there may be unnecessary copies happening here
    PyObject *xdig_obj, *xdig_sort_obj,
        *xdig_uniq_obj, *numpy, *unique;
    numpy = PyImport_ImportModule("numpy");
    unique = PyObject_GetAttrString(numpy, "unique");
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    PyArrayObject *axes[2] = {arrs[0], arrs[1]};
    Iterface iter = make_iterface(axes, 2);
    // NOTE: these should always be identically sized
    long arrsize = PyArray_SIZE(arrs[0]);
    long (*xdig)[arrsize] = malloc(sizeof *xdig);
    long (*ydig)[arrsize] = malloc(sizeof *ydig);
    for (long i = 0; i < arrsize; i++) {
        npy_intp itersize = *iter.size;
        long indices[2];
        hist_index(&iter, &space, indices);
        (*xdig)[i] = indices[0];
        (*ydig)[i] = indices[1];
        itersize--;
        stride(iter);
    }
    NpyIter_Deallocate(iter.iter);
    npy_intp arrshape[1] = {arrsize};
    // TODO: this is probably an unnecessary copy
    xdig_obj = arr_to_np_long(xdig, arrshape);
    long *xdig_sort, *xdig_uniq;
    xdig_sort_obj = np_argsort(xdig_obj);
    if (xdig_sort_obj == NULL) return NULL;
    np_to_arr(xdig_sort_obj, &xdig_sort);
    xdig_uniq_obj = np_unique(xdig_obj);
    if (xdig_uniq_obj == NULL) return NULL;
    long nx_uniq = PyArray_SIZE((PyArrayObject *) xdig_uniq_obj);
    np_to_arr(xdig_uniq_obj, &xdig_uniq);
    decref_all(4, xdig_obj, xdig_uniq_obj, unique, numpy);
    double *vals;
    np_to_arr((PyObject *) arrs[2], &vals);
    long x_sort_ix = 0;
    double (*medians)[nx * ny] = malloc(sizeof *medians);
    for (long mi = 0; mi < nx * ny; mi++) (*medians)[mi] = NAN;
    long elcount = 0;
    double* xvals;
    np_to_arr((PyObject *) arrs[0], &xvals);
    for (long xix = 0; xix < nx_uniq; xix++) {
        long xbin = xdig_uniq[xix];
        // TODO: is it actually more efficient to loop over the array once
        //  to count the bins, allocate xbin_indices of the actually-required
        //  size, and then loop over it again?
        long (*xbin_indices)[arrsize] = malloc(sizeof *xbin_indices);
        for (long i = 0; i < arrsize; i++) (*xbin_indices)[i] = 0;
        long xbin_elcount = 0;
        for(;;) {
            (*xbin_indices)[xbin_elcount] = xdig_sort[x_sort_ix];
            xbin_elcount += 1;
            if (x_sort_ix + 1 >= arrsize) break;
            x_sort_ix += 1;
            if ((*xdig)[xdig_sort[x_sort_ix]] != xbin) break;
        }
        if (xbin_elcount == 0) continue;
        long (*match_buckets)[ny][xbin_elcount] = malloc(sizeof *match_buckets);
        long (*match_count)[ny] = malloc(sizeof *match_count);
        for (long i = 0; i < ny; i++) (*match_count)[i] = 0;
        for (long j = 0; j < xbin_elcount; j++) {
            long ybin = (*ydig)[(*xbin_indices)[j]];
            (*match_buckets)[ybin][(*match_count)[ybin]] = (*xbin_indices)[j];
            (*match_count)[ybin] += 1;
        }
        for (long ybin = 0; ybin < ny; ybin++) {
            long binsize = (*match_count)[ybin];
            if (binsize == 0) continue;
            double (*binvals)[binsize] = malloc(sizeof *binvals);
            for (long ix_ix = 0; ix_ix < binsize; ix_ix++) {
                (*binvals)[ix_ix] = vals[(*match_buckets)[ybin][ix_ix]];
                elcount += 1;
            }
            qsort(binvals, binsize, sizeof(double), longcomp);
            double median;
            if (binsize % 2 == 1) median = (*binvals)[binsize / 2];
            else median = (
                (*binvals)[binsize / 2] + (*binvals)[binsize / 2 - 1]
            ) / 2;
            (*medians)[ybin + space.ny * xbin] = median;
            free(binvals);
        }
        free_all(3, match_buckets, match_count, xbin_indices);
    }
    npy_intp shape[1] = {nx * ny};
    PyObject *medarr = np_array(1, shape, NPY_DOUBLE, medians);
    bind_array_destructor(medarr, (void *) medians, "median");
    // note: not necessary to decrement references to these because
    //  we're explicitly freeing their underlying memory and nothing outisde
    //  of this routine knows about them
    free_all(3, xdig_sort, xdig, ydig);
    // np_to_arr steals references to arrs[2] and arrs[0]
    return medarr;
}

static PyObject* genhist(PyObject *self, PyObject *args) {
    long nx, ny;
    int opmask;
    double xmin, xmax, ymin, ymax;
    PyObject *x_arg, *y_arg, *val_arg;
    if (
        !PyArg_ParseTuple(args, "OOOddddlli",
        &x_arg, &y_arg, &val_arg, &xmin, &xmax,
        &ymin, &ymax, &nx, &ny, &opmask)
    ) {
        pyraise(TypeError, "Bad argument list")
    }
    // TODO: there must be a cleaner way to do this, right?
    if (Py_IsNone(val_arg) && opmask != 1) {
        pyraise(TypeError, "vals may only be None for 'count'")
    }
    if ((opmask <= 0) | (opmask > 255)) {
        pyraise(ValueError, "op bitmask out of range")
    }
    // TODO: this is so gross
    ReturnSpec spec;
    spec = init_returnspec(opmask, &spec);
    if ((spec.median == 1) && (opmask != 32)) {
        pyraise(ValueError, "median can only be computed alone");
    }
    if ((opmask >= 64) && (opmask % 64 != 0)) {
        pyraise(ValueError, "min/max cannot be computed with non-min/max stats")
    }
    PyObject* (*binfunc)(PyArrayObject**, double*, double*, long, long, ReturnSpec);
    if (spec.std == 1) binfunc = binned_std;
    else if ((spec.mean == 1) || (spec.count + spec.sum == 2)) {
        binfunc = binned_countvals;
    }
    else if (spec.sum == 1) binfunc = binned_sum;
    else if (spec.count == 1) binfunc = binned_count;
    else if ((spec.min + spec.max) == 2) binfunc = binned_minmax;
    else if (spec.min == 1) binfunc = binned_min;
    else if (spec.max == 1) binfunc = binned_max;
    else if (spec.median == 1) binfunc = binned_median;
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
        if (n_arrs == 2) {
            decref_all(2, arrays[0], arrays[1]);
        }
        else {
            decref_all(3, arrays[0], arrays[1], arrays[2]);
        }
        return NULL;
    }
    double xbounds[2] = {xmin, xmax};
    double ybounds[2] = {ymin, ymax};
    PyObject *binned_arr = binfunc(arrays, xbounds, ybounds, nx, ny, spec);
    decref_arrays(n_arrs, arrays);
    if (binned_arr == NULL) {
        if (PyErr_Occurred() == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Unclassified error in binning");
        }
        return NULL;
    }
    return binned_arr;
}

static PyMethodDef QuickbinMethods[] = {
    {
        "genhist",
        (PyCFunction) genhist,
        METH_VARARGS,
        "Generalized histogram function."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef quickbin_core_mod = {
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
int main(int argc, char *argv[]) {
    return 0;
}
