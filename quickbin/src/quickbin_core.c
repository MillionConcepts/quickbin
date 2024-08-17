#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <float.h>

enum HistOp {
    OP_COUNT = 0,
    OP_SUM = 1,
    OP_MEAN = 2,
    OP_STD = 3,
    OP_MEDIAN = 4,
    OP_MIN = 5,
    OP_MAX = 6,
    OP_MEDIAN2 = 7
};

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
    for (int i = 0; i < iter.n; i++) {
        iter.data[i] += iter.stride[i];
    }
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
        NULL, 0);
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

static inline void hist_index(Iterface *iter, Histspace *space, long *indices) {
    double tx = *(double *) (*iter).data[0];
    double ty = *(double *) (*iter).data[1];
    // TODO, maybe: make the bounds check modal instead of simply enforcing bounds range up top

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

static int check_arrs(PyArrayObject *arrays[], int n_arrays) {
    npy_intp insize = PyArray_SIZE(arrays[0]);
    for (int i = 0; i < n_arrays; i++) {
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

int longcomp(const void* a, const void* b) {
    long *aval = (long *) a, *bval = (long *) b;
    if (*aval > *bval) return 1;
    if (*bval > *aval) return -1;
    return 0;
}

#define np_digitize(arr1, arr2) \
PyObject_CallFunctionObjArgs( \
    digitize, (PyObject *) arr1, (PyObject *) arr2, Py_False, NULL \
)

#define np_argsort(arr) \
PyArray_ArgSort((PyArrayObject *) arr, 0, NPY_QUICKSORT);   \

#define np_unique(arr) \
PyObject_CallFunctionObjArgs(unique, arr, NULL);

static void np_to_arr(PyObject *np_obj, void *c_array) {
    PyArray_AsCArray(
            &np_obj,
            c_array,
            PyArray_DIMS((PyArrayObject *) np_obj),
            PyArray_NDIM((PyArrayObject *) np_obj),
            PyArray_DescrFromType(PyArray_TYPE((PyArrayObject *) np_obj))
    );
}

#define arr_to_np_double(arr, shape) \
PyArray_Copy( \
    (PyArrayObject *) PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE, arr) \
)


static PyArrayObject *init_ndarray2d(npy_intp *dims, npy_intp dtype, npy_intp fill) {
    PyArrayObject *arr2d = (PyArrayObject *) PyArray_SimpleNew(2, dims, dtype);
    PyArray_FILLWBYTE(arr2d, fill);
    return arr2d;
}

static PyObject* binned_count(
    PyArrayObject *arrs[2],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny
) {
    Iterface iter = make_iterface(arrs, 2);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    long dims[2] = {nx, ny};
    PyArrayObject *countarr = init_ndarray2d(dims, NPY_LONG, 0);
    long *count = (long *)PyArray_DATA(countarr);
    do {
        npy_intp size = *iter.size;
        while (size--) {
            long indices[2];
            hist_index(&iter, &space, indices);
            if (indices[0] >= 0) {
                count[indices[1] + ny * indices[0]] += 1;
            }
            stride(iter);
        }
    } while (iter.iternext(iter.iter));
    NpyIter_Deallocate(iter.iter);
    return (PyObject *) countarr;
}

static PyObject* binned_sum(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*count)[nx * ny] = malloc(sizeof *count);
    for (long i = 0; i < nx * ny; i++) (*count)[i] = 0;
    do {
        npy_intp size = *iter.size;
        while (size--) {
            long indices[2];
            hist_index(&iter, &space, indices);
            if (indices[0] >= 0) {
                double tw = *(double *) iter.data[2];
                (*count)[indices[1] + ny * indices[0]] += tw;
            }
            stride(iter);
        }
    } while (iter.iternext(iter.iter));
    NpyIter_Deallocate(iter.iter);
    npy_intp outlen[1] = {nx * ny};
    PyObject *countarr = PyArray_Copy(
        (PyArrayObject *)
        PyArray_SimpleNewFromData(1, outlen, NPY_DOUBLE, count)
    );
    return countarr;
}

static PyObject* binned_mean(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*count)[nx * ny] = malloc(sizeof *count);
    double (*val)[nx * ny] = malloc(sizeof *val);
    double (*mean)[nx * ny] = malloc(sizeof *mean);
    for (long i = 0; i < nx * ny; i++) {
        (*count)[i] = 0;
        (*val)[i] = 0;
        (*mean)[i] = 0;
    }
    do {
        npy_intp size = *iter.size;
        while (size--) {
            long indices[2];
            hist_index(&iter, &space, indices);
                double tw = *(double *) iter.data[2];
                (*count)[indices[1] + ny * indices[0]] += 1;
                (*val)[indices[1] + ny * indices[0]] += tw;
            stride(iter);
        }
    } while (iter.iternext(iter.iter));
    for (long i = 0; i < nx * ny; i++) {
        if ((*count)[i] == 0) {
            (*mean)[i] = NAN;
        } else {
            (*mean)[i] = (*val)[i] / (*count)[i];
        }
    }
    free(count);
    free(val);
    NpyIter_Deallocate(iter.iter);
    // this copy prevents the memory from being deallocated when we leave
    // function scope in C.
    // but can it ever actually be deallocated by Python?
    npy_intp outlen[1] = {nx * ny};
    PyObject *meanarr = PyArray_Copy(
        (PyArrayObject *)
        PyArray_SimpleNewFromData(1, outlen, NPY_DOUBLE, mean)
    );
    return meanarr;
}

#define ASSIGN_COUNTVAL \
(*count)[indices[1] + ny * indices[0]] += 1; \
(*val)[indices[1] + ny * indices[0]] += tw;


static PyObject* binned_std(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*count)[nx * ny] = malloc(sizeof *count);
    double (*val)[nx * ny] = malloc(sizeof *val);
    double (*sqr)[nx * ny] = malloc(sizeof *sqr);
    for (long i = 0; i < nx * ny; i++) {
        (*count)[i] = 0;
        (*val)[i] = 0;
        (*sqr)[i] = 0;
    }
    do {
        npy_intp size = *iter.size;
        while (size--) {
            long indices[2];
            hist_index(&iter, &space, indices);
            double tw = *(double *) iter.data[2];
            ASSIGN_COUNTVAL
            (*sqr)[indices[1] + ny * indices[0]] += (tw * tw);
            stride(iter);
        }
    } while (iter.iternext(iter.iter));
    double (*std)[nx * ny] = malloc(sizeof *std);
    for (long i = 0; i < nx * ny; i++) {
        if ((*count)[i] == 0) {
            (*std)[i] = NAN;
        } else {
            (*std)[i] = sqrt(
                ((*sqr)[i] * (*count)[i] - ((*val)[i] * (*val)[i]))
                / ((*count)[i] * (*count)[i])
            );
        }
    }
    free(count);
    free(val);
    free(sqr);
    NpyIter_Deallocate(iter.iter);
    npy_intp outlen[1] = {nx * ny};
    PyObject *stdarr = PyArray_Copy(
        (PyArrayObject *)
        PyArray_SimpleNewFromData(1, outlen, NPY_DOUBLE, std)
    );
    return stdarr;
}

static PyObject* binned_min(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny
    // this feels _painfully_ repetitive with binned_max()
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*min)[nx * ny] = malloc(sizeof *min);
    for (long i = 0; i < nx * ny; i++) {
        (*min)[i] = DBL_MAX;
    }
    do {
        npy_intp size = *iter.size;
        while (size--) {
            long indices[2];
            hist_index(&iter, &space, indices);
            double tw = *(double *) iter.data[2];
            if ((*min)[indices[1] + ny * indices[0]] > tw) {
                (*min)[indices[1] + ny * indices[0]] = tw;
            }
            stride(iter);
        }
    } while (iter.iternext(iter.iter));
    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with DBL_MAX;
    //  just have a special case up top
    for (long i = 0; i < nx * ny; i++) {
        if ((*min)[i] == DBL_MAX) (*min)[i] = NAN;
    }
    // see notes on this copy operation above
    npy_intp outlen[1] = {nx * ny};
    PyObject *minarr = PyArray_Copy(
        (PyArrayObject *)
        PyArray_SimpleNewFromData(1, outlen, NPY_DOUBLE, min)
    );
    return minarr;
}

static PyObject* binned_max(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny
    // this feels _painfully_ repetitive with binned_min()
) {
    Iterface iter = make_iterface(arrs, 3);
    if (iter.initialized == 0) return NULL;
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double (*max)[nx * ny] = malloc(sizeof *max);
    for (long i = 0; i < nx * ny; i++) {
        (*max)[i] = DBL_MIN;
    }
    do {
        npy_intp size = *iter.size;
        while (size--) {
            long indices[2];
            hist_index(&iter, &space, indices);
            double tw = *(double *) iter.data[2];
            if ((*max)[indices[1] + ny * indices[0]] < tw) {
                (*max)[indices[1] + ny * indices[0]] = tw;
            }
            stride(iter);
        }
    } while (iter.iternext(iter.iter));
    // TODO: this will produce NaNs in the perverse case where
    //  an array is filled entirely with DBL_MIN
    for (long i = 0; i < nx * ny; i++) {
        if ((*max)[i] == DBL_MIN) (*max)[i] = NAN;
    }
    // see notes on this copy operation above
    npy_intp outlen[1] = {nx * ny};
    PyObject *maxarr = PyArray_Copy(
        (PyArrayObject *)
        PyArray_SimpleNewFromData(1, outlen, NPY_DOUBLE, max)
    );
    return maxarr;
}


static PyObject* binned_median(
    PyArrayObject *arrs[3],
    double xbounds[2],
    double ybounds[2],
    long nx,
    long ny
) {
    PyObject *xdig_obj, *ydig_obj, *xdig_sort_obj,
        *xdig_uniq_obj, *ydig_uniq_obj,
        *numpy, *digitize, *unique,
        *xbin_obj, *ybin_obj;
    numpy = PyImport_ImportModule("numpy");
    digitize = PyObject_GetAttrString(numpy, "digitize");
    unique = PyObject_GetAttrString(numpy, "unique");
    Histspace space = make_histspace(xbounds, ybounds, nx, ny);
    double xbins[nx], ybins[ny];
    for (long i = 0; i < nx; i++) {
        xbins[i] = space.xmin + (double) i / space.xscl;
    }
    for (long i = 0; i < ny; i++) {
        ybins[i] = space.ymin + (double) i / space.yscl;
    }
    long xshape[1] = {nx};
    long yshape[1] = {ny};
    long arrsize = PyArray_SIZE(arrs[0]);
    xbin_obj = arr_to_np_double(xbins, xshape);
    ybin_obj = arr_to_np_double(ybins, yshape);
    xdig_obj = np_digitize(arrs[0], xbin_obj);
    ydig_obj = np_digitize(arrs[1], ybin_obj);
    long *xdig, *ydig, *xdig_sort;
    xdig_sort_obj = np_argsort(xdig_obj);
    np_to_arr(xdig_sort_obj, &xdig_sort);
    Py_DECREF(xdig_sort_obj);
    np_to_arr(xdig_obj, &xdig);
    np_to_arr(ydig_obj, &ydig);
    long *xdig_uniq, *ydig_uniq;
    xdig_uniq_obj = np_unique(xdig_obj);
    Py_DECREF(xdig_obj);
    long nx_uniq = PyArray_SIZE((PyArrayObject *) xdig_uniq_obj);
    np_to_arr(xdig_uniq_obj, &xdig_uniq);
    Py_DECREF(xdig_uniq_obj);
    ydig_uniq_obj = np_unique(ydig_obj);
    Py_DECREF(ydig_obj);
    long ny_uniq = PyArray_SIZE((PyArrayObject *) ydig_uniq_obj);
    np_to_arr(ydig_uniq_obj, &ydig_uniq);
    Py_DECREF(ydig_uniq_obj);
    Py_DECREF(unique);
    Py_DECREF(digitize);
    Py_DECREF(numpy);
    double *vals;
    np_to_arr((PyObject *) arrs[2], &vals);
    long x_sort_ix = 0;
    double (*medians)[nx * ny] = malloc(sizeof *medians);
    for (long mi = 0; mi < nx * ny; mi++) {
        (*medians)[mi] = NAN;
    }
    long elcount = 0;
    double* xvals;
    np_to_arr((PyObject *) arrs[0], &xvals);
    for (long xix = 0; xix < nx_uniq; xix++) {
        long xbin = xdig_uniq[xix] - 1;
        long (*outer_indices)[arrsize] = malloc(sizeof *outer_indices);
        long outer_label_size = 0;
        for(;;) {
            (*outer_indices)[outer_label_size] = xdig_sort[x_sort_ix];
            outer_label_size += 1;
            x_sort_ix += 1;
            if (x_sort_ix >= arrsize) break;
            if (xdig[xdig_sort[x_sort_ix]] != xbin + 1) break;
        }
        long (*xy_matchix)[ny_uniq][outer_label_size]
            = malloc(sizeof *xy_matchix);
        long (*xy_matchix_count)[ny_uniq]
            = malloc(sizeof *xy_matchix_count);
        for (long i = 0; i < ny_uniq; i++) {
            (*xy_matchix_count)[i] = 0;
        }
        for (long j = 0; j < outer_label_size; j++) {
            long ybin = ydig[(*outer_indices)[j]] - 1;
            (*xy_matchix)[ybin][(*xy_matchix_count)[ybin]] = (*outer_indices)[j];
            (*xy_matchix_count)[ybin] += 1;
        }
        for (long yix = 0; yix < ny_uniq; yix++) {
            long ybin = ydig_uniq[yix] - 1;
            long binsize = (*xy_matchix_count)[ybin];
            if (binsize == 0) continue;
            double (*binvals)[binsize] = malloc(sizeof *binvals);
            for (long ix_ix = 0; ix_ix < binsize; ix_ix++) {
                (*binvals)[ix_ix] = vals[(*xy_matchix)[ybin][ix_ix]];
                elcount += 1;
            }
            qsort(binvals, binsize, sizeof(double), longcomp);
            double median;
            if (binsize % 2 == 1) {
                median = (*binvals)[binsize / 2];
            } else {
                median = (
                    (*binvals)[binsize / 2] + (*binvals)[binsize / 2 - 1]
                ) / 2;
            }
            (*medians)[ybin + space.ny * xbin] = median;
            free(binvals);
        }
        free(xy_matchix);
        free(xy_matchix_count);
        free(outer_indices);
    }
    long shape[1] = {nx * ny};
    PyObject *medarr = arr_to_np_double(medians, shape);
    return medarr;
}

static PyObject* genhist(PyObject *self, PyObject *args) {
    long nx, ny;
    int op;
    double xmin, xmax, ymin, ymax;
    PyObject *x_arg, *y_arg, *val_arg;
    if (
        !PyArg_ParseTuple(args, "OOOddddlli",
              &x_arg, &y_arg, &val_arg, &xmin, &xmax,
              &ymin, &ymax, &nx, &ny, &op)
    ) {
        PyErr_SetString(PyExc_TypeError, "Bad argument");
        return NULL;
    }
    if (Py_IsNone(val_arg) && op != OP_COUNT) {
        PyErr_SetString(
            PyExc_TypeError,
            "value array may only be None for 'count' op"
        );
        return NULL;
    }
    PyArrayObject *arrays[3];
    arrays[0] = (PyArrayObject *) PyArray_FROM_O(x_arg);
    arrays[1] = (PyArrayObject *) PyArray_FROM_O(y_arg);
    int ok;
    if (op != OP_COUNT) {
        arrays[2] = (PyArrayObject *) PyArray_FROM_O(val_arg);
        ok = check_arrs(arrays, 3);
    }
    else ok = check_arrs(arrays, 2);
    if (ok == 0) {
        Py_DECREF(arrays);
        return NULL;
    }
    PyObject* (*binfunc)(PyArrayObject**, double*, double*, long, long);
    switch(op) {
        case OP_COUNT:
            binfunc = binned_count;
            break;
        case OP_SUM:
            binfunc = binned_sum;
            break;
        case OP_MEAN:
            binfunc = binned_mean;
            break;
        case OP_STD:
            binfunc = binned_std;
            break;
        case OP_MEDIAN:
            binfunc = binned_median;
            break;
        case OP_MIN:
            binfunc = binned_min;
            break;
        case OP_MAX:
            binfunc = binned_max;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "unknown operation");
            return NULL;
    }
    double xbounds[2] = {xmin, xmax};
    double ybounds[2] = {ymin, ymax};
    PyObject *binned_arr = binfunc(arrays, xbounds, ybounds, nx, ny);
    if (binned_arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "unclassified error in binning");
        return NULL;
    }
    Py_DECREF(arrays);
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
    import_array();
    return PyModule_Create(&quickbin_core_mod);
}

// dummy main()
int main(int argc, char *argv[]) {
    return 0;
}
