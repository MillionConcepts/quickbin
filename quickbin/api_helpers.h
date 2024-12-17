#ifndef API_HELPERS_H
#define API_HELPERS_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL quickbin_PyArray_API

#ifndef I_WILL_CALL_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <Python.h>
#include <stdbool.h>

// core Python CAPI shorthand

#define GETATTR PyObject_GetAttrString


#define PYCALL_1(FUNC, ARG) \
    PyObject_CallFunctionObjArgs(FUNC, (PyObject *) ARG, NULL);


// numpy C API shorthand

#define NP_ARGSORT(ARR) \
    PyArray_ArgSort((PyArrayObject *) ARR, 0, NPY_QUICKSORT)


static inline PyArrayObject*
init_ndarray1d(const npy_intp size, const npy_intp dtype, const npy_intp fill) {
    PyArrayObject *arr1d = (PyArrayObject *)
    PyArray_SimpleNew(1, (npy_intp []){size}, dtype);
    if (arr1d == NULL) return NULL;
    PyArray_FILLWBYTE(arr1d, fill);
    return arr1d;
}


static inline void
free_pointer_array(void *targets[], const int n) {
    for (int i = 0; i < n; i++) {
        if (targets[i] != NULL) free(targets[i]);
        targets[i] = NULL;
    }
}

typedef struct
ObjectCleanupInfo {
    bool delete_array_data;
    PyObject *object;
} ObjectCleanupInfo;


static inline int
prep_cleanup(
    ObjectCleanupInfo **cleanup,
    PyObject *target,
    const bool delete_array_data,
    const int ix
) {
    cleanup[ix]->object = target;
    cleanup[ix]->delete_array_data = delete_array_data;
    return ix + 1;
}


static inline void
clean_up_pyobjects(ObjectCleanupInfo *cleanup[], const int nclean) {
    for (int i = 0; i < nclean; i++) {
        if (cleanup[i]->delete_array_data == true) {
            free(PyArray_DATA((PyArrayObject *) cleanup[i]->object));
            Py_SET_REFCNT(cleanup[i]->object, 0);
        } else if (Py_REFCNT(cleanup[i]->object) != 0) {
            Py_DECREF(cleanup[i]->object);
        }
    }
}

static inline void
do_object_cleanup(
    void *tofree[],
    const int nfree,
    ObjectCleanupInfo **toclean,
    const int nclean,
    const int cleansize
) {
    free_pointer_array(tofree, nfree);
    clean_up_pyobjects(toclean, nclean);
    for (int i = 0; i < cleansize; i++) free(toclean[i]);
    free(toclean);
}

#define ABORT_IF_NULL(                                                        \
    MAYBE_NULL, OBJNAME, TOFREE, NFREE, PYOBJCLEAN, NPYOBJ, CLEANSZ           \
)                                                                             \
do {                                                                          \
    if (MAYBE_NULL == NULL) {                                                 \
        PyErr_SetString(PyExc_RuntimeError,                                   \
                        "OBJNAME initialization failed");                     \
        do_object_cleanup(TOFREE, NFREE, PYOBJCLEAN, NPYOBJ, CLEANSZ);                 \
        return NULL;                                                          \
    }                                                                         \
} while(0)


#endif // API_HELPERS_H
