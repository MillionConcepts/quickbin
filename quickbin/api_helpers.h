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

// core Python CAPI shorthand

#define GETATTR PyObject_GetAttrString
#define ValueError PyExc_ValueError

// impl note: returns 0 instead of NULL so it can be used in functions
// that return pointers, functions that return integers, and functions
// that return booleans
#define PYRAISE(exc_class, msg) do {        \
    PyErr_SetString(exc_class, msg);        \
    return 0;                               \
} while (0)

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

#define PYCALL_1(FUNC, ARG) \
    PyObject_CallFunctionObjArgs(FUNC, (PyObject *) ARG, NULL);


// numpy C API shorthand

#define NP_ARGSORT(ARR) \
    PyArray_ArgSort((PyArrayObject *) ARR, 0, NPY_QUICKSORT)


static inline PyArrayObject*
init_ndarray1d(const npy_intp size, const npy_intp dtype, const npy_intp fill) {
    PyArrayObject *arr1d = (PyArrayObject *)
    PyArray_SimpleNew(1, (npy_intp []){size}, dtype);
    PyArray_FILLWBYTE(arr1d, fill);
    return arr1d;
}

static inline void
destroy_ndarray(PyArrayObject *arr) {
    free(PyArray_DATA(arr));
    Py_SET_REFCNT(arr, 0);
}

static inline void
destroy_all_ndarrays(const int n, void **ptrs) {
    for (int i = 0; i < n; i++) destroy_ndarray((PyArrayObject *) ptrs[i]);
}

#define DESTROY_ALL_NDARRAYS(...) destroy_all_ndarrays ( \
   sizeof((void *[]){__VA_ARGS__}) / sizeof(void *),     \
   (void *[]){__VA_ARGS__}                               \
)


#endif // PI_HELPERS_H
