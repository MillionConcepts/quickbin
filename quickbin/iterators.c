#include "iterators.h"

bool
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

void
init_histspace(
    Histspace *space,
    const double xbounds[static 2],
    const double ybounds[static 2],
    const long nx,
    const long ny
) {
    space->xscl = (double) nx / (xbounds[1] - xbounds[0]);
    space->yscl = (double) ny / (ybounds[1] - ybounds[0]);
    space->xmin = xbounds[0];
    space->ymin = ybounds[0];
    space->nx = nx;
    space->ny = ny;
}