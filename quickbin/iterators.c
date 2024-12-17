#include "iterators.h"

bool
init_iterface(Iterface *iter, PyArrayObject *arrays[], int n_arrays) {
    PyArray_Descr* dtypes[n_arrays];
    npy_uint32 op_flags[n_arrays];
    for (int ix = 0; ix < n_arrays; ix++) {
        dtypes[ix] = PyArray_DESCR(arrays[ix]);
        op_flags[ix] = NPY_ITER_READONLY;
    }
    iter->iter = NpyIter_AdvancedNew(
            n_arrays, arrays, NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
            NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, dtypes, -1, NULL,
            NULL, 0
    );
    if (! iter->iter) {
        return false;
    }
    iter->iternext = NpyIter_GetIterNext(iter->iter, NULL);
    if (! iter->iternext) {
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


