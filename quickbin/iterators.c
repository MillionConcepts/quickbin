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
    const double ibounds[static 2],
    const double jbounds[static 2],
    const long ni,
    const long nj
) {
    space->iscl = (double) ni / (ibounds[1] - ibounds[0]);
    space->jscl = (double) nj / (jbounds[1] - jbounds[0]);
    space->imin = ibounds[0];
    space->jmin = jbounds[0];
    space->ni = ni;
    space->nj = nj;
}

bool
for_nditer_big_step(
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
    return true;
}
