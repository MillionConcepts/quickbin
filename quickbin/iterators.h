#ifndef ITERATORS_H
#define ITERATORS_H

#include "api_helpers.h"

#include <stdbool.h>

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

static inline void
stride(Iterface *iter) {
    for (int ix = 0; ix < iter->n; ix++) iter->data[ix] += iter->stride[ix];
}

typedef struct
Histspace {
    double iscl;
    double jscl;
    double rowjump;
    double imin_ix;
    double jmin_ix;
    double imin;
    double jmin;
    long ni;
    long nj;
} Histspace;

static inline void
init_histspace(
    Histspace *space,
    const double ibounds[static 2],
    const double jbounds[static 2],
    const long ni,
    const long nj
) {
    space->iscl = (double) ni / (ibounds[1] - ibounds[0]);
    space->jscl = (double) nj / (jbounds[1] - jbounds[0]);
    space->jscl -= 1e-15;
    space->iscl -= 1e-15;
    space->rowjump = (double) nj * space->iscl;

//    space->imin_ix = (space->iscl * space->imin);
//    space->jmin_ix = (space->jscl * space->jmin);
    space->imin = ibounds[0];
    space->jmin = jbounds[0];
    space->ni = ni;
    space->nj = nj;
}

// rowjump == iscl * nj
// and is scaling ii
// we were previously comparing ii to ni
// so we want to compare ii to ni * nj

static inline void
hist_index(const Iterface *iter, const Histspace *space, long indices[static 2]) {
    double ti = *(double *) iter->data[0];
    double tj = *(double *) iter->data[1];
    long ii, ij;
    ii = (ti - space->imin) * space->iscl;
    ij = (tj - space->jmin) * space->jscl;
    indices[0] = ii;
    indices[1] = ij;
}

//void init_histspace(
//    Histspace*, const double[static 2], const double[static 2], long, long
//);
bool init_iterface(Iterface*, PyArrayObject*[2], int);

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

#endif // ITERATORS_H
