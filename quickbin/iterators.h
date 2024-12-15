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
    double imin;
    double jmin;
    long ni;
    long nj;
} Histspace;

static inline void
hist_index(const Iterface *iter, const Histspace *space, long indices[static 2]) {
    double ti = *(double *) iter->data[0];
    double tj = *(double *) iter->data[1];
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

    long ii, ij;
//    if (inbounds) {  // DEAD
    ii = (ti - space->imin) * space->iscl;
    ij = (tj - space->jmin) * space->jscl;
    if (ii == space->ni) ii -= 1;
    if (ij == space->nj) ij -= 1;
//    }  // DEAD
    indices[0] = ii;
    indices[1] = ij;
}

void init_histspace(
    Histspace*, const double[static 2], const double[static 2], long, long
);
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
