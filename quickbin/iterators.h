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

bool for_nditer_big_step(long[static 2], Iterface *, const Histspace *,
                         double *val);

static inline bool
for_nditer_step(
    long indices[static 2],
    Iterface *iter,
    const Histspace *space,
    double *val
) {
    if (iter->size == 0) {
        if (!for_nditer_big_step(indices, iter, space, val))
            return false;
    }
    hist_index(iter, space, indices);
    if (val)
        *val = *(double *) iter->data[2];
    iter->size -= 1;
    stride(iter);
    return true;
}

#define FOR_NDITER(ITER, SPACE, IXS, VAL)               \
    for (long IXS[2] = {-1, -1};                        \
         for_nditer_step(IXS, ITER, SPACE, VAL);        \
         )

#define FOR_NDITER_COUNT(ITER, SPACE, IXS)              \
    for (long IXS[2] = {-1, -1};                        \
         for_nditer_step(IXS, ITER, SPACE, NULL);       \
         )

#endif // ITERATORS_H
