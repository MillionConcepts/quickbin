#ifndef OPMASK_H
#define OPMASK_H

// Operations that can be performed by genhist.
#define GH_COUNT        1u
#define GH_SUM          2u
#define GH_MEAN         4u
#define GH_STD          8u
#define GH_MEDIAN      16u
#define GH_MIN         32u

#define GH_MAX         64u

#define GH_ALL    ( GH_COUNT | GH_SUM | GH_MEAN | GH_STD \
                  | GH_MEDIAN | GH_MIN | GH_MAX )
// Mapping between the GH_* bits and strings.
struct genhist_op_name {
    unsigned short opbit;
    char name[12];
};

static const struct genhist_op_name GENHIST_OP_NAMES[] = {
        { GH_COUNT,         "count"  },
        { GH_SUM,           "sum"    },
        { GH_MEAN,          "mean"   },
        { GH_STD,           "std"    },
        { GH_MEDIAN,        "median" },
        { GH_MIN,           "min"    },
        { GH_MAX,           "max"    },
        { 0,                ""       },
};

#endif // OPMASK_H
