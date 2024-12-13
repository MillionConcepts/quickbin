#define I_WILL_CALL_IMPORT_ARRAY
#include "binning.h"
#include "opmask.h"

#include <stdbool.h>

// Expose the GH_ values to Python as a mapping from names to bits.
static bool
make_ops_mapping(PyObject *module) {
    PyObject *opsdict = PyDict_New();
    if (!opsdict)
        return false;
    for (const struct genhist_op_name *op = GENHIST_OP_NAMES; op->opbit; op++) {
        PyObject *val = PyLong_FromLong(op->opbit);
        if (!val) {
            Py_DECREF(opsdict);
            return false;
        }
        if (PyDict_SetItemString(opsdict, op->name, val)) {
            Py_DECREF(val);
            Py_DECREF(opsdict);
            return false;
        }
        // PyDict_SetItemString *does not* take ownership of val
        Py_DECREF(val);
    }

    PyObject *opsmap = PyDictProxy_New(opsdict);
    Py_DECREF(opsdict);
    if (PyModule_AddObject(module, "OPS", opsmap)) {
        Py_DECREF(opsmap);
        return false;
    }

    // PyModule_AddObject takes our ref to opsmap on success
    return true;
}

static PyMethodDef
QuickbinMethods[] = {
    {
        "_binned_count",
        (PyCFunction) binned_count,
        METH_FASTCALL,
        "Binned count function."
    },
    {
        "_binned_sum",
        (PyCFunction) binned_sum,
        METH_FASTCALL,
        "Binned sum function."
    },
    {
        "_binned_countvals",
        (PyCFunction) binned_countvals,
        METH_FASTCALL,
        "Binned count / sum / mean function."
    },
    {
        "_binned_minmax",
        (PyCFunction) binned_minmax,
        METH_FASTCALL,
        "Binned min + max function."
    },
    {
        "_binned_std",
        (PyCFunction) binned_std,
        METH_FASTCALL,
        "Binned standard deviation function."
    },
    {
        "_binned_median",
        (PyCFunction) binned_median,
        METH_FASTCALL,
        "Binned median function."
    },
    {NULL, NULL, 0, NULL}
};

static struct
PyModuleDef quickbin_core_mod = {
    PyModuleDef_HEAD_INIT,
    "quickbin_core",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
    or -1 if the module keeps state in global variables. */
    QuickbinMethods
};

PyMODINIT_FUNC PyInit_quickbin_core(void) {
    import_array();
    PyObject *m = PyModule_Create(&quickbin_core_mod);
    if (!m)
        return NULL;
    if (!make_ops_mapping(m)) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}

// dead code for reference

//static bool
//check_opmask(
//    const long int raw_opmask,
//    unsigned int *p_opmask,
//    BINFUNC **p_binfunc
//) {
//    if (raw_opmask <= 0 || raw_opmask > GH_ALL) {
//        PYRAISE(ValueError, "op bitmask out of range");
//    }
//    // if we get here, this conversion will not truncate
//    unsigned int opmask = (unsigned int) raw_opmask;
//
//    if (opmask & GH_MEDIAN) {
//        if (opmask != GH_MEDIAN) {
//            PYRAISE(ValueError, "median can only be computed alone");
//        }
//        *p_opmask = opmask;
//        *p_binfunc = binned_median;
//        return true;
//    }
//
//    if (opmask & (GH_MIN | GH_MAX)) {
//        if (opmask & ~(GH_MIN | GH_MAX)) {
//            PYRAISE(ValueError,
//                    "min/max cannot be computed with non-min/max stats");
//        }
//        *p_opmask = opmask;
//        if (opmask == GH_MIN)      *p_binfunc = binned_min;
//        else if (opmask == GH_MAX) *p_binfunc = binned_max;
//        else /* GH_MIN | GH_MAX */ *p_binfunc = binned_minmax;
//        return true;
//    }
//
//    *p_opmask = opmask;
//    if (opmask & GH_STD)
//        *p_binfunc = binned_std;
//    else
//        *p_binfunc = binned_countvals;
//    return true;
//}


//static PyObject*
//genhist(PyObject *self, PyObject *args) {
//    long nx, ny, long_mask;
//    double xmin, xmax, ymin, ymax;
//    PyObject *x_y_val_tuple;
//    if (
//        !PyArg_ParseTuple(args, "Oddddlll",
//        &x_y_val_tuple, &xmin, &xmax,
//        &ymin, &ymax, &nx, &ny, &long_mask)
//    ) {
//        return NULL; // PyArg_ParseTuple has set an exception
//    }
//
//    unsigned int opmask = 0;
//    BINFUNC *binfunc = 0;
//    if (!check_opmask(long_mask, &opmask, &binfunc)) {
//        return NULL; // check_opmask has set an exception
//    }
//
//    // NOTE: doing this silly-looking thing because PyArg_ParseTuple
//    //  will not interpret python None as 'O'
//    PyObject *x_arg = PyList_GetItem(x_y_val_tuple, 0);
//    PyObject *y_arg = PyList_GetItem(x_y_val_tuple, 1);
//    PyObject *val_arg = PyList_GetItem(x_y_val_tuple, 2);
//    if (Py_IsNone(val_arg) && opmask != GH_COUNT) {
//        PYRAISE(TypeError, "vals may only be None when only computing count");
//    }
//
//    PyArrayObject *arrays[3];
//    arrays[0] = (PyArrayObject *) PyArray_FROM_O(x_arg);
//    arrays[1] = (PyArrayObject *) PyArray_FROM_O(y_arg);
//    long n_arrs = (opmask == GH_COUNT) ? 2 : 3;
//    // TODO: are these decrefs necessary in these failure branches?
//    if (n_arrs == 3) {
//        arrays[2] = (PyArrayObject *) PyArray_FROM_O(val_arg);
//        if (check_arrs(arrays, n_arrs) == false){
//            DECREF_ALL(arrays[0], arrays[1], arrays[2]);
//            return NULL;
//        }
//    }
//    else if (check_arrs(arrays, 2) == false) {
//        DECREF_ALL(arrays[0], arrays[1]);
//        return NULL;
//    }
//    double xbounds[2] = {xmin, xmax};
//    double ybounds[2] = {ymin, ymax};
//    PyObject *binned_arr = binfunc(arrays, xbounds, ybounds, nx, ny, opmask);
//    decref_arrays(n_arrs, arrays);
//    if (binned_arr == NULL) {
//        if (PyErr_Occurred() == NULL) {
//            PyErr_SetString(PyExc_RuntimeError, "Unclassified error in binning");
//        }
//        return NULL;
//    }
//    return binned_arr;
//}
//
//PyObject*
//arrtest(PyObject *self, PyObject *args) {
//    PyObject *xarg, *yarg, *varg, *resarg;
//    if (!PyArg_ParseTuple(args, "O", &xarg)) { return NULL; }
//    PyArrayObject *xarr = (PyArrayObject*) PyArray_FROM_O(xarg);
//    return Py_None;
//}


////    PyArrayObject *yarr = (PyArrayObject*) PyArray_FROM_O(yarg);
////    PyArrayObject *varr = (PyArrayObject*) PyArray_FROM_O(varg);
////    PyArrayObject *resarr = (PyArrayObject*) PyArray_FROM_O(resarg);
//    printf("xarr is this big: %li\n\n", PyArray_SIZE(xarr));
////    printf("yarr is this big: %li\n\n", PyArray_SIZE(yarr));
////    printf("varr is this big: %li\n\n", PyArray_SIZE(varr));
////    printf("resarrr is this big: %li\n\n", PyArray_SIZE(resarr));
//    PyErr_SetString(ValueError, "Suck it, loser\n\n\n");
//    return NULL;
//}
