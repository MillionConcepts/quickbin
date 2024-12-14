#define I_WILL_CALL_IMPORT_ARRAY
#include "binning.h"

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

    "This module contains pointy-end implementations of binning functions "
    "exposed at high level by `quickbin.base.bin2d()`.\n"
    "Caution:\n"
    "    In normal usage, these functions should only be called by the\n"
    "    handler functions in `quickbin._binning_handlers`, which should\n"
    "    themselves only be called by `quickbin.base.bin2d()`. Calling them\n"
    "    without the intermediate setup steps performed by those functions\n"
    "    may produce undesired behavior.",

    -1,       /* size of per-interpreter state of the module,
    or -1 if the module keeps state in global variables. */
    QuickbinMethods
};

PyMODINIT_FUNC PyInit_quickbin_core(void) {
    import_array();
    return PyModule_Create(&quickbin_core_mod);
}
