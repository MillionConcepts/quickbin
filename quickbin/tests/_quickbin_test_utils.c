#define PY_SSIZE_T_CLEAN
#include <stdbool.h>
#include <Python.h>

PyObject*
refleak(PyObject *self, PyObject *const *args, Py_ssize_t n_args) {
    bool bad = false;
    if (n_args == 0) bad = true;
    else if((PyObject*) args[0] == NULL) bad = true;
    if (bad == true) {
        PyErr_Format(
            PyExc_TypeError,
            "Someone has called a function that shouldn't be called in a way "
            "they shouldn't."
        );
        return NULL;
    }
    Py_INCREF(args[0]);
    Py_RETURN_NONE;
}


static PyMethodDef
    QuickbinTestMethods[] = {
    {
        "refleak",
        (PyCFunction) refleak,
        METH_FASTCALL,
        "This function exists to leak a reference. Do not call it."
    },
    {NULL, NULL, 0, NULL}
};

static struct
    PyModuleDef quickbin_test_utils = {
    PyModuleDef_HEAD_INIT,
    "_quickbin_test_utils",   /* name of module */

    "Test utilities for `quickbin` that require interaction with the Python "
    "C-API.\n\n"
    "Caution:\n"
    "    Functions in this module exist to do messed-up things. You\n"
    "probably should not call them, and if you do, it's on you.",
    -1,       /* size of per-interpreter state of the module,
    or -1 if the module keeps state in global variables. */
    QuickbinTestMethods
};

PyMODINIT_FUNC PyInit__quickbin_test_utils(void) {
    return PyModule_Create(&quickbin_test_utils);
}
