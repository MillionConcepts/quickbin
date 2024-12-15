#ifndef BINNING_H
#define BINNING_H

#include "api_helpers.h"

PyObject* binned_count(PyObject *, PyObject *const *, Py_ssize_t);
PyObject* binned_countvals(PyObject *, PyObject *const *, Py_ssize_t);
PyObject* binned_median(PyObject *, PyObject *const *, Py_ssize_t);
PyObject* binned_median_2(PyObject *, PyObject *const *, Py_ssize_t);
PyObject* binned_minmax(PyObject *, PyObject *const *, Py_ssize_t);
PyObject* binned_sum(PyObject *, PyObject *const *, Py_ssize_t);
PyObject* binned_std(PyObject *, PyObject *const *, Py_ssize_t);

#endif //BINNING_H
