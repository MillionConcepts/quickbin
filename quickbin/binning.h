#ifndef BINNING_H
#define BINNING_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <Python.h>
#include <stdbool.h>
#include <stdio.h>

#include "api_helpers.h"
#include "iterators.h"
#include "opmask.h"

PyObject* arrtest_outside(PyObject *self, PyObject *args);
PyObject* binned_count(PyObject *self, PyObject *args);
PyObject* binned_countvals(PyObject *self, PyObject *args);
PyObject* binned_median(PyObject *self, PyObject *args);
PyObject* binned_min(PyObject *self, PyObject *args);
PyObject* binned_max(PyObject *self, PyObject *args);
PyObject* binned_minmax(PyObject *self, PyObject *args);
PyObject* binned_sum(PyObject *self, PyObject *args);
PyObject* binned_std(PyObject *self, PyObject *args);

#endif //BINNING_H
