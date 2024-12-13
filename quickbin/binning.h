#ifndef BINNING_H
#define BINNING_H

#include "api_helpers.h"

PyObject* binned_count(PyObject *self, PyObject *args);
PyObject* binned_countvals(PyObject *self, PyObject *args);
PyObject* binned_median(PyObject *self, PyObject *args);
PyObject* binned_min(PyObject *self, PyObject *args);
PyObject* binned_max(PyObject *self, PyObject *args);
PyObject* binned_minmax(PyObject *self, PyObject *args);
PyObject* binned_sum(PyObject *self, PyObject *args);
PyObject* binned_std(PyObject *self, PyObject *args);

#endif //BINNING_H
