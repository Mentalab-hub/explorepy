#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *int24to32_convert(PyObject *self, PyObject *args) {
  Py_buffer view;
  const char *byteorder;

  if (!PyArg_ParseTuple(args, "s*s", &view, &byteorder)) {
    return NULL;
  }

  if (view.len % 3 != 0) {
    PyErr_SetString(PyExc_ValueError, "Input length must be multiple of 3");
    PyBuffer_Release(&view);
    return NULL;
  }

  npy_intp dims[] = {view.len / 3};
  PyObject *result_array = PyArray_SimpleNew(1, dims, NPY_INT32);
  if (result_array == NULL) {
    PyBuffer_Release(&view);
    return NULL;
  }

  int32_t *result_data = (int32_t *)PyArray_DATA((PyArrayObject *)result_array);
  const uint8_t *input_data = (const uint8_t *)view.buf;

  if (strcmp(byteorder, "little") == 0) {
    for (Py_ssize_t i = 0; i < view.len; i += 3) {
      int32_t value =
          input_data[i] | (input_data[i + 1] << 8) | (input_data[i + 2] << 16);
      if (value & 0x800000) {
        value |= 0xFF000000;
      }
      result_data[i / 3] = value;
    }
  } else {
    for (Py_ssize_t i = 0; i < view.len; i += 3) {
      int32_t value =
          (input_data[i] << 16) | (input_data[i + 1] << 8) | input_data[i + 2];
      if (value & 0x800000) {
        value |= 0xFF000000;
      }
      result_data[i / 3] = value;
    }
  }

  PyBuffer_Release(&view);
  return result_array;
}

static PyMethodDef Int24Methods[] = {{"convert", int24to32_convert,
                                      METH_VARARGS,
                                      "Convert int24 data to int32 array."},
                                     {NULL, NULL, 0, NULL}};

static struct PyModuleDef int24module = {PyModuleDef_HEAD_INIT, "int24to32",
                                         NULL, -1, Int24Methods};

PyMODINIT_FUNC PyInit_int24to32(void) {
  import_array();
  return PyModule_Create(&int24module);
}
