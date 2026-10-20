/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** my_torch.cpp
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string_view>

static PyObject *simple_add(PyObject *self, PyObject *args)
{
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b))
        return nullptr;

    return PyLong_FromLong(a + b);
}

static PyMethodDef my_torch_methods[] = {
    {"simple_add", simple_add, METH_VARARGS, "Add two integers."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

static constexpr std::string_view LIB_NAME = "my_torch";
static constexpr std::string_view LIB_DOC = "A simple library like pytorch.";

PyABIInfo_VAR(abi_info);

static PySlot my_torch_slots[] = {
    PySlot_STATIC_DATA(Py_mod_abi, &abi_info),
    PySlot_STATIC_DATA(Py_mod_name, const_cast<char *>(LIB_NAME.data())),
    PySlot_STATIC_DATA(Py_mod_doc, const_cast<char *>(LIB_DOC.data())),
    PySlot_STATIC_DATA(Py_mod_methods, my_torch_methods),
    PySlot_END
};

PyMODEXPORT_FUNC PyModExport_my_torch(void) {
    return my_torch_slots;
}

static PyModuleDef my_torch_module = {
    PyModuleDef_HEAD_INIT,
    "my_torch",
    LIB_DOC.data(),
    -1,
    my_torch_methods
};

PyMODINIT_FUNC PyInit_my_torch(void) {
    return PyModule_Create(&my_torch_module);
}