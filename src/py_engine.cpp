#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"

namespace py = pybind11;

using ValuePtr = std::shared_ptr<Value>;

// Define the Python module
PYBIND11_MODULE(smolgrad, m) {
    m.doc() = "Pybind11 wrapper for the C++ Value class (autograd implementation).";

    // Bind the Value class.
    // Since Value uses std::shared_ptr (through std::enable_shared_from_this),
    // we use py::share_ptr<Value> to ensure proper lifetime management
    // between C++ shared pointers and Python references.
    py::class_<Value, ValuePtr>(m, "Value")
        // Expose the constructor.
        // pybind11 is smart enough to handle the default arguments (prev={} and op="")
        // defined in the C++ constructor: Value(float data, ...)
        .def(py::init<float>(), py::arg("data"))
        
        // Expose public methods
        .def("get_data", &Value::get_data, "Retrieve the scalar data value.")
        .def("get_grad", &Value::get_grad, "Retrieve the gradient value.")
        .def("backward", &Value::backward, "Compute the gradient using backpropagation.")

        // Bind special methods (operators) for Pythonic usage
        // Note: For binary operations returning a shared_ptr, we bind them directly
        // to the C++ operator functions.
        
        // Addition: Enables value_a + value_b
        .def("__add__", [](const ValuePtr& a, const ValuePtr& b) {
            return *a + b;
        }, py::is_operator())

        // Multiplication: Enables value_a * value_b
        .def("__mul__", [](const ValuePtr& a, const ValuePtr& b) {
            return *a * b;
        }, py::is_operator())
        
        // The Value class overloads are symmetric (Value + Value), but pybind11 
        // also needs to handle the case of `float + Value`. Since your C++
        // implementation only handles Value + Value, we need to create Value 
        // objects from floats on the fly for better Python interoperability.
        
        // r-Add (for float + Value): float gets promoted to a Value
        .def("__radd__", [](float a, const ValuePtr& b) {
            return std::make_shared<Value>(a) + b;
        }, py::is_operator())

        // r-Mul (for float * Value): float gets promoted to a Value
        .def("__rmul__", [](float a, const ValuePtr& b) {
            return std::make_shared<Value>(a) * b;
        }, py::is_operator())

        // String representation for printing (like Python's __repr__)
        .def("__repr__", [](const ValuePtr& v) {
            // This is a simple representation; you could make this more detailed
            return py::str("Value(data={0}, grad={1})").format(v->get_data(), v->get_grad());
        });
}
