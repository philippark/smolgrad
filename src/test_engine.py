# Before running this script, you must successfully build and install the module 
# using: python setup.py install

import py_engine

# --- Test 1: Basic Forward Pass and Backward Pass ---
print("--- Test 1: Basic Calculation (a*b + c*b) ---")
a = py_engine.Value(2.0)
b = py_engine.Value(3.0)
c = py_engine.Value(-1.0)

# Build the expression graph: L = (a * b) + (c * b)
# The overloaded operators `__mul__` and `__add__` are called from the C++ side.
d = a * b
e = c * b
L = d + e

print(f"L (Forward Pass): {L}")
print(f"Data value of L: {L.get_data()}")

# Compute the gradients
L.backward()

# Expected results (from manual differentiation):
# dL/dL = 1.0
# dL/dd = 1.0
# dL/de = 1.0
# dL/da = dL/dd * dd/da = 1.0 * b.data = 3.0
# dL/db = dL/dd * dd/db + dL/de * de/db = 1.0 * a.data + 1.0 * c.data = 2.0 + (-1.0) = 1.0
# dL/dc = dL/de * de/dc = 1.0 * b.data = 3.0

print("\nGradients (Backward Pass):")
print(f"Gradient of a: {a.get_grad()} (Expected: 3.0)")
print(f"Gradient of b: {b.get_grad()} (Expected: 1.0)")
print(f"Gradient of c: {c.get_grad()} (Expected: 3.0)")


# --- Test 2: Interoperability with Python Floats (radd and rmul) ---
print("\n--- Test 2: Python Float Interoperability (float * Value) ---")
x = py_engine.Value(5.0)

# Python expression: z = 10.0 * x
# This triggers the __rmul__ binding in C++
z = 10.0 * x 

print(f"z (10.0 * x): {z}")
print(f"Data value of z: {z.get_data()}")

# Compute gradient for z
z.backward()

# Expected: dL/dx = 10.0
print(f"Gradient of x: {x.get_grad()} (Expected: 10.0)")
