#include "engine.h"

/**
 * @brief Value constructor
 * @param data Scalar value
 * @param prev Set of previous Value nodes (parents in the computation graph)
 * @param op Operation that produced this Value (e.g., '+', '*')
 *
 * Initializes a Value node for the computation graph, sets up backward function for autograd.
 */
Value::Value(float data, std::unordered_set<std::shared_ptr<Value>> prev, std::string op) {
    this->data = data; // The scalar value
    this->grad = 0.0;  // Gradient initialized to zero
    this->prev = std::move(prev); // Parents in the computation graph
    this->op = std::move(op); // Operation that produced this value
    // Default backward function: recursively calls backward on all parents
    this->_backward = [this] {
        for (const auto& child : this->prev) {
            child->_backward();
        }
    };
}

/**
 * @brief Retrieves the scalar value 
 *  
 * @return The scalar value (type: double) 
 */
double Value::get_data() const {
    return this->data;
}

/**
 * @brief Retrieves the gradient
 * 
 * @return The gradient value (type: double)
 */
double Value::get_grad() const {
    return this->grad;
}

/**
 * @brief Computes gradients for all nodes in the computation graph
 *
 * Performs a topological sort of the graph, then calls backward on each node in reverse order.
 * Sets the gradient of the output node to 1.0 (seed for backpropagation).
 */
void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo; // Topologically sorted nodes
    std::unordered_set<std::shared_ptr<Value>> visited; // Track visited nodes

    // Helper function to build topological order
    std::function<void(const std::shared_ptr<Value>&)> build_topo = [&](const std::shared_ptr<Value>& v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            // Recursively visit all parents
            for (const auto& child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(shared_from_this()); // Start from this node

    grad = 1.0; // Seed gradient for output node

    // Traverse nodes in reverse topological order and call their backward functions
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const auto& v = *it;
        v->_backward();
    }
}

/**
 * @brief Addition operator for Value nodes
 * @param other The other Value node to add
 * @return New Value node representing the sum
 *
 * Sets up backward function for gradient propagation through addition.
 */
std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(data + other->get_data(), out_prev, "+");

    // Backward function: propagate gradient equally to both operands
    out->_backward = [this, other, out] {
        this->grad += out->grad;
        other->grad += out->grad;
    };

    return out;
}

/**
 * @brief Multiplication operator for Value nodes
 * @param other The other Value node to multiply
 * @return New Value node representing the product
 *
 * Sets up backward function for gradient propagation through multiplication.
 */
std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(data * other->get_data(), out_prev, "*");

    // Backward function: propagate gradient using product rule
    out->_backward = [this, other, out] {
        this->grad += other->data * out->grad;
        other->grad += this->data * out->grad;
    };

    return out;
}

/**
 * @brief Global addition operator for shared_ptr<Value>
 * @param lhs Left operand
 * @param rhs Right operand
 * @return Resulting Value node
 */
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs;
}

/**
 * @brief Global multiplication operator for shared_ptr<Value>
 * @param lhs Left operand
 * @param rhs Right operand
 * @return Resulting Value node
 */
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}