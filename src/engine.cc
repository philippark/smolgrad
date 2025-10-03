#include "engine.h"

Value::Value(double _data) {
    data = _data;
}

Value Value::operator+(const Value& other) const {
    return Value(this->data + other.data);   
}

Value Value::operator*(const Value& other) const {
    return Value(this->data * other.data);
}
