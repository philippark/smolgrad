#pragma once

#include <vector>
#include <memory>

class Value {
public:
    double data;

    Value(double _data);

    Value operator+(const Value& other) const;

    Value operator*(const Value& other) const;
};