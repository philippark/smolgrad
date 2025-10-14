#ifndef ENGINE_H
#define ENGINE_H

#include <memory>
#include <unordered_set>
#include <functional>
#include <string>

class Value : public std::enable_shared_from_this<Value> {
private:
    double data;
    double grad;
    std::function<void()> _backward;
    std::unordered_set<std::shared_ptr<Value>> prev;
    std::string op;

public:
    Value(float data, std::unordered_set<std::shared_ptr<Value>> prev = {}, std::string op = "");

    double get_data() const;
    double get_grad() const;

    void backward();

    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);  
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);

#endif