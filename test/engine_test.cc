#include "../src/engine.h"
#include <iostream>

int main() {
    Value a(2);
    Value b(3);

    Value x = a+b;
    std::cout << x.data << "\n";

    Value y = a*b;
    std::cout << y.data << "\n";
}