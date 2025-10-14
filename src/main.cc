#include "engine.h"

#include <iostream>
#include <memory>

int main() {
    auto v1 = std::make_shared<Value>(2);
    auto v2 = std::make_shared<Value>(3);

    auto v3 = v1 * v2;

    std::cout << v1->get_data() << " " << v1->get_grad() <<"\n";
    std::cout << v2->get_data() << " " << v2->get_grad() << "\n";
    std::cout << v3->get_data() << " " << v3->get_grad() << "\n";

    v3->backward();
    std::cout << "final:\n";
    std::cout << v1->get_data() << " " << v1->get_grad() << "\n";
    std::cout << v2->get_data() << " " << v2->get_grad() << "\n";
    std::cout << v3->get_data() << " " << v3->get_grad() << "\n";

    return 0;
}