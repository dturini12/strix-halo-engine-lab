#include "description.hpp"
#include <iostream>

int main() {
    std::cout << strix::gemma4::export_model_description();
    return std::cout ? 0 : 1;
}
