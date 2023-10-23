#include <iostream>
#include <chrono>

int main() {
    auto resolution = std::chrono::high_resolution_clock::period::den /
                      std::chrono::high_resolution_clock::period::num;

    std::cout << "Resolution of high_resolution_clock: " << resolution << " ticks per second" << std::endl;
}
