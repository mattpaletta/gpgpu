//
// Created by mattp on 7/24/2019.
//

// https://devblogs.nvidia.com/even-easier-introduction-cuda/

#ifndef GPUGPU_ADDITION_CPU_H
#define GPUGPU_ADDITION_CPU_H

#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(const int n, const float *x, float *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    constexpr int N = 1'000'000;

    auto *x = new float[N];
    auto *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the CPU
    add(N, x, y);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}

#endif //GPUGPU_ADDITION_CPU_H
