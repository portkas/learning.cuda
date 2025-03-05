#include <iostream>
#include <cmath>
#include <memory>
#include <chrono>

void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;
    std::unique_ptr<float[]> x(new float[N]);
    std::unique_ptr<float[]> y(new float[N]);

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    add(N, x.get(), y.get());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0));
    }
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    return 0;
}