// main.cpp

#include <iostream>
#include <vector>
#include <cmath>

// Forward declaration of the CUDA wrapper function from vector_add.cu
extern "C" void launch_vector_add(const float *h_A, const float *h_B, float *h_C, int N);

int main()
{
    // 1. Setup Parameters
    const int N = 1 << 20; // 2^20 elements (1,048,576)
    size_t bytes = N * sizeof(float);

    std::cout << "Vector size: " << N << " elements (" << bytes / (1024*1024) << " MB)\n";

    // 2. Allocate Host Vectors
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N); // Vector to store GPU result
    std::vector<float> h_ref(N); // Vector for CPU verification

    // 3. Initialize Vectors (A[i] = i; B[i] = i*2)
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
        h_ref[i] = h_A[i] + h_B[i]; // CPU reference calculation
    }

    // 4. Launch the CUDA Vector Addition
    std::cout << "Starting CUDA calculation...\n";
    launch_vector_add(h_A.data(), h_B.data(), h_C.data(), N);
    std::cout << "CUDA calculation finished.\n";

    // 5. Verification
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        if (std::fabs(h_C[i] - h_ref[i]) > 1e-5)
        {
            std::cerr << "Verification failed at index " << i 
                      << ": GPU=" << h_C[i] << ", CPU=" << h_ref[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success)
    {
        std::cout << "\n✅ Verification successful! Vector addition completed correctly on the GPU.\n";
        std::cout << "Example: C[10] = " << h_C[10] << " (10 + 20 = 30)\n";
    }
    else
    {
        std::cerr << "\n❌ Verification FAILED.\n";
    }

    return 0;
}