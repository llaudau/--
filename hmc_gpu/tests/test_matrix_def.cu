#include <iostream>
#include "gauge_operation.cuh"
#include "matrix.cuh"
#include "complex.cuh"
// Tell C++ that this function is defined in the .cu file
extern "C" void launch_gpu_addmn(void* h_A, void* h_B, void* h_C);

// int main() {
//     using namespace qcdcuda;

//     // 1. Prepare Host Matrices
//     Matrix<complex<float>, 3> A, C;
//     A.setIdentity();
    
//     complex<float> b=complex<float>(1.0f, 2.0f);
//     std::cout << "Testing Split Compilation..." << std::endl;

//     // 2. Call the GPU wrapper
//     launch_gpu_addmn(&A, &b, &C);

//     // 3. Check results
//     // Expected at (0,0): Identity (1,0) + B (1,1) = (2,1)
//     std::cout << "Result at (0,0): (" << C(0,0).real() << ", " << C(0,0).imag() << ")" << std::endl;
//     std::cout << "Result at (1,0): (" << C(0,0).real() << ", " << C(1,0).imag() << ")" << std::endl;
//     std::cout << "Result at (2,0): (" << C(0,0).real() << ", " << C(2,0).imag() << ")" << std::endl;
//     std::cout << "Result at (0,1): (" << C(0,0).real() << ", " << C(0,1).imag() << ")" << std::endl;
//     std::cout << "Result at (1,1): (" << C(0,0).real() << ", " << C(1,1).imag() << ")" << std::endl;
//     std::cout << "Result at (2,1): (" << C(0,0).real() << ", " << C(2,1).imag() << ")" << std::endl;
//     std::cout << "Result at (0,2): (" << C(0,0).real() << ", " << C(0,2).imag() << ")" << std::endl;
//     std::cout << "Result at (1,2): (" << C(0,0).real() << ", " << C(1,2).imag() << ")" << std::endl;
//     std::cout << "Result at (2,2): (" << C(0,0).real() << ", " << C(2,2).imag() << ")" << std::endl;
//     return 0;
// }
using namespace qcdcuda;


// __global__ void setup_rng(curandState *state, unsigned long seed) {
//     int id = threadIdx.x + blockIdx.x * blockDim.x;
//     /* Each thread gets the same seed, a different sequence number, 
//        and no offset. */
//     curand_init(seed, id, 0, &state[id]);
// }
// __global__ void randomsu3_matrix(Matrix<complex<float>, 3> *d_A, curandState *global_state){
//     int id = threadIdx.x + blockIdx.x * blockDim.x;
//     curandState localState = global_state[id];
    
//     // 2. Generate the matrix (passing the pointer to the local state)
//     // Assuming you want to fill an array of matrices
//     generate_gaussian_su3_algebra(d_A[id], &localState);
    
//     // 3. Copy state back to global memory
//     global_state[id] = localState;
    
// }
// __global__ void compute_trace(Matrix<complex<float>, 3> *d_A, complex<float> *d_traces, int num_matrices) {
//     int id = threadIdx.x + blockIdx.x * blockDim.x;
//     if (id < num_matrices) {
//         // Compute the square of the matrix
//         Matrix<complex<float>, 3> squared_matrix = d_A[id] * conj(d_A[id]);
        
//         // Compute the trace of the squared matrix
//         complex<float> trace =squared_matrix.trace();
        
//         // Store the trace in the output array
//         d_traces[id] = trace;
//     }
// }

    
// int main() {
//     const int num_matrices = 100000; // Number of matrices
//     const int threads_per_block = 128;
//     const int num_blocks = (num_matrices + threads_per_block - 1) / threads_per_block;
//     float a=0.0;
//     float b=0.0;


//     // Allocate memory for RNG states
//     curandState* d_rng_states;
//     cudaMalloc(&d_rng_states, num_matrices * sizeof(curandState));

//     // Allocate memory for matrices
//     qcdcuda::Matrix<qcdcuda::complex<float>, 3>* d_matrices;
//     cudaMalloc(&d_matrices, num_matrices * sizeof(qcdcuda::Matrix<qcdcuda::complex<float>, 3>));

//     // Allocate memory for traces
//     qcdcuda::complex<float>* d_traces;
//     cudaMalloc(&d_traces, num_matrices * sizeof(qcdcuda::complex<float>));
//     qcdcuda::complex<float> h_traces[num_matrices];

//     // Initialize RNG states
//     setup_rng<<<num_blocks, threads_per_block>>>(d_rng_states, 1234ULL);
//     cudaDeviceSynchronize();

//     // Generate random SU(3) matrices
//     randomsu3_matrix<<<num_blocks, threads_per_block>>>(d_matrices, d_rng_states);
//     cudaDeviceSynchronize();

//     // Compute the trace of the square of each matrix
//     compute_trace<<<num_blocks, threads_per_block>>>(d_matrices, d_traces, num_matrices);
//     cudaDeviceSynchronize();

//     // Copy traces back to the host
//     cudaMemcpy(h_traces, d_traces, num_matrices * sizeof(qcdcuda::complex<float>), cudaMemcpyDeviceToHost);

//     // Print the traces
//     for (int i = 0; i < num_matrices; i++) {
//         // std::cout << "Trace of Matrix^2 [" << i << "]: (" 
//         //           << h_traces[i].real() << ", " << h_traces[i].imag() << ")" << std::endl;
//         a+=h_traces[i].real();
//         b+=h_traces[i].real()*h_traces[i].real();
//     }
//     std::cout<<a/num_matrices<<std::endl;
//     std::cout<<b/num_matrices<<std::endl;

//     // Free memory
//     cudaFree(d_rng_states);
//     cudaFree(d_matrices);
//     cudaFree(d_traces);

//     return 0;
// }
int main(){
    return 0;
}