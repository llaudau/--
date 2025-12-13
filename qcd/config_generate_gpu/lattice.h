// lattice.h

#include <cuda_runtime.h>
#include <cstdlib>

// Define the precision of your scalar field. 
// For production, double is often preferred for stability.
using FieldValue = float; 
// or using FieldValue = double;

// lattice.h

class Lattice {
private:
    int Nx, Ny, Nz, Nt;
    size_t total_sites;
    
    // Host (CPU) pointer for the scalar field phi
    FieldValue* h_phi; 
    
    // Device (GPU) pointer for the scalar field phi
    FieldValue* d_phi; 

public:
    // Constructor and Destructor
    Lattice(int nx, int ny, int nz, int nt);
    ~Lattice();

    // 1. Indexing Utility (Important for mapping 4D to 1D)
    __host__ __device__ 
    size_t get_index(int x, int y, int z, int t) const;
    
    // 2. Memory Management (Host methods)
    void allocate_gpu_memory();
    void copy_host_to_device();
    void copy_device_to_host();
    
    // 3. Initialization (Host method)
    void initialize_field_random(float scale);
    
    // 4. Update Function (Host Wrapper that launches the Kernel)
    void update_phi_field(float mass_sq, float lambda, int n_sweeps);
    
    // 5. Accessors (Optional, but useful for debugging)
    // Getters to retrieve the field pointers (for kernel launching)
    FieldValue* get_device_field() const { return d_phi; }
    
    // ... potentially other measurement functions ...
};

// ----------------------------------------------------
// Kernel Declarations (Host code needs these)
// ----------------------------------------------------

extern "C" {
    void launch_phi_update_kernel(FieldValue* d_phi, size_t total_sites, int Nx, int Ny, int Nz, int Nt, 
                                 float mass_sq, float lambda);
}

// Implementation (likely in Lattice.cpp, or inline in .h)

__host__ __device__ 
size_t Lattice::get_index(int x, int y, int z, int t) const {
    // Standard lexicographical ordering: t*Nz*Ny*Nx + z*Ny*Nx + y*Nx + x
    return (size_t)t * Nz * Ny * Nx + 
           (size_t)z * Ny * Nx + 
           (size_t)y * Nx + 
           (size_t)x;
}