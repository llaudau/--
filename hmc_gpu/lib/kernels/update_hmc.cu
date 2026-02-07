#include "lattice.cuh"


#define KERNEL_CHECK(name) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("Error launching %s: %s\n", name, cudaGetErrorString(err)); \
    } \
}

// __global__ Matrix<complex<num_type>,3> stape(LatticeView lat){
//     int site_dir=blockIdx.x*blockDim.x+threadIdx.x;
//     if (site_dir>=lat.volume) return;
//     Matrix<complex<num_type>,3> Amunu;
//     Amunu.setZero();
//     for (int nu=0;nu<4;nu++){
//         if (nu==)
//     }

// }



__global__ void kernel_refresh_mom(LatticeView lat, curandState* d_states){
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site>=lat.volume) return;
    curandState localState = d_states[site];
    for (int mu=0;mu<4; mu++){
        int link_idx = site + mu * lat.volume;
        generate_gaussian_su3_algebra(lat.d_moms[link_idx],&localState);
    }
    d_states[site]=localState;
}


__global__ void update_mom(LatticeView lat,num_type epsi){
    int site = blockIdx.x*blockDim.x+threadIdx.x; 
    if (site>=lat.volume) return;
    for( int mu=0;mu<4;mu++){
        int link_index=lat.link_idx(site,mu);
        // if (site==0 and mu==0){
        //     printf("cold start\n");
        //     printf("(0,0): Real = %f, Imag = %f\n", (double) (lat.d_links[link_index](0,0).real()), (double)(lat.d_links[link_index](0,0).imag()));
        //     printf("(1,0): Real = %f, Imag = %f\n", (double) (lat.d_links[link_index](1,0).real()), (double)(lat.d_links[link_index](1,0).imag()));
        //     printf("(2,0): Real = %f, Imag = %f\n", (double) (lat.d_links[link_index](2,0).real()), (double)(lat.d_links[link_index](2,0).imag()));
        //     printf("(1,1): Real = %f, Imag = %f\n", (double) (lat.d_links[link_index](1,1).real()), (double)(lat.d_links[link_index](1,1).imag()));
        //     printf("(2,2): Real = %f, Imag = %f\n", (double) (lat.d_links[link_index](2,2).real()), (double)(lat.d_links[link_index](2,2).imag()));
        // }
        lat.d_links[link_index]+=complex<num_type>(0.0,epsi)*lat.d_moms[link_index]*lat.d_links[link_index]+complex<num_type>(-epsi*epsi/2,0)*lat.d_moms[link_index]*lat.d_moms[link_index]*lat.d_links[link_index];
        // check reunitarize  
        reunitarize(lat.d_links[link_index]);

    }
}


__global__ void update_force(LatticeView lat, num_type epsi) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lat.volume) return;

    complex<num_type> coeff ;
    coeff=complex<num_type>(0.0,-epsi * lat.beta / 12.0);
    for (int mu = 0; mu < 4; mu++) {
        int idx_mu = lat.link_idx(site, mu);
        auto U_mu = lat.d_links[idx_mu]; // LOAD ONCE to register
        
        Matrix<complex<num_type>, 3> Staple;
        Staple.setZero();

        for (int nu = 0; nu < 4; nu++) {
            if (nu == mu) continue;
            
            // Pre-calculate neighbor indices to avoid repeated d_hopping reads
            int site_p_mu = lat.neighbor(site, mu);
            int site_p_nu = lat.neighbor(site, nu);
            int site_m_nu = lat.neighbor(site, nu + 4);
            int site_p_mu_m_nu = lat.neighbor(site_p_mu, nu + 4);

            // Upper Staple
            Staple += lat.d_links[lat.link_idx(site_p_mu, nu)] * lat.d_links[lat.link_idx(site_p_nu, mu)].dagger() * lat.d_links[lat.link_idx(site, nu)].dagger();

            // Lower Staple
            Staple += lat.d_links[lat.link_idx(site_p_mu_m_nu, nu)].dagger() * lat.d_links[lat.link_idx(site_m_nu, mu)].dagger() * lat.d_links[lat.link_idx(site_m_nu, nu)];
        }
        

        // Force = TA(U * Staple_dag)
        Matrix<complex<num_type>, 3> Temp = U_mu * Staple;
        
        // This is the Traceless Anti-Hermitian projection
        Matrix< complex <num_type>, 3> Force = (Temp - Temp.dagger())* coeff;
        complex<num_type> tr_third = Force.trace() *(num_type) (1.0/3.0);

        // 2. Subtract only from the diagonal
        Force(0,0) =Force(0,0)- tr_third;
        Force(1,1) =Force(1,1)- tr_third;
        Force(2,2) =Force(2,2)- tr_third;
        lat.d_moms[idx_mu] =lat.d_moms[idx_mu]- Force ;
        
        
        
    }

}

void GaugeField::update_1step(num_type length,int num_steps){
    num_type eps=length/num_steps;

    
    cudaMemcpy( d_links_old,d_links,  params.volume * 4 * sizeof(Matrix<complex<num_type>,3>), cudaMemcpyDeviceToDevice);
    kernel_refresh_mom<<<params.blocks,params.threads>>>(this->view(),d_rng_states);
    num_type action_i,action_f;
    num_type Hi=Hamilt(action_i);
    update_force<<<params.blocks,params.threads>>>(this->view(),eps/2.0);


    for(int i =0 ;i< num_steps; i++){
        update_mom<<<params.blocks,params.threads>>>(this->view(),eps);
        update_force<<<params.blocks,params.threads>>>(this->view(),eps);
    }
    update_mom<<<params.blocks,params.threads>>>(this->view(),eps); 
    update_force<<<params.blocks,params.threads>>>(this->view(),eps/2.0);
    num_type Hf=Hamilt(action_f);
    if (not acc_rej(Hi,Hf)){
        cudaMemcpy( d_links,d_links_old,  params.volume * 4 * sizeof(Matrix<complex<num_type>,3>), cudaMemcpyDeviceToDevice);
    }
}
