#include "lattice.h"
#include <random>

void Lattice::randomize(unsigned int seed)
{
    std::mt19937 rng(seed);  // Mersenne Twister
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < volume; ++i) {
        data[i] = dist(rng);
    }
}

Lattice::Lattice(int Ns_, int Nt_, double lambda_,double beta_)
    : Ns(Ns_), Nt(Nt_), lambda(lambda_),beta(beta_)
{
    volume = Ns * Ns * Ns * Nt;
    data = new double[volume];
    mom = new double[volume];
    blocksPerGrid = (volume + threadsPerBlock - 1) / threadsPerBlock;

    // initialize the hopping field
    hopping_field =new int[8*volume];
    calculate_hopping();


}

Lattice::~Lattice()
{
    delete[] data;
}

int Lattice::calculate_neighbor(int v, int i) {

    int x = v % Ns;
    int y = (v / Ns) % Ns;
    int z = (v / (Ns * Ns)) % Ns;
    int t = v / (Ns * Ns * Ns);

    switch (i) {
        case 0: x = (x + 1) % Ns; break;               // +x
        case 1: x = (x - 1 + Ns) % Ns; break;          // -x
        case 2: y = (y + 1) % Ns; break;               // +y
        case 3: y = (y - 1 + Ns) % Ns; break;          // -y
        case 4: z = (z + 1) % Ns; break;               // +z
        case 5: z = (z - 1 + Ns) % Ns; break;          // -z
        case 6: t = (t + 1) % Ns; break;               // +t
        case 7: t = (t - 1 + Ns) % Ns; break;          // -t
        default:
            return -1; // invalid direction
    }

    return x
         + Ns * (y
         + Ns * (z
         + Ns * t));
}

void Lattice::calculate_hopping() {
    for (int i=0;i<8;i++){
        for(int v=0; v<volume;v++){
            hopping_field[i*volume+v]=calculate_neighbor(v,i);
        }
    }
}

inline int Lattice::index(int x, int y, int z, int t) const
{
    return x
         + Ns * (y
         + Ns * (z
         + Ns * t));
}

double& Lattice::operator()(int x, int y, int z, int t)
{
    return data[index(x, y, z, t)];
}
