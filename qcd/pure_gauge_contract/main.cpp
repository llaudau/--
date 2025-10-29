#include "lattice.h"
#include "read.h"

int main(){

    LatticeData data0=read_GaugeFieldData("/Users/wangkehe/Git_repository/qcd/config_generate/pure_gauge_data/field0.bin");    
    Lattice *my_lattice=new Lattice(data0.Lt,data0.Lx,2.0);
    my_lattice->read_from_ext(data0.field);
    ComplexD shit= my_lattice->Wilsonloop(1,1,0,0);
    std::cout<<shit<<std::endl;
    return 0;
}