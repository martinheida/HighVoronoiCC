#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/integration/integration_view.hpp>
#include <highvoronoi/integration/integrator.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {
using Scalar=double; using Index=std::uint32_t; constexpr int Dimension=2;
using Params=highvoronoi::DataBaseParams<Scalar,Index>;
using Database=highvoronoi::HVDataBase<highvoronoi::ReadWriteLock,Params,Dimension>;
using Mesh=highvoronoi::VoronoiMesh<Scalar,Dimension,Database>;
using Point=Mesh::NodePoint; using Nodes=Mesh::InternalNodes; using Integral=highvoronoi::VoronoiIntegral<Mesh,double,double>;
std::size_t checks=0,failures=0;
void check(bool ok,std::string_view msg){++checks;if(ok)std::cout<<"[OK]   "<<msg<<'\n';else{++failures;std::cerr<<"[FAIL] "<<msg<<'\n';}}
Point point(double x,double y){Point p;p<<x,y;return p;}
Mesh make_mesh(){Nodes nodes(1);nodes.set(0,point(.5,.5));auto db=std::make_shared<Database>(1024,Params{highvoronoi::DirectHash{1024}});auto boundary=Mesh::BoundaryType::cuboid(point(1,1),point(0,0),std::vector<Index>{});return Mesh(std::move(nodes),std::move(boundary),std::move(db));}
struct RecursiveAlgorithm {
    Integral* dependency=nullptr;
    RecursiveAlgorithm* dependency_algorithm=nullptr;
    void prepare_integration(Integral&) {
        if(dependency && dependency_algorithm)
            (void)highvoronoi::integrate(*dependency,*dependency_algorithm);
    }
    template<class Update> void integrate_cell(Update&) {}
};
}
int main(){
    Mesh mesh=make_mesh();
    highvoronoi::IntegralDataOptions options;options.volume=false;options.area=false;options.bulk_integral=false;options.interface_integral=false;
    Integral a(mesh,0,options,128,128), b(mesh,0,options,128,128);
    RecursiveAlgorithm aa,bb;aa.dependency=&b;aa.dependency_algorithm=&bb;bb.dependency=&a;bb.dependency_algorithm=&aa;
    bool rejected=false;try{(void)highvoronoi::integrate(a,aa);}catch(const std::logic_error&){rejected=true;}
    check(rejected,"generic integration driver rejects A -> B -> A dependency cycle");
    std::cout<<"checks: "<<checks<<"\nfailures: "<<failures<<'\n';return failures==0?0:1;
}
