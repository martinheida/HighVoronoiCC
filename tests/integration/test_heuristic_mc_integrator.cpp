#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/integration/heuristic_mc_integrator.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace {
using Scalar=double; using Index=std::uint32_t; constexpr int Dimension=3;
using DatabaseParameters=highvoronoi::DataBaseParams<Scalar,Index>;
using Database=highvoronoi::HVDataBase<highvoronoi::ReadWriteLock,DatabaseParameters,Dimension>;
using Mesh=highvoronoi::VoronoiMesh<Scalar,Dimension,Database>;
using Nodes=Mesh::InternalNodes; using Point=Mesh::NodePoint; using Boundary=Mesh::BoundaryType;
using Integral=highvoronoi::VoronoiIntegral<Mesh,double,double>;
using EdgeParameters=highvoronoi::EdgeBufferParams<>;
using RayParameters=highvoronoi::RaycastParameters<highvoronoi::InRangeRaycast,Scalar>;
using Refine=highvoronoi::RefineVoronoi<Mesh,highvoronoi::geometry::KDSearch,RayParameters,
    highvoronoi::SingleThread,highvoronoi::SingleThread,DatabaseParameters,EdgeParameters>;
std::size_t checks=0,failures=0;
void check(bool ok,std::string_view msg){++checks;if(ok)std::cout<<"[OK]   "<<msg<<'\n';else{++failures;std::cerr<<"[FAIL] "<<msg<<'\n';}}
Point point(double x,double y,double z){Point p;p<<x,y,z;return p;}
Boundary boundary(){return Boundary::cuboid(point(1,1,1),point(0,0,0),std::vector<Index>{});}
Mesh make_mesh(){std::vector<Point> pts;for(double x:{.25,.75})for(double y:{.25,.75})for(double z:{.25,.75})pts.push_back(point(x,y,z));Nodes n(8);for(Index i=0;i<8;++i)n.set(i,pts[i]);auto db=std::make_shared<Database>(262144,DatabaseParameters{highvoronoi::DirectHash{262144}});return Mesh(std::move(n),boundary(),std::move(db));}
RayParameters ray_parameters(){RayParameters p;p.variance_tolerance=9e-14;return p;}
void compute(Mesh& mesh){auto tree=highvoronoi::geometry::make_search_tree(mesh,highvoronoi::geometry::KDSearch{8,1});auto ray=highvoronoi::make_raycaster(tree,ray_parameters());using C=highvoronoi::ComputeVoronoi<Mesh,decltype(ray),highvoronoi::SingleThread,highvoronoi::SingleThread,DatabaseParameters,EdgeParameters>;C c(mesh,ray,{}, {},std::nullopt,DatabaseParameters{highvoronoi::DirectHash{262144}},EdgeParameters{highvoronoi::DirectHash{262144}});c.compute();}
highvoronoi::IntegralDataOptions full(){highvoronoi::IntegralDataOptions o;o.volume=o.area=o.bulk_integral=o.interface_integral=true;return o;}
}
int main(){
    Mesh mesh=make_mesh();compute(mesh);
    Integral integral(mesh,2,full(),4096,4096);
    auto f=[](const Point& x){return std::array<double,2>{1.0,x[0]*x[0]+x[1]*x[1]+x[2]*x[2]};};
    highvoronoi::MonteCarloOptions opts;opts.interface_rays=30000;opts.seed=12345;
    auto algorithm=highvoronoi::make_heuristic_mc_algorithm(integral,f,opts);
    auto report=highvoronoi::integrate(integral,algorithm);
    check(report.updated_cells==8,"HeuristicMC computes all NEW cells");
    check(algorithm.monte_carlo_options().heuristic,"HeuristicMC forces MC geometry-only mode");
    check(algorithm.monte_carlo_options().bulk_samples_per_ray==1,"HeuristicMC disables unused bulk sampling work");
    typename Integral::Data::CellData cell;double vol=0,b0=0,b1=0;bool complete=true,iface=true;
    for(Index pub=0;pub<mesh.size();++pub){Index st=mesh.index_mapping().public_to_internal(pub);complete=complete&&integral.data().read_cell(st,cell);if(!complete)continue;vol+=cell.volume();b0+=cell.bulk_integral()[0];b1+=cell.bulk_integral()[1];for(std::size_t k=0;k<cell.area().size();++k)iface=iface&&std::abs(cell.interface_integral()[2*k]-cell.area()[k])<1e-12;}
    std::cout<<"volume_sum="<<vol<<" heuristic_constant_bulk="<<b0<<" nonlinear_bulk="<<b1<<'\n';
    check(complete,"HeuristicMC publishes complete geometry+function records");
    check(iface,"constant heuristic interface component equals finalized MC area");
    check(std::abs(vol-1.0)<0.035,"MC geometry volume converges near unit cube volume");
    check(std::abs(b0-1.0)<0.035,"heuristic constant bulk from MC areas converges near one");
    check(std::abs(b1-1.0)<0.08,"nonlinear heuristic-MC bulk is near analytic integral one");
    auto second=highvoronoi::integrate(integral,algorithm);
    check(second.updated_cells==0,"unchanged second HeuristicMC pass updates zero cells");

    std::vector<Point> added{point(.43,.57,.36),point(.62,.41,.68)};
    Refine refine(mesh,added,highvoronoi::geometry::KDSearch{8,1},ray_parameters(),{}, {},DatabaseParameters{highvoronoi::DirectHash{262144}},EdgeParameters{highvoronoi::DirectHash{262144}});(void)refine.compute();
    bool dirty=false;for(std::size_t i=0;i<integral.dirty_tracker()->size();++i)dirty=dirty||integral.dirty_tracker()->dirty(i);
    check(dirty,"mesh refine marks HeuristicMC integral dirty");
    auto refined=highvoronoi::integrate(integral,algorithm);
    check(refined.updated_cells>0,"HeuristicMC recomputes dirty/new cells after refine");

    std::cout<<"checks: "<<checks<<"\nfailures: "<<failures<<'\n';return failures==0?0:1;
}
