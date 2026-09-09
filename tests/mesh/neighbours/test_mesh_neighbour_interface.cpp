
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {
using Scalar=double;
using Index=std::uint32_t;
using Params=highvoronoi::DataBaseParams<Scalar,Index>;
using DB=highvoronoi::HVDataBase<highvoronoi::detail::ReadWriteLock, Params, 2>;
using Mesh=highvoronoi::VoronoiMesh<Scalar,2,DB>;
using Point=Mesh::NodePoint;
int failed=0;
void ck(bool b,const char*t){std::cout<<(b?"[OK] ":"[FAIL] ")<<t<<'\n';if(!b)++failed;}
Point p(double x,double y){Point q;q<<x,y;return q;}
}
int main(){
  Mesh::InternalNodes nodes(Index{3}); nodes.set(0,p(0,0)); nodes.set(1,p(1,0)); nodes.set(2,p(0,1));
  Mesh::BoundaryType boundary;
  Point base=p(-1,0), n=p(-1,0);
  boundary.add(highvoronoi::Plane<2,Scalar,Index>(base,n,highvoronoi::BoundaryCondition::Dirichlet));
  auto db=std::make_shared<DB>(128,Params{highvoronoi::DirectHash{128}});
  Mesh mesh(std::move(nodes),boundary,db);
  std::vector<Index> raw{Index{1}, Index{1}, (std::numeric_limits<Index>::max)()-Index{1}};
  mesh.store_internal_neighbours(Index{0},raw);
  std::vector<Index> out;
  ck(!mesh.neighbours(Index{0},out),"initially dirty returns false but loads data");
  ck(out.size()==3 && out[0]==1 && out[1]==1 && out[2]==mesh.size(),"duplicates and boundary map survive");
  mesh.set_dirty(Index{0},false);
  ck(mesh.neighbours(Index{0},out),"clean state returns true");
  auto addr=mesh.store_vertex(p(.5,.5),std::vector<Index>{0,1,2});
  (void)addr;
  ck(mesh.dirty(0)&&mesh.dirty(1)&&mesh.dirty(2),"vertex insert marks incident cells dirty");
  std::vector<Index> adj;
  mesh.compute_adjecents(0,adj);
  ck(adj.size()==2&&adj[0]==1&&adj[1]==2,"adjecents collects shared-vertex candidates");
  return failed?1:0;
}


