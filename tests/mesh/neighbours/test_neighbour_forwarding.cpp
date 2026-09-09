#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/mesh/high_voronoi_compute_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using Scalar=double;
using Index=std::uint32_t;
using Params=highvoronoi::DataBaseParams<Scalar,Index>;
using DB=highvoronoi::HVDataBase<highvoronoi::detail::ReadWriteLock, Params, 2>;
using Mesh=highvoronoi::HighVoronoiMesh<Scalar,2,DB>;
using Point=Mesh::NodePoint;
Point p(double x,double y){Point q;q<<x,y;return q;}
int fails=0; void ck(bool b,const char*t){std::cout<<(b?"[OK] ":"[FAIL] ")<<t<<'\n';if(!b)++fails;}

template<class T, class=void> struct has_neighbour_address : std::false_type {};
template<class T> struct has_neighbour_address<T,std::void_t<decltype(std::declval<T&>().neighbour_address(Index{}))>> : std::true_type {};
template<class T, class=void> struct has_dirty_tracker_request : std::false_type {};
template<class T> struct has_dirty_tracker_request<T,std::void_t<decltype(std::declval<T&>().request_neighbour_dirty_tracker())>> : std::true_type {};
}

int main(){
  Mesh::BoundaryType boundary;
  Mesh mesh(Index{2}, boundary, std::in_place, std::size_t{128}, Params{highvoronoi::DirectHash{128}});
  (void)mesh.append_visible_node(p(0,0));
  (void)mesh.append_visible_node(p(1,0));
  (void)mesh.append_visible_node(p(0,1));
  mesh.store_internal_neighbours(Index{0}, std::vector<Index>{1,1,2});

  highvoronoi::HighVoronoiDataMeshView<Mesh> data(mesh);
  std::vector<Index> b;
  ck(!data.neighbours(Index{0},b) && b==std::vector<Index>({1,1,2}),
     "data view forwards neighbour geometry and dirty state");
  data.set_dirty(Index{0},false);
  ck(!mesh.dirty(Index{0}),"view dirty update reaches persistent owner");

  highvoronoi::HighVoronoiComputeRoundState<Mesh, highvoronoi::detail::BitVector> state(mesh.internal_node_count(), Index{0}, mesh.internal_node_count());
  highvoronoi::HighVoronoiComputeMesh<Mesh, highvoronoi::detail::BitVector> compute(mesh,state);
  ck(compute.neighbours(Index{0},b) && b==std::vector<Index>({1,1,2}),
     "compute view forwards neighbour geometry");
  compute.set_dirty(Index{0},true);
  ck(mesh.dirty(Index{0}),"compute view forwards dirty mutation");

  static_assert(!has_neighbour_address<decltype(data)>::value,
                "data views must not expose persistent neighbour addresses");
  static_assert(!has_neighbour_address<decltype(compute)>::value,
                "compute views must not expose persistent neighbour addresses");
  static_assert(!has_dirty_tracker_request<decltype(data)>::value,
                "data views must not own/forward consumer tracker registration");
  static_assert(!has_dirty_tracker_request<decltype(compute)>::value,
                "compute views must not own/forward consumer tracker registration");

  const auto address = mesh.neighbour_address(Index{0});
  std::vector<Index> historical;
  ck(address != 0 && mesh.read_internal_neighbours_at(address,historical) &&
         historical==std::vector<Index>({1,1,2}),
     "persistent owner exposes retained logical neighbour records");

  auto tracker = mesh.request_neighbour_dirty_tracker();
  mesh.propagate_neighbour_dirty();
  ck(tracker->dirty(0),
     "persistent owner registers and updates external dirty trackers");

  ck(mesh.neighbour_version()==0 && mesh.advance_neighbour_version()==1 &&
         mesh.neighbour_version()==1,
     "persistent owner alone owns the global neighbour version");
  return fails?1:0;
}
