#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>
#include <cstdint>
#include <iostream>
#include <memory>

using Scalar=double; using Index=std::uint32_t; constexpr int D=2;
using DBP=highvoronoi::DataBaseParams<Scalar,Index>;
using DB=highvoronoi::HVDataBase<highvoronoi::ReadWriteLock,DBP>;
using Mesh=highvoronoi::VoronoiMesh<Scalar,D,DB>;
using EdgeP=highvoronoi::EdgeBufferParams<>;
using Point=Mesh::VertexPoint;
Point p(double x,double y){ Point r; r<<x,y; return r; }
int main(){
 auto db=std::make_shared<DB>(1024,DBP{highvoronoi::DirectHash{256}});
 Mesh::InternalNodes nodes(Index{3}); nodes.set(0,p(0,0)); nodes.set(1,p(2,0)); nodes.set(2,p(0,2));
 Mesh mesh(std::move(nodes),db);
 auto tree=highvoronoi::geometry::make_search_tree(mesh,highvoronoi::geometry::KDSearch{4,1});
 auto ray=highvoronoi::make_raycaster(tree,highvoronoi::RaycastParameters<highvoronoi::ClassicRaycast,Scalar>{});
 using RC=decltype(ray);
 using Compute=highvoronoi::ComputeVoronoi<Mesh,RC,highvoronoi::SingleThread,highvoronoi::SingleThread,DBP,EdgeP>;
 Compute compute(mesh,ray,{}, {}, std::nullopt, DBP{highvoronoi::DirectHash{64}}, EdgeP{highvoronoi::DirectHash{64}});
 compute.compute();
 const auto report=highvoronoi::verify_mesh_complete(mesh, Scalar{1e-20}, true, std::cerr);
 std::cout << "finite endpoints=" << report.unique_finite_edge_endpoints
           << " local duplicates=" << report.duplicate_local_edge_representations
           << " infinite=" << report.infinite_edges
           << " edge closure=" << report.all_edges_have_two_occurrences
           << " complete=" << report.complete() << '\n';
 if (!(report.complete() &&
      report.unique_finite_edge_endpoints == 0 &&
      report.infinite_edges == 3))
    return 1;
 for (const auto& edge : mesh.infinite_edges()) {
     if (!db->erase(edge.address, edge.sigma)) return 2;
     break;
 }
 const auto broken=highvoronoi::verify_mesh_complete(mesh, Scalar{1e-20}, false, std::cerr);
 std::cout << "after removing one infinite ray: complete=" << broken.complete() << '\n';
 return broken.complete() ? 3 : 0;
}
