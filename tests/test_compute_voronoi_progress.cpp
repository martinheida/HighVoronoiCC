#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Scalar = double;
using Index = std::uint32_t;
constexpr int Dimension = 3;
constexpr Index NodeCount = 40;
using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::ReadWriteLock, DatabaseParameters, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::VertexPoint;
using Boundary = Mesh::BoundaryType;
using RayParameters = highvoronoi::RaycastParameters<highvoronoi::ClassicRaycast, Scalar>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;

std::size_t checks=0, failures=0;
void check(bool ok, std::string_view msg) {
    ++checks;
    if (ok) std::cout << "[OK]   " << msg << '\n';
    else { ++failures; std::cerr << "[FAIL] " << msg << '\n'; }
}

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(32768, DatabaseParameters{highvoronoi::DirectHash{32768}});
}
Boundary boundary() {
    Point dimensions = Point::Constant(Scalar{8});
    Point offset = Point::Constant(Scalar{-4});
    return Boundary::cuboid(dimensions, offset, std::vector<Index>{});
}
Nodes nodes() {
    Nodes result(NodeCount);
    std::mt19937_64 rng(0x485650524f475245ULL);
    std::uniform_real_distribution<Scalar> u(-2.5,2.5);
    for (Index i=0;i<NodeCount;++i) {
        Point p;
        for (int d=0;d<Dimension;++d) p[d]=u(rng);
        result.set(i,p);
    }
    return result;
}

template<class MeshThreading>
std::string run(bool verbose, MeshThreading mesh_threading) {
    Mesh mesh(nodes(), boundary(), make_database());
    auto tree = highvoronoi::geometry::make_search_tree(mesh, highvoronoi::geometry::KDSearch{8,1});
    RayParameters rp; rp.variance_tolerance=1e-12;
    auto raycaster = highvoronoi::make_raycaster(tree,rp);
    using RayCaster=decltype(raycaster);
    using Compute=highvoronoi::ComputeVoronoi<Mesh,RayCaster,MeshThreading,highvoronoi::SingleThread,DatabaseParameters,EdgeParameters>;
    Compute compute(mesh,raycaster,std::move(mesh_threading),highvoronoi::SingleThread{},std::nullopt,
                    DatabaseParameters{highvoronoi::DirectHash{2048}},
                    EdgeParameters{highvoronoi::DirectHash{4096}});
    std::ostringstream captured;
    auto* old=std::cout.rdbuf(captured.rdbuf());
    try { compute.compute(verbose); }
    catch (...) { std::cout.rdbuf(old); throw; }
    std::cout.rdbuf(old);
    return captured.str();
}
}

int main() {
    const std::string silent = run(false, highvoronoi::SingleThread{});
    check(silent.empty(), "default/non-verbose mesh compute produces no progress output");

    const std::string serial = run(true, highvoronoi::SingleThread{});
    check(serial.find("Mesh compute") != std::string::npos,
          "verbose serial compute reports mesh progress");
    check(serial.find("100.0 %") != std::string::npos,
          "verbose serial compute reaches 100 percent");

    const std::string parallel = run(true, highvoronoi::MultiThread{3});
    check(parallel.find("Mesh compute") != std::string::npos,
          "verbose mesh-parallel compute reports mesh progress");
    check(parallel.find("100.0 %") != std::string::npos,
          "mesh-parallel progress reaches 100 percent");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures==0 ? 0 : 1;
}
