#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/high_voronoi/compute_high_voronoi.hpp>
#include <highvoronoi/integration/high_voronoi_integration_view.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/integration/polygon_integrator.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/parameters.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {
using Scalar = double;
using Index = std::uint32_t;
constexpr int Dim = 3;
using DBP = highvoronoi::DataBaseParams<Scalar,Index>;
using DB = highvoronoi::HVDataBase<highvoronoi::ReadWriteLock,DBP,Dim>;
using Mesh = highvoronoi::HighVoronoiMesh<Scalar,Dim,DB>;
using Point = Mesh::NodePoint;
using EdgeP = highvoronoi::EdgeBufferParams<>;
using RayP = highvoronoi::RaycastParameters<highvoronoi::InRangeRaycast,Scalar>;
using Compute = highvoronoi::ComputeHighVoronoi<
    Mesh, highvoronoi::geometry::KDSearch, RayP,
    highvoronoi::SingleThread, highvoronoi::SingleThread, DBP, EdgeP>;
using Integral = highvoronoi::VoronoiIntegral<Mesh,double,double>;

Point point(double x,double y,double z){ Point p; p<<x,y,z; return p; }
DBP dbp(){ return DBP{highvoronoi::DirectHash{65536}}; }
RayP rayp(){ RayP p; p.variance_tolerance=9e-14; return p; }

highvoronoi::IntegralDataOptions opts(){
    highvoronoi::IntegralDataOptions o;
    o.volume=o.area=o.bulk_integral=o.interface_integral=true;
    return o;
}

bool close(double a,double b,double t=5e-10){ return std::abs(a-b)<=t; }

bool equal_integrals(Mesh& mesh, Integral& a, Integral& b){
    Integral::Data::CellData ca,cb;
    for(Index pub=0; pub<mesh.visible_public_count(); ++pub){
        const Index stable=mesh.visible_public_to_internal(pub);
        if(!a.data().read_cell(stable,ca)||!b.data().read_cell(stable,cb)) return false;
        if(ca.neighbours()!=cb.neighbours() || !close(ca.volume(),cb.volume()) ||
           ca.area().size()!=cb.area().size() ||
           ca.bulk_integral().size()!=cb.bulk_integral().size() ||
           ca.interface_integral().size()!=cb.interface_integral().size()) return false;
        for(std::size_t i=0;i<ca.area().size();++i) if(!close(ca.area()[i],cb.area()[i])) return false;
        for(std::size_t i=0;i<ca.bulk_integral().size();++i) if(!close(ca.bulk_integral()[i],cb.bulk_integral()[i])) return false;
        for(std::size_t i=0;i<ca.interface_integral().size();++i) if(!close(ca.interface_integral()[i],cb.interface_integral()[i])) return false;
    }
    return true;
}
}

int main(){
    auto boundary = Mesh::BoundaryType::cuboid(
        point(1,1,1), point(0,0,0), std::vector<Index>{0,1});
    Mesh mesh(Index{Dim}, boundary, std::in_place, std::size_t{65536}, dbp());

    std::mt19937_64 rng(0x4856494e54454752ULL);
    std::uniform_real_distribution<double> u(0.04,0.96);
    std::vector<Point> pts;
    while(pts.size()<12){
        Point p=point(u(rng),u(rng),u(rng));
        bool ok=true;
        for(const auto& q:pts) ok &= (p-q).norm()>0.09;
        if(ok) pts.push_back(p);
    }
    for(const auto& p:pts) (void)mesh.append_visible_node(p);

    Compute compute(mesh, highvoronoi::geometry::KDSearch{8,1}, rayp(),
                    highvoronoi::SingleThread{}, highvoronoi::SingleThread{},
                    dbp(), EdgeP{highvoronoi::DirectHash{65536}});
    (void)compute.compute();

    Integral polygon_serial(mesh, 2, opts(), 4096, 4096);
    Integral polygon_parallel(mesh, 2, opts(), 4096, 4096);
    Integral fast_serial(mesh, 2, opts(), 4096, 4096);
    Integral fast_parallel(mesh, 2, opts(), 4096, 4096);

    const auto f = [](const Point& x) {
        return std::array<double, 2>{
            1.0,
            x[0] + 2.0 * x[1] + 3.0 * x[2]};
    };

    auto polygon_serial_algorithm =
        highvoronoi::make_polygon_algorithm(polygon_serial, f);
    auto polygon_parallel_algorithm =
        highvoronoi::make_polygon_algorithm(polygon_parallel, f);
    auto fast_serial_algorithm =
        highvoronoi::make_fast_polygon_algorithm(fast_serial, f);
    auto fast_parallel_algorithm =
        highvoronoi::make_fast_polygon_algorithm(fast_parallel, f);

    const auto polygon_serial_report = highvoronoi::integrate(
        polygon_serial,
        polygon_serial_algorithm);
    const auto polygon_parallel_report = highvoronoi::integrate(
        polygon_parallel,
        polygon_parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});
    const auto fast_serial_report = highvoronoi::integrate(
        fast_serial,
        fast_serial_algorithm);
    const auto fast_parallel_report = highvoronoi::integrate(
        fast_parallel,
        fast_parallel_algorithm,
        highvoronoi::ParallelIntegrationExecution{3});

    std::cout << "visible=" << mesh.visible_public_count()
              << " internal=" << mesh.internal_node_count()
              << " polygon_serial_updates=" << polygon_serial_report.updated_cells
              << " polygon_parallel_updates=" << polygon_parallel_report.updated_cells
              << " fast_serial_updates=" << fast_serial_report.updated_cells
              << " fast_parallel_updates=" << fast_parallel_report.updated_cells
              << '\n';

    if (mesh.internal_node_count() <= mesh.visible_public_count()) {
        std::cerr << "periodic fixture created no reference nodes\n";
        return 2;
    }

    const auto visible = mesh.visible_public_count();
    if (polygon_serial_report.updated_cells != visible ||
        polygon_parallel_report.updated_cells != visible ||
        fast_serial_report.updated_cells != visible ||
        fast_parallel_report.updated_cells != visible) {
        std::cerr << "wrong update count\n";
        return 3;
    }

    if (!equal_integrals(mesh, polygon_serial, polygon_parallel)) {
        std::cerr << "serial/parallel HighVoronoi Polygon mismatch\n";
        return 4;
    }
    if (!equal_integrals(mesh, fast_serial, fast_parallel)) {
        std::cerr << "serial/parallel HighVoronoi FastPolygon mismatch\n";
        return 5;
    }
    if (!equal_integrals(mesh, polygon_serial, fast_serial) ||
        !equal_integrals(mesh, polygon_serial, fast_parallel)) {
        std::cerr << "HighVoronoi Polygon/FastPolygon mismatch\n";
        return 6;
    }

    const auto fast_cache = fast_parallel_algorithm.parallel_cache_stats();
    if (fast_cache.entries == 0 || fast_cache.stores == 0 ||
        fast_cache.hits == 0) {
        std::cerr << "parallel HighVoronoi FastPolygon did not reuse shared facets\n";
        return 7;
    }

    std::cout << "HighVoronoi Polygon/FastPolygon serial/parallel check OK\n"
              << "FastPolygon parallel cache: hits=" << fast_cache.hits
              << " misses=" << fast_cache.misses
              << " stores=" << fast_cache.stores
              << " entries=" << fast_cache.entries << '\n';
    return 0;
}
