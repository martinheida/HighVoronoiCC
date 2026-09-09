#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/integration/heuristic_integrator.hpp>
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
using Scalar = double;
using Index = std::uint32_t;
constexpr int Dimension = 3;
using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock, DatabaseParameters, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;
using Integral = highvoronoi::VoronoiIntegral<Mesh, double, double>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::InRangeRaycast, Scalar>;
using Refine = highvoronoi::RefineVoronoi<
    Mesh,
    highvoronoi::geometry::KDSearch,
    RayParameters,
    highvoronoi::SingleThread,
    highvoronoi::SingleThread,
    DatabaseParameters,
    EdgeParameters>;

std::size_t checks = 0;
std::size_t failures = 0;
void check(bool ok, std::string_view message) {
    ++checks;
    if (ok) std::cout << "[OK]   " << message << '\n';
    else { ++failures; std::cerr << "[FAIL] " << message << '\n'; }
}
bool close(double a, double b, double tol=1e-9) {
    return std::abs(a-b) <= tol;
}
Point point(double x,double y,double z) { Point p; p<<x,y,z; return p; }
Boundary boundary() {
    return Boundary::cuboid(point(1,1,1),point(0,0,0),std::vector<Index>{});
}
std::vector<Point> points() {
    std::vector<Point> p;
    for(double x:{0.25,0.75}) for(double y:{0.25,0.75}) for(double z:{0.25,0.75})
        p.push_back(point(x,y,z));
    return p;
}
Mesh make_mesh() {
    auto p=points(); Nodes nodes(static_cast<Index>(p.size()));
    for(std::size_t i=0;i<p.size();++i) nodes.set(static_cast<Index>(i),p[i]);
    auto db=std::make_shared<Database>(262144,DatabaseParameters{highvoronoi::DirectHash{262144}});
    return Mesh(std::move(nodes),boundary(),std::move(db));
}
RayParameters ray_parameters() {
    RayParameters p; p.variance_tolerance=Scalar{9e-14}; return p;
}
void compute(Mesh& mesh) {
    auto tree=highvoronoi::geometry::make_search_tree(mesh,highvoronoi::geometry::KDSearch{8,1});
    auto ray=highvoronoi::make_raycaster(tree,ray_parameters());
    using Compute=highvoronoi::ComputeVoronoi<
        Mesh,decltype(ray),highvoronoi::SingleThread,highvoronoi::SingleThread,
        DatabaseParameters,EdgeParameters>;
    Compute c(mesh,ray,highvoronoi::SingleThread{},highvoronoi::SingleThread{},
              std::nullopt,DatabaseParameters{highvoronoi::DirectHash{262144}},
              EdgeParameters{highvoronoi::DirectHash{262144}});
    c.compute();
}
highvoronoi::IntegralDataOptions geometry_options() {
    highvoronoi::IntegralDataOptions o;
    o.volume=true; o.area=true; o.bulk_integral=false; o.interface_integral=false;
    return o;
}
highvoronoi::IntegralDataOptions full_options() {
    highvoronoi::IntegralDataOptions o;
    o.volume=true; o.area=true; o.bulk_integral=true; o.interface_integral=true;
    return o;
}

void test_current_source() {
    Mesh mesh=make_mesh(); compute(mesh);
    Integral source(mesh,1,geometry_options(),4096,4096);
    auto source_algorithm=highvoronoi::make_fast_polygon_algorithm(source);
    (void)highvoronoi::integrate(source,source_algorithm);

    Integral target(mesh,2,full_options(),4096,4096);
    const auto f=[](const Point& x) {
        return std::array<double,2>{1.0,1.0+2*x[0]+3*x[1]+4*x[2]};
    };
    auto heuristic=highvoronoi::make_heuristic_algorithm(
        target,source,source_algorithm,f,highvoronoi::HeuristicOptions{});
    const auto report=highvoronoi::integrate(target,heuristic);
    check(report.updated_cells==8,"heuristic computes every NEW target cell");

    typename Integral::Data::CellData s,t;
    double bulk0=0,bulk1=0;
    bool geometry_equal=true,interfaces=true;
    for(Index pub=0;pub<mesh.size();++pub) {
        Index stable=mesh.index_mapping().public_to_internal(pub);
        geometry_equal = geometry_equal && source.data().read_geometry_cell(stable,s);
        geometry_equal = geometry_equal && target.data().read_cell(stable,t);
        geometry_equal = geometry_equal && close(s.volume(),t.volume());
        geometry_equal = geometry_equal && s.neighbours()==t.neighbours();
        geometry_equal = geometry_equal && s.area().size()==t.area().size();
        if(s.area().size()==t.area().size()) {
            for(std::size_t k=0;k<s.area().size();++k)
                geometry_equal = geometry_equal && close(s.area()[k],t.area()[k]);
        }
        bulk0 += t.bulk_integral()[0];
        bulk1 += t.bulk_integral()[1];
        for(std::size_t k=0;k<t.area().size();++k)
            interfaces = interfaces && close(t.interface_integral()[2*k],t.area()[k]);
    }
    check(geometry_equal,"heuristic target copies authoritative source geometry");
    check(interfaces,"heuristic f=1 interface component equals source area");
    check(close(bulk0,1.0) && close(bulk1,5.5),
          "heuristic integrates constant/affine Cartesian test exactly");
}

void test_force_source_update_and_dirty_independence() {
    Mesh mesh=make_mesh(); compute(mesh);
    Integral source(mesh,1,geometry_options(),4096,4096);
    auto source_algorithm=highvoronoi::make_fast_polygon_algorithm(source);

    Integral target(mesh,1,full_options(),4096,4096);
    const auto one=[](const Point&){ return 1.0; };

    auto strict=highvoronoi::make_heuristic_algorithm(
        target,source,source_algorithm,one,highvoronoi::HeuristicOptions{false});
    bool rejected=false;
    try { (void)highvoronoi::integrate(target,strict); }
    catch(const std::logic_error&) { rejected=true; }
    check(rejected,"strict heuristic rejects incomplete source geometry");

    auto forced=highvoronoi::make_heuristic_algorithm(
        target,source,source_algorithm,one,highvoronoi::HeuristicOptions{true});
    const auto first=highvoronoi::integrate(target,forced);
    check(first.updated_cells==8,"forced heuristic updates missing source then target");

    std::vector<Point> added{point(0.43,0.57,0.36),point(0.62,0.41,0.68)};
    Refine refine(
        mesh,added,highvoronoi::geometry::KDSearch{8,1},ray_parameters(),
        highvoronoi::SingleThread{},highvoronoi::SingleThread{},
        DatabaseParameters{highvoronoi::DirectHash{262144}},
        EdgeParameters{highvoronoi::DirectHash{262144}});
    (void)refine.compute();

    bool source_dirty=false,target_dirty=false;
    for(std::size_t s=0;s<source.dirty_tracker()->size();++s) {
        source_dirty = source_dirty || source.dirty_tracker()->dirty(s);
        target_dirty = target_dirty || target.dirty_tracker()->dirty(s);
    }
    check(source_dirty && target_dirty,
          "mesh refine propagates dirty state independently to source and target");

    const auto updated=highvoronoi::integrate(target,forced);
    check(updated.updated_cells>0 && updated.updated_cells<=mesh.size(),
          "forced heuristic recomputes the target cells dirtied by source refresh");

    typename Integral::Data::CellData source_cell, target_cell;
    bool refreshed_values_match=true;
    for(Index pub=0;pub<mesh.size();++pub) {
        const Index stable=mesh.index_mapping().public_to_internal(pub);
        refreshed_values_match = refreshed_values_match &&
            source.data().read_geometry_cell(stable,source_cell) &&
            target.data().read_cell(stable,target_cell);
        if(!refreshed_values_match) break;
        refreshed_values_match = refreshed_values_match &&
            source_cell.neighbours()==target_cell.neighbours() &&
            source_cell.area().size()==target_cell.area().size() &&
            target_cell.interface_integral().size()==target_cell.area().size();
        if(!refreshed_values_match) break;
        refreshed_values_match = refreshed_values_match &&
            close(source_cell.volume(),target_cell.volume()) &&
            close(target_cell.bulk_integral()[0],source_cell.volume(),2e-8);
        for(std::size_t k=0;k<source_cell.area().size();++k) {
            refreshed_values_match = refreshed_values_match &&
                close(source_cell.area()[k],target_cell.area()[k]) &&
                close(target_cell.interface_integral()[k],source_cell.area()[k],2e-8);
        }
    }
    check(refreshed_values_match,
          "after forced refine update f=1 uses refreshed source volume/area on every cell/interface");

    bool source_clean=true,target_clean=true;
    for(std::size_t s=0;s<source.dirty_tracker()->size();++s) {
        source_clean = source_clean && !source.dirty_tracker()->dirty(s);
        target_clean = target_clean && !target.dirty_tracker()->dirty(s);
    }
    check(source_clean && target_clean,
          "successful source+target update clears each integral's own dirty tracker");
}

void test_self_source_rejected() {
    Mesh mesh=make_mesh(); compute(mesh);
    Integral integral(mesh,1,full_options(),4096,4096);
    auto source_algorithm=highvoronoi::make_fast_polygon_algorithm(integral);
    const auto one=[](const Point&){return 1.0;};
    bool rejected=false;
    try {
        auto invalid=highvoronoi::make_heuristic_algorithm(
            integral,integral,source_algorithm,one,highvoronoi::HeuristicOptions{true});
        (void)invalid;
    } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected,"heuristic rejects source == target immediately");
}
}

int main() {
    test_current_source();
    test_force_source_update_and_dirty_independence();
    test_self_source_rejected();
    std::cout << "checks: " << checks << "\nfailures: " << failures << '\n';
    return failures==0?0:1;
}
