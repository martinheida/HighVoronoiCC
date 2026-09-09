#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/integration/voronoi_integral.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Scalar = double;
using Index = std::uint32_t;
constexpr int Dimension = 4;
using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::ReadWriteLock, DatabaseParameters, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;
using Integral = highvoronoi::VoronoiIntegral<Mesh, double, double>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using RayParameters = highvoronoi::RaycastParameters<highvoronoi::InRangeRaycast, Scalar>;
using Refine = highvoronoi::RefineVoronoi<
    Mesh, highvoronoi::geometry::KDSearch, RayParameters,
    highvoronoi::SingleThread, highvoronoi::SingleThread,
    DatabaseParameters, EdgeParameters>;

constexpr std::array<double,4> Exact{{1.0, 1.0/3.0, 1.0/4.0, 7.0/12.0}};

std::size_t checks=0, failures=0;
void check(bool ok, std::string_view msg) {
    ++checks;
    if (ok) std::cout << "[OK]   " << msg << '\n';
    else { ++failures; std::cerr << "[FAIL] " << msg << '\n'; }
}

Point point(double a,double b,double c,double d) { Point p; p<<a,b,c,d; return p; }
Boundary unit_boundary() {
    return Boundary::cuboid(point(1,1,1,1),point(0,0,0,0),std::vector<Index>{});
}
RayParameters ray_parameters() { RayParameters p; p.variance_tolerance=Scalar{9e-14}; return p; }
DatabaseParameters database_parameters() { return DatabaseParameters{highvoronoi::DirectHash{1048576}}; }
EdgeParameters edge_parameters() { return EdgeParameters{highvoronoi::DirectHash{1048576}}; }

std::unique_ptr<Mesh> make_mesh(const std::vector<Point>& pts) {
    Nodes nodes(static_cast<Index>(pts.size()));
    for (Index i=0;i<static_cast<Index>(pts.size());++i) nodes.set(i,pts[static_cast<std::size_t>(i)]);
    auto db=std::make_shared<Database>(1048576,database_parameters());
    return std::make_unique<Mesh>(std::move(nodes),unit_boundary(),std::move(db));
}
void compute_full(Mesh& mesh) {
    auto tree=highvoronoi::geometry::make_search_tree(mesh,highvoronoi::geometry::KDSearch{8,1});
    auto raycaster=highvoronoi::make_raycaster(tree,ray_parameters());
    using RayCaster=decltype(raycaster);
    using Compute=highvoronoi::ComputeVoronoi<Mesh,RayCaster,highvoronoi::SingleThread,highvoronoi::SingleThread,DatabaseParameters,EdgeParameters>;
    Compute compute(mesh,raycaster,highvoronoi::SingleThread{},highvoronoi::SingleThread{},std::nullopt,database_parameters(),edge_parameters());
    compute.compute();
}

std::array<double,4> function(const Point& x) {
    return {{1.0, x[0]*x[0], x[1]*x[1]*x[1], x[2]*x[2] + x[0]*x[0]*x[0]}};
}

struct Result {
    std::array<double,4> integral{{0,0,0,0}};
    std::array<double,4> error{{0,0,0,0}};
    double score=0.0;
    std::size_t updated=0;
};

template <class Algorithm>
Result integrate_and_measure(Mesh& mesh, Integral& integral, Algorithm& algorithm) {
    Result r;
    const auto report=highvoronoi::integrate(integral,algorithm);
    r.updated=report.updated_cells;
    typename Integral::Data::CellData cell;
    for(Index pub=0; pub<mesh.size(); ++pub) {
        const Index stable=mesh.index_mapping().public_to_internal(pub);
        if(!integral.data().read_cell(stable,cell)) throw std::runtime_error("incomplete integral cell");
        if(cell.bulk_integral().size()!=4) throw std::runtime_error("wrong bulk component count");
        for(std::size_t k=0;k<4;++k) r.integral[k]+=cell.bulk_integral()[k];
    }
    for(std::size_t k=0;k<4;++k) r.error[k]=std::abs(r.integral[k]-Exact[k]);
    for(std::size_t k=1;k<4;++k) r.score += r.error[k]/Exact[k];
    return r;
}

void print_result(std::string_view family, std::size_t n, std::size_t count, const Result& r) {
    std::cout << std::setprecision(15)
              << family << " n=" << n << " nodes=" << count << " updated=" << r.updated
              << "\n  I = [" << r.integral[0] << ", " << r.integral[1] << ", " << r.integral[2] << ", " << r.integral[3] << "]"
              << "\n  E = [" << r.error[0] << ", " << r.error[1] << ", " << r.error[2] << ", " << r.error[3] << "]"
              << "\n  relative nonlinear L1 score = " << r.score << '\n';
}

struct SeriesEntry {
    std::size_t nodes = 0;
    Result result;
};

double log_log_slope(
    const std::vector<SeriesEntry>& series,
    std::size_t component) {
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    const double floor = 1.0e-30;
    const double count = static_cast<double>(series.size());
    for (const auto& entry : series) {
        const double x = std::log(static_cast<double>(entry.nodes));
        const double y = std::log(std::max(entry.result.error[component], floor));
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    return (count * sxy - sx * sy) / (count * sxx - sx * sx);
}

double score_log_log_slope(const std::vector<SeriesEntry>& series) {
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    const double floor = 1.0e-30;
    const double count = static_cast<double>(series.size());
    for (const auto& entry : series) {
        const double x = std::log(static_cast<double>(entry.nodes));
        const double y = std::log(std::max(entry.result.score, floor));
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    return (count * sxy - sx * sy) / (count * sxx - sx * sx);
}

std::vector<Point> cartesian_points(std::size_t n) {
    std::vector<Point> pts; pts.reserve(n*n*n*n);
    for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j)
    for(std::size_t k=0;k<n;++k) for(std::size_t l=0;l<n;++l) {
        pts.push_back(point((i+0.5)/n,(j+0.5)/n,(k+0.5)/n,(l+0.5)/n));
    }
    return pts;
}

std::vector<Point> random_points(std::size_t count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0,1.0);
    std::vector<Point> pts; pts.reserve(count);
    while(pts.size()<count) pts.push_back(point(u(rng),u(rng),u(rng),u(rng)));
    return pts;
}

highvoronoi::IntegralDataOptions integral_options() {
    highvoronoi::IntegralDataOptions o;
    o.volume=true; o.area=true; o.bulk_integral=true; o.interface_integral=true;
    return o;
}

void cartesian_convergence() {
    std::cout << "\n=== 4D Cartesian convergence ===\n";
    std::vector<SeriesEntry> series;
    series.reserve(5);

    for (std::size_t n = 2; n <= 6; ++n) {
        auto pts = cartesian_points(n);
        auto mesh = make_mesh(pts);
        compute_full(*mesh);
        Integral integral(*mesh, 4, integral_options(), 16384, 65536);
        auto algorithm = highvoronoi::make_fast_polygon_algorithm(integral, function);
        const Result result = integrate_and_measure(*mesh, integral, algorithm);
        print_result("cartesian", n, pts.size(), result);
        check(result.error[0] < 1e-7,
              "Cartesian constant component integrates to unit hypercube volume");
        series.push_back(SeriesEntry{pts.size(), result});
    }

    bool componentwise_monotone = true;
    bool score_monotone = true;
    for (std::size_t level = 1; level < series.size(); ++level) {
        for (std::size_t component = 1; component < 4; ++component) {
            componentwise_monotone = componentwise_monotone &&
                series[level].result.error[component] <
                series[level - 1].result.error[component];
        }
        score_monotone = score_monotone &&
            series[level].result.score < series[level - 1].result.score;
    }
    check(componentwise_monotone,
          "Cartesian nonlinear errors decrease at every refinement level");
    check(score_monotone,
          "Cartesian combined nonlinear error decreases at every refinement level");

    std::cout << "  Cartesian log-log slopes:";
    for (std::size_t component = 1; component < 4; ++component) {
        const double slope = log_log_slope(series, component);
        std::cout << " c" << component << '=' << slope;
        check(slope < 0.0,
              "Cartesian nonlinear component has negative convergence slope");
    }
    const double score_slope = score_log_log_slope(series);
    std::cout << " score=" << score_slope << '\n';
    check(score_slope < 0.0,
          "Cartesian combined nonlinear error has negative convergence slope");
}

void random_refine_convergence() {
    std::cout << "\n=== 4D nested random/refine convergence ===\n";
    constexpr std::array<std::size_t, 5> target{{16, 81, 256, 625, 1296}};

    // One deterministic random pool. Every later stage keeps every earlier
    // generator and appends only the next slice through RefineVoronoi.
    const auto pool = random_points(target.back(), 0x4856504f4c59434full);
    std::vector<Point> initial(
        pool.begin(),
        pool.begin() + static_cast<std::ptrdiff_t>(target.front()));

    auto mesh = make_mesh(initial);
    compute_full(*mesh);
    Integral integral(*mesh, 4, integral_options(), 16384, 65536);
    auto algorithm = highvoronoi::make_fast_polygon_algorithm(integral, function);

    std::vector<SeriesEntry> series;
    series.reserve(target.size());

    Result first = integrate_and_measure(*mesh, integral, algorithm);
    print_result("random-refine", 2, target[0], first);
    check(first.error[0] < 1e-7,
          "Random constant component integrates to unit hypercube volume initially");
    series.push_back(SeriesEntry{target[0], first});

    for (std::size_t level = 1; level < target.size(); ++level) {
        std::vector<Point> added(
            pool.begin() + static_cast<std::ptrdiff_t>(target[level - 1]),
            pool.begin() + static_cast<std::ptrdiff_t>(target[level]));

        Refine refine(
            *mesh,
            added,
            highvoronoi::geometry::KDSearch{8, 1},
            ray_parameters(),
            highvoronoi::SingleThread{},
            highvoronoi::SingleThread{},
            database_parameters(),
            edge_parameters());
        (void)refine.compute();

        const Result result = integrate_and_measure(*mesh, integral, algorithm);
        print_result("random-refine", level + 2, target[level], result);
        check(result.error[0] < 1e-7,
              "Random constant component integrates to unit hypercube volume after refine");
        series.push_back(SeriesEntry{target[level], result});
    }

    // For a random nested sequence we deliberately do NOT require the absolute
    // global error to decrease at every single level. Local approximation gets
    // finer under RefineVoronoi, but signed global errors can temporarily lose
    // a favourable cancellation. Convergence means a downward trend and a
    // genuinely smaller final error, not a lucky monotone cancellation pattern.
    for (std::size_t component = 1; component < 4; ++component) {
        const double slope = log_log_slope(series, component);
        std::cout << "  random component " << component
                  << " log-log slope=" << slope << '\n';
        check(series.back().result.error[component] <
                  series.front().result.error[component],
              "Nested random final nonlinear component error is below initial error");
        check(slope < 0.0,
              "Nested random nonlinear component has negative convergence slope");
    }

    const double score_slope = score_log_log_slope(series);
    std::size_t improving_steps = 0;
    for (std::size_t level = 1; level < series.size(); ++level) {
        improving_steps +=
            series[level].result.score < series[level - 1].result.score ? 1U : 0U;
    }
    std::cout << "  random score log-log slope=" << score_slope
              << " improving steps=" << improving_steps
              << '/' << (series.size() - 1) << '\n';

    check(series.back().result.score < series.front().result.score,
          "Nested random final combined nonlinear error is below initial error");
    check(score_slope < 0.0,
          "Nested random combined nonlinear error has negative convergence slope");
}
}

int main() {
    std::cout.setf(std::ios::unitbuf);
    cartesian_convergence();
    random_refine_convergence();
    std::cout << "\nchecks="<<checks<<" failures="<<failures<<'\n';
    return failures==0?0:1;
}
