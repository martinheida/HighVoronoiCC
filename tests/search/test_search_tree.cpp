#include <highvoronoi/search/search_tree_factory_crtp.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Scalar = double;
using MeshIndex = std::uint16_t;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    ++performed_checks;
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

bool close(Scalar left, Scalar right) {
    return std::abs(left - right) <= 1.0e-12;
}

// Deliberately not std::array: the mesh point and backend point are distinct.
struct MeshPoint2D {
    Scalar coordinates[2]{};

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return 2;
    }

    [[nodiscard]] Scalar& operator[](std::size_t coordinate) noexcept {
        return coordinates[coordinate];
    }

    [[nodiscard]] const Scalar& operator[](
        std::size_t coordinate) const noexcept {
        return coordinates[coordinate];
    }
};

MeshPoint2D point(Scalar x, Scalar y) {
    return MeshPoint2D{{x, y}};
}

class StoredNodes2D {
public:
    explicit StoredNodes2D(std::vector<MeshPoint2D> points)
        : points_(std::move(points)) {}

    [[nodiscard]] MeshIndex size() const noexcept {
        return static_cast<MeshIndex>(points_.size());
    }

    [[nodiscard]] const Scalar* stable_node_data(
        MeshIndex index) const {
        return points_.at(index).coordinates;
    }

    [[nodiscard]] Scalar get_data(
        MeshIndex index,
        MeshIndex coordinate) const {
        return points_.at(index)[coordinate];
    }

    void copy_node(MeshIndex index, Scalar* target) const {
        target[0] = points_.at(index)[0];
        target[1] = points_.at(index)[1];
    }

private:
    std::vector<MeshPoint2D> points_;
};

class ExtendedNodes2D {
public:
    ExtendedNodes2D(
        std::vector<MeshPoint2D> points,
        std::vector<MeshIndex> active_boundary_indices)
        : points_(std::move(points)),
          active_boundary_indices_(
              std::move(active_boundary_indices)) {}

    void copy_node(MeshIndex index, Scalar* target) const {
        target[0] = points_.at(index)[0];
        target[1] = points_.at(index)[1];
    }

    [[nodiscard]] MeshIndex active_boundary_size() const noexcept {
        return static_cast<MeshIndex>(active_boundary_indices_.size());
    }

    [[nodiscard]] MeshIndex active_boundary_index(
        MeshIndex position) const {
        return active_boundary_indices_.at(position);
    }

private:
    std::vector<MeshPoint2D> points_;
    std::vector<MeshIndex> active_boundary_indices_;
};

class FixedMesh {
public:
    using NodeScalar = Scalar;
    using Index = MeshIndex;
    using NodePoint = MeshPoint2D;

    static constexpr int DimensionAtCompileTime = 2;

    FixedMesh(
        StoredNodes2D nodes,
        ExtendedNodes2D extended_nodes)
        : nodes_(std::move(nodes)),
          extended_nodes_(std::move(extended_nodes)) {}

    [[nodiscard]] MeshIndex size() const noexcept {
        return nodes_.size();
    }

    [[nodiscard]] MeshIndex dimension() const noexcept {
        return MeshIndex{2};
    }

    [[nodiscard]] const StoredNodes2D& nodes() const noexcept {
        return nodes_;
    }

    [[nodiscard]] const ExtendedNodes2D&
    extended_nodes() const noexcept {
        return extended_nodes_;
    }

private:
    StoredNodes2D nodes_;
    ExtendedNodes2D extended_nodes_;
};

FixedMesh make_fixed_mesh() {
    std::vector<MeshPoint2D> ordinary{
        point(0.0, 0.0),
        point(2.0, 0.0),
        point(0.0, 2.0)};

    std::vector<MeshPoint2D> extended = ordinary;
    extended.push_back(point(0.4, 0.4));  // active boundary index 3
    extended.push_back(point(0.41, 0.4)); // closer, but inactive index 4

    return FixedMesh(
        StoredNodes2D(std::move(ordinary)),
        ExtendedNodes2D(
            std::move(extended),
            std::vector<MeshIndex>{MeshIndex{3}}));
}

// ============================================================================
// Dynamic-dimension mesh
// ============================================================================

class DynamicNodes {
public:
    explicit DynamicNodes(std::vector<std::vector<Scalar>> points)
        : points_(std::move(points)) {}

    [[nodiscard]] MeshIndex size() const noexcept {
        return static_cast<MeshIndex>(points_.size());
    }

    [[nodiscard]] const Scalar* stable_node_data(
        MeshIndex index) const {
        return points_.at(index).data();
    }

    [[nodiscard]] Scalar get_data(
        MeshIndex index,
        MeshIndex coordinate) const {
        return points_.at(index).at(coordinate);
    }

    void copy_node(MeshIndex index, Scalar* target) const {
        const auto& source = points_.at(index);
        std::copy(source.begin(), source.end(), target);
    }

private:
    std::vector<std::vector<Scalar>> points_;
};

class DynamicExtendedNodes {
public:
    DynamicExtendedNodes(
        std::vector<std::vector<Scalar>> points,
        std::vector<MeshIndex> active)
        : points_(std::move(points)), active_(std::move(active)) {}

    void copy_node(MeshIndex index, Scalar* target) const {
        const auto& source = points_.at(index);
        std::copy(source.begin(), source.end(), target);
    }

    [[nodiscard]] MeshIndex active_boundary_size() const noexcept {
        return static_cast<MeshIndex>(active_.size());
    }

    [[nodiscard]] MeshIndex active_boundary_index(
        MeshIndex position) const {
        return active_.at(position);
    }

private:
    std::vector<std::vector<Scalar>> points_;
    std::vector<MeshIndex> active_;
};

class DynamicMesh {
public:
    using NodeScalar = Scalar;
    using Index = MeshIndex;
    using NodePoint = std::vector<Scalar>;

    static constexpr int DimensionAtCompileTime = -1;

    DynamicMesh(
        DynamicNodes nodes,
        DynamicExtendedNodes extended_nodes,
        MeshIndex dimension)
        : nodes_(std::move(nodes)),
          extended_nodes_(std::move(extended_nodes)),
          dimension_(dimension) {}

    [[nodiscard]] MeshIndex size() const noexcept {
        return nodes_.size();
    }

    [[nodiscard]] MeshIndex dimension() const noexcept {
        return dimension_;
    }

    [[nodiscard]] const DynamicNodes& nodes() const noexcept {
        return nodes_;
    }

    [[nodiscard]] const DynamicExtendedNodes&
    extended_nodes() const noexcept {
        return extended_nodes_;
    }

private:
    DynamicNodes nodes_;
    DynamicExtendedNodes extended_nodes_;
    MeshIndex dimension_;
};

DynamicMesh make_dynamic_mesh() {
    std::vector<std::vector<Scalar>> ordinary{
        {0.0, 0.0, 0.0},
        {3.0, 0.0, 0.0}};
    std::vector<std::vector<Scalar>> extended = ordinary;
    extended.push_back({0.5, 0.0, 0.0});

    return DynamicMesh(
        DynamicNodes(std::move(ordinary)),
        DynamicExtendedNodes(
            std::move(extended),
            std::vector<MeshIndex>{MeshIndex{2}}),
        MeshIndex{3});
}

struct CustomLinearSearch {};

} // namespace

namespace highvoronoi::geometry {

template <class Mesh>
struct SearchTreeFactory<::CustomLinearSearch, Mesh, void> {
    using type = BruteForceSearchTree<Mesh>;

    [[nodiscard]] static type make(
        Mesh& mesh,
        const ::CustomLinearSearch&) {
        return type(mesh);
    }
};

} // namespace highvoronoi::geometry

namespace {

template <class Tree>
void test_fixed_backend(Tree& tree, std::string_view name) {
    std::cout << "\n============================================================\n";
    std::cout << name << '\n';
    std::cout << "============================================================\n";

    auto data = tree.make_backend_data(point(0.41, 0.4));
    data.reserve(8);

    using Data = std::decay_t<decltype(data)>;
    using BackendPoint = typename Data::Point;
    using PublicIndex = typename Data::List::MeshIndex;

    check((!std::is_same_v<BackendPoint, FixedMesh::NodePoint>),
          "backend point type is independent of Mesh::NodePoint");
    check((std::is_same_v<PublicIndex, MeshIndex>),
          "public result list uses Mesh::Index");

    data.point[0] = 0.41;
    data.point[1] = 0.4;

    const auto skip_nothing = [](MeshIndex) { return false; };
    const auto nearest = tree.nn(data, skip_nothing);

    check(nearest.has_value(), "nn returns a result");
    check(nearest && nearest->index == MeshIndex{3},
          "active boundary node 3 wins nn");
    check(nearest && close(nearest->distance, Scalar{0.01}),
          "nn returns boundary distance 0.01");
    check(data.list.size() == 1 && data.list[0] == MeshIndex{3},
          "SearchData::list receives the public result");
    check(data.list[0] != MeshIndex{4},
          "inactive boundary node 4 is ignored");

    const std::size_t list_capacity = data.list.capacity();
    const std::size_t public_capacity = data.public_candidates.capacity();

    tree.knn(data, std::size_t{3}, skip_nothing);
    check(data.list.size() == 3,
          "knn writes three entries into the recycled list");
    check(data.list.size() >= 2 &&
              data.list[0] == MeshIndex{3} &&
              data.list[1] == MeshIndex{0},
          "knn merges boundary and public results by distance");

    tree.inrange(data, Scalar{1.0}, skip_nothing);
    check(data.list.size() == 2,
          "inrange finds boundary node 3 and ordinary node 0");
    check(data.list.size() == 2 &&
              data.list[0] == MeshIndex{3} &&
              data.list[1] == MeshIndex{0},
          "inrange result order is deterministic");
    check(data.list.capacity() >= list_capacity &&
              data.public_candidates.capacity() >= public_capacity,
          "repeated searches retain allocated capacities");

    const auto skip_boundary = [](MeshIndex index) {
        return index >= MeshIndex{3};
    };
    const auto ordinary_nearest = tree.nn(data, skip_boundary);
    check(ordinary_nearest &&
              ordinary_nearest->index == MeshIndex{0},
          "skip operates in Mesh::Index numbering");
}

void test_nanoflann_native_fields() {
    std::cout << "\n============================================================\n";
    std::cout << "NANOFLANN NATIVE DATA TYPES\n";
    std::cout << "============================================================\n";

    FixedMesh mesh = make_fixed_mesh();
    using Tree = highvoronoi::geometry::NanoflannSearchTree<
        FixedMesh,
        std::size_t>;
    Tree tree(mesh, 4, 1);
    auto data = tree.make_backend_data(point(0.41, 0.4));
    data.reserve(8);

    using BackendIndex = typename Tree::BackendIndex;
    using NativeVectorValue =
        typename decltype(data.backend_indices)::value_type;

    check((std::is_same_v<BackendIndex, std::size_t>),
          "nanoflann backend index is std::size_t");
    check((std::is_same_v<NativeVectorValue, std::size_t>),
          "SearchData exposes a backend-native index buffer");
    check((!std::is_same_v<BackendIndex, MeshIndex>),
          "backend and mesh index types are genuinely different");

    tree.knn(data, 2, [](MeshIndex) { return false; });
    check(!data.backend_indices.empty(),
          "nanoflann writes into the reusable native index buffer");
    check(data.list.size() == 2,
          "native results are exported to the public Mesh::Index list");

    const std::size_t native_index_capacity =
        data.backend_indices.capacity();
    const std::size_t native_distance_capacity =
        data.backend_squared_distances.capacity();

    tree.inrange(data, Scalar{2.0}, [](MeshIndex) { return false; });
    check(data.backend_indices.capacity() >= native_index_capacity &&
              data.backend_squared_distances.capacity() >=
                  native_distance_capacity,
          "nanoflann native buffers retain capacity across searches");
}

void test_dynamic_dimension() {
    std::cout << "\n============================================================\n";
    std::cout << "DYNAMIC DIMENSION\n";
    std::cout << "============================================================\n";

    DynamicMesh mesh = make_dynamic_mesh();
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{});
    auto data = tree.make_backend_data(
        std::vector<Scalar>{0.4, 0.0, 0.0});

    check(data.point.size() == 3,
          "dynamic backend point is initialized to mesh dimension");

    const auto nearest = tree.nn(
        data,
        [](MeshIndex) { return false; });
    check(nearest && nearest->index == MeshIndex{2},
          "dynamic KD-tree includes the active boundary node");
}

void test_factories_and_custom_keyword() {
    std::cout << "\n============================================================\n";
    std::cout << "FACTORIES AND CUSTOM KEYWORD\n";
    std::cout << "============================================================\n";

    FixedMesh mesh = make_fixed_mesh();

    auto brute = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::BruteForceSearch{});
    test_fixed_backend(brute, "BRUTE-FORCE SEARCH TREE");

    auto kd = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{4, 1});
    test_fixed_backend(kd, "NANOFLANN SEARCH TREE");

    check(
        highvoronoi::geometry::IsSearchTreeKeywordV<
            CustomLinearSearch,
            FixedMesh>,
        "a user-defined keyword is recognized at compile time");

    auto custom = highvoronoi::geometry::make_search_tree(
        mesh,
        CustomLinearSearch{});
    auto data = custom.make_backend_data(point(0.41, 0.4));
    const auto nearest = custom.nn(
        data,
        [](MeshIndex) { return false; });
    check(nearest && nearest->index == MeshIndex{3},
          "custom keyword constructs a functioning custom tree");
}

} // namespace

int main() {
    test_factories_and_custom_keyword();
    test_nanoflann_native_fields();
    test_dynamic_dimension();

    std::cout << "\n============================================================\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}
