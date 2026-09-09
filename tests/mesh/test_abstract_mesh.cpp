

#include <highvoronoi/mesh/abstract_mesh.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint16_t;
using Point = highvoronoi::StaticPoint<Scalar, 2>;
using Sigma = std::vector<Index>;
constexpr Scalar tolerance = 1e-12;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    ++performed_checks;
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
    } else {
        ++failed_checks;
        std::cerr << "    [FAIL] " << description << '\n';
    }
}

template <class Function>
void run_test(std::string_view name, Function&& function) {
    std::cout << "\n[TEST] " << name << '\n';
    const std::size_t failures_before = failed_checks;

    try {
        std::forward<Function>(function)();
    } catch (const std::exception& exception) {
        ++failed_checks;
        std::cerr << "    [FAIL] unexpected exception: "
                  << exception.what() << '\n';
    } catch (...) {
        ++failed_checks;
        std::cerr << "    [FAIL] unexpected non-standard exception\n";
    }

    std::cout << (failed_checks == failures_before ? "[PASS] " : "[FAIL] ")
              << name << '\n';
}

Point make_point(Scalar x, Scalar y) {
    Point result;
    result << x, y;
    return result;
}

template <class PointLike>
bool point_equals(
    const PointLike& point,
    Scalar x,
    Scalar y) {
    return std::abs(point[0] - x) <= tolerance &&
           std::abs(point[1] - y) <= tolerance;
}

// ============================================================================
// Small database test double
// ============================================================================

class TestDatabase {
public:
    using Scalar = ::Scalar;
    using Index = ::Index;
    using Sigma = ::Sigma;
    using VertexPoint = ::Point;
    static constexpr int DimensionAtCompileTime = 2;

    struct Record {
        Sigma sigma;
        Point position{};
        Point direction{};
        bool deleted = false;
    };

    [[nodiscard]] std::size_t push(
        const VertexPoint& position,
        const Sigma& sigma) {
        for (const auto& record : records_) {
            if (!record.deleted &&
                record.sigma.size() == sigma.size() &&
                std::equal(record.sigma.begin(), record.sigma.end(), sigma.begin())) {
                return 0;
            }
        }

        Record record;
        record.sigma.assign(sigma.begin(), sigma.end());
        record.position = position;
        records_.push_back(std::move(record));
        return records_.size();
    }

    [[nodiscard]] std::size_t push_facet(
        const VertexPoint& position,
        Sigma& sigma,
        const VertexPoint& direction) {
        std::sort(sigma.begin(), sigma.end());
        const std::size_t address = push(position, sigma);
        if (address != 0) {
            records_.back().direction = direction;
        }
        return address;
    }

    void read(
        std::size_t address,
        VertexPoint& position,
        Sigma& sigma) const {
        const Record& record = records_.at(address - 1);
        sigma.clear();
        if (record.deleted) {
            return;
        }

        sigma.assign(record.sigma.begin(), record.sigma.end());
        position = record.position;
    }

    void read_facet(
        std::size_t address,
        VertexPoint& position,
        Sigma& sigma,
        VertexPoint& direction) const {
        read(address, position, sigma);
        if (!sigma.empty()) {
            direction = records_.at(address - 1).direction;
        }
    }

    [[nodiscard]] bool contains(const Sigma& sigma) const {
        for (const auto& record : records_) {
            if (!record.deleted &&
                record.sigma.size() == sigma.size() &&
                std::equal(record.sigma.begin(), record.sigma.end(), sigma.begin())) {
                return true;
            }
        }
        return false;
    }

    bool erase(std::size_t address, const Sigma& sigma) {
        Record& record = records_.at(address - 1);
        if (record.deleted ||
            record.sigma.size() != sigma.size() ||
            !std::equal(record.sigma.begin(), record.sigma.end(), sigma.begin())) {
            return false;
        }

        record.deleted = true;
        return true;
    }

private:
    std::vector<Record> records_;
};

// ============================================================================
// Minimal vector-like address list
// ============================================================================

template <class T>
class AppendOnlyVectorLike {
public:
    [[nodiscard]] std::size_t size() const noexcept {
        return data_.size();
    }

    [[nodiscard]] T operator[](std::size_t index) const {
        return data_[index];
    }

    void push_back(T value) {
        data_.push_back(value);
    }

private:
    std::vector<T> data_;
};

// ============================================================================
// Public node adapter using the unified node interface
// ============================================================================

class PublicNodes final
    : public highvoronoi::AbstractVoronoiNodes<Scalar, 2, Index> {
public:
    using Base = highvoronoi::AbstractVoronoiNodes<Scalar, 2, Index>;

    PublicNodes(
        const highvoronoi::VoronoiNodes<Scalar, 2, Index>& internal_nodes,
        const std::vector<Index>& public_to_internal)
        : Base(Index{2}),
          internal_nodes_(internal_nodes),
          public_to_internal_(public_to_internal) {}

    [[nodiscard]] Index size() const noexcept override {
        return static_cast<Index>(public_to_internal_.size());
    }

    [[nodiscard]] Scalar get_data(
        Index public_index,
        Index coordinate) const override {
        return internal_nodes_.get_data(
            public_to_internal_.at(public_index),
            coordinate);
    }

protected:
    void copy_node_impl(Index public_index, Scalar* target) const override {
        internal_nodes_.copy_node(
            public_to_internal_.at(public_index),
            target);
    }

private:
    const highvoronoi::VoronoiNodes<Scalar, 2, Index>& internal_nodes_;
    const std::vector<Index>& public_to_internal_;
};

// ============================================================================
// Concrete AbstractMesh fixture
// ============================================================================

template <class AddressListT = std::vector<std::size_t>>
class TestMesh final
    : public highvoronoi::AbstractMesh<
          Scalar,
          Scalar,
          Index,
          2,
          TestDatabase,
          AddressListT,
          highvoronoi::DenseIndexMapping<Index>> {
public:
    using Base = highvoronoi::AbstractMesh<
        Scalar,
        Scalar,
        Index,
        2,
        TestDatabase,
        AddressListT,
        highvoronoi::DenseIndexMapping<Index>>;

    using typename Base::Address;
    using typename Base::AddressList;
    using typename Base::BoundaryType;
    using typename Base::ExtendedNodesAccess;
    using typename Base::NodesAccess;

    explicit TestMesh(Index node_count = Index{4})
        : Base(Index{2}, highvoronoi::DenseIndexMapping<Index>::identity(node_count)),
          internal_nodes_(node_count),
          public_to_internal_(static_cast<std::size_t>(node_count)),
          internal_to_public_(static_cast<std::size_t>(node_count)),
          public_nodes_(internal_nodes_, public_to_internal_),
          primary_(static_cast<std::size_t>(node_count)),
          secondary_(static_cast<std::size_t>(node_count)) {
        for (Index i = 0; i < node_count; ++i) {
            public_to_internal_[i] = i;
            internal_to_public_[i] = i;
            internal_nodes_.set(
                i,
                make_point(
                    static_cast<Scalar>(i),
                    static_cast<Scalar>(i) + Scalar{0.5}));
        }
    }

private:
    [[nodiscard]] const NodesAccess& nodes_impl() const noexcept override {
        return public_nodes_;
    }

    [[nodiscard]] ExtendedNodesAccess& extended_nodes_impl() noexcept override {
        return public_nodes_;
    }

    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept override {
        return public_nodes_;
    }

    [[nodiscard]] const BoundaryType& boundary_impl() const noexcept override {
        return boundary_;
    }

    void set_boundary_impl(BoundaryType boundary) override {
        boundary_ = std::move(boundary);
    }

    [[nodiscard]] TestDatabase& database_impl() noexcept override {
        return database_;
    }

    [[nodiscard]] const TestDatabase& database_impl() const noexcept override {
        return database_;
    }

    [[nodiscard]] Index internal_node_count_impl() const noexcept override {
        return internal_nodes_.size();
    }

    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        return primary_.at(internal_node);
    }

    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return secondary_.at(internal_node);
    }

    void register_primary_vertex_impl(
        Index internal_node,
        Address address) override {
        primary_.at(internal_node).push_back(address);
    }

    void register_secondary_vertex_impl(
        Index internal_node,
        Address address) override {
        secondary_.at(internal_node).push_back(address);
    }

    [[nodiscard]] const AddressList&
    infinite_edge_addresses_impl() const override {
        return infinite_edges_;
    }

    void register_infinite_edge_impl(Address address) override {
        infinite_edges_.push_back(address);
    }

    void mark_internal_node_deleted_impl(Index internal_node) override {
        const auto old_public =
            this->index_mapping().internal_to_public(internal_node);
        if (!old_public) {
            return;
        }
        public_to_internal_.erase(
            public_to_internal_.begin() +
            static_cast<std::ptrdiff_t>(*old_public));

        std::fill(
            internal_to_public_.begin(),
            internal_to_public_.end(),
            deleted_marker());

        for (std::size_t public_index = 0;
             public_index < public_to_internal_.size();
             ++public_index) {
            internal_to_public_[public_to_internal_[public_index]] =
                static_cast<Index>(public_index);
        }
        this->index_mapping().assign(
            public_to_internal_,
            internal_to_public_);
    }

    [[nodiscard]] static constexpr Index deleted_marker() noexcept {
        return std::numeric_limits<Index>::max();
    }

    highvoronoi::VoronoiNodes<Scalar, 2, Index> internal_nodes_;
    std::vector<Index> public_to_internal_;
    std::vector<Index> internal_to_public_;
    PublicNodes public_nodes_;
    BoundaryType boundary_;
    TestDatabase database_;
    std::vector<AddressList> primary_;
    std::vector<AddressList> secondary_;
    AddressList infinite_edges_;
};

// ============================================================================
// Tests
// ============================================================================

void test_unified_node_point_contract() {
    using Mesh = TestMesh<>;

    static_assert(std::is_same_v<
        typename Mesh::NodePoint,
        highvoronoi::PointType<Scalar, 2>>);
    static_assert(std::is_same_v<
        typename Mesh::VertexPoint,
        highvoronoi::PointType<Scalar, 2>>);
    static_assert(std::is_same_v<
        typename Mesh::NodesAccess,
        highvoronoi::AbstractVoronoiNodes<Scalar, 2, Index>>);
    static_assert(std::is_same_v<
        typename Mesh::ExtendedNodesAccess,
        typename Mesh::NodesAccess>);

    Mesh mesh;
    const auto point = mesh.nodes()[Index{2}];

    check(
        point_equals(point, 2.0, 2.5),
        "operator[] returns the common owning NodePoint");

    Point copied;
    mesh.nodes().copy_node(Index{3}, copied);
    check(
        point_equals(copied, 3.0, 3.5),
        "copy_node writes into caller-owned NodePoint storage");

    check(
        std::abs(mesh.nodes().get_data(Index{1}, Index{1}) - 1.5) <= tolerance,
        "get_data provides coordinate-wise node access");
}

void test_store_contains_and_duplicate() {
    TestMesh<> mesh;

    Sigma scratch;
    const Sigma sigma{Index{2}, Index{0}, Index{1}};
    const Point position = make_point(10.0, 11.0);

    const auto address = mesh.store_vertex(position, sigma, scratch);
    check(address != 0, "first vertex insertion succeeds");
    check(
        scratch == Sigma({Index{0}, Index{1}, Index{2}}),
        "caller-owned sigma scratch receives canonical internal indices");

    check(mesh.contains_vertex(sigma), "contains_vertex finds stored signature");

    const auto duplicate = mesh.store_vertex(position, sigma);
    check(duplicate == 0, "duplicate signature is rejected by the database");
}

void test_primary_secondary_and_iteration() {
    TestMesh<> mesh;

    const auto a = mesh.store_vertex(
        make_point(10.0, 10.0),
        Sigma{Index{1}, Index{2}});
    const auto b = mesh.store_vertex(
        make_point(20.0, 20.0),
        Sigma{Index{0}, Index{1}});

    std::size_t primary_at_one = 0;
    for (const auto& record : mesh.primary_vertices(Index{1})) {
        ++primary_at_one;
        check(record.address == a, "cell 1 primary range contains its owned vertex");
    }

    std::size_t secondary_at_one = 0;
    for (const auto& record : mesh.secondary_vertices(Index{1})) {
        ++secondary_at_one;
        check(record.address == b, "cell 1 secondary range contains the lower-owner vertex");
    }

    std::size_t combined = 0;
    for ([[maybe_unused]] const auto& record : mesh.vertices(Index{1})) {
        ++combined;
    }

    check(primary_at_one == 1, "primary range has one record");
    check(secondary_at_one == 1, "secondary range has one record");
    check(combined == 2, "combined range contains primary and secondary records");
}

void test_erase_vertex() {
    TestMesh<> mesh;

    const auto address = mesh.store_vertex(
        make_point(5.0, 6.0),
        Sigma{Index{0}, Index{2}});

    Sigma scratch;
    check(mesh.erase_vertex(address, scratch), "erase_vertex removes an active record");
    check(!mesh.contains_vertex(Sigma{Index{0}, Index{2}}),
          "erased signature is no longer contained");
    check(!mesh.erase_vertex(address, scratch),
          "erasing the same tombstoned record again returns false");

    std::size_t visible = 0;
    for ([[maybe_unused]] const auto& record : mesh.vertices(Index{0})) {
        ++visible;
    }
    check(visible == 0, "vertex iteration skips tombstoned records");
}

void test_node_deletion_updates_public_mapping() {
    TestMesh<> mesh;

    (void)mesh.store_vertex(
        make_point(1.0, 1.0),
        Sigma{Index{0}, Index{1}});
    (void)mesh.store_vertex(
        make_point(2.0, 2.0),
        Sigma{Index{1}, Index{2}});
    (void)mesh.store_vertex(
        make_point(3.0, 3.0),
        Sigma{Index{2}, Index{3}});

    const std::size_t deleted = mesh.erase_nodes_if(
        [](Index public_index, const Point&) {
            return public_index == Index{1};
        });

    check(deleted == 1, "erase_nodes_if deletes exactly one public node");
    check(mesh.size() == Index{3}, "public node count shrinks after deletion");
    check(
        point_equals(mesh.nodes()[Index{1}], 2.0, 2.5),
        "public node 1 now maps to former internal node 2");

    std::size_t remaining = 0;
    for ([[maybe_unused]] const auto& record : mesh.vertices(Index{1})) {
        ++remaining;
    }
    check(remaining == 1,
          "vertices touching deleted node are tombstoned while unaffected vertex remains");
}

void test_custom_address_list_type() {
    using CustomList = AppendOnlyVectorLike<std::size_t>;
    using Mesh = TestMesh<CustomList>;

    static_assert(std::is_same_v<typename Mesh::AddressList, CustomList>);

    Mesh mesh;
    (void)mesh.store_vertex(
        make_point(4.0, 4.0),
        Sigma{Index{0}, Index{1}});

    std::size_t count = 0;
    for ([[maybe_unused]] const auto& record : mesh.vertices(Index{1})) {
        ++count;
    }

    check(count == 1,
          "AbstractMesh works with AddressList exposing only size/operator[]/push_back");
}

} // namespace

int main() {
    std::cout
        << "============================================================\n"
        << "AbstractMesh unified-node tests\n"
        << "============================================================\n";

    run_test("unified node/point contract", test_unified_node_point_contract);
    run_test("store, contains and duplicate", test_store_contains_and_duplicate);
    run_test("primary/secondary iteration", test_primary_secondary_and_iteration);
    run_test("erase vertex", test_erase_vertex);
    run_test("node deletion and mapping", test_node_deletion_updates_public_mapping);
    run_test("custom address-list type", test_custom_address_list_type);

    std::cout
        << "\n============================================================\n"
        << "Checks: " << performed_checks
        << " | failed: " << failed_checks << '\n'
        << "============================================================\n";

    return failed_checks == 0 ? 0 : 1;
}




