#include <highvoronoi/geometry/abstract_mesh.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file test_abstract_mesh.cpp
 * @brief Focused tests for AbstractMesh algorithms and its node-access bridge.
 *
 * VoronoiNodes has its own modular test suite. That suite already verifies:
 *
 * - Stored, Computed, and Hybrid ordinary-node semantics,
 * - Stored -> Stored, Computed -> Hybrid, and Hybrid -> Hybrid extension,
 * - operator[] return types and ownership,
 * - mirror activation and precomputed mirror storage.
 *
 * This file does not repeat those tests. It verifies only:
 *
 * 1. AbstractMesh maps its BaseNodeAccessMode to the same extended access mode
 *    selected by voronoi_nodes.hpp.
 * 2. AbstractMesh vertex storage, address registration, iteration, deletion,
 *    signature conversion, buffer reuse, and filtering.
 * 3. AddressListT needs only size(), const operator[](), and push_back().
 *
 * The behavioural fixture deliberately uses Stored nodes. The algorithms under
 * test operate through the AbstractMesh interface; the detailed behaviour of
 * the other node-access modes belongs to the VoronoiNodes test suite.
 */

namespace {

using Scalar = double;
using Index = std::uint16_t;
using Point = highvoronoi::StaticPoint<Scalar, 2>;
using Sigma = std::vector<Index>;
constexpr Scalar tolerance = 1e-12;

// ============================================================================
// Minimal documented test framework
// ============================================================================

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

void explain_test(
    std::string_view purpose,
    std::string_view setup,
    std::string_view success_condition) {
    std::cout << "  Purpose:  " << purpose << '\n'
              << "  Setup:    " << setup << '\n'
              << "  Success:  " << success_condition << '\n';
}

template <typename Function>
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

    if (failed_checks == failures_before) {
        std::cout << "[PASS] " << name << '\n';
    } else {
        std::cout << "[FAIL] " << name << '\n';
    }
}

Point make_point(Scalar x, Scalar y) {
    Point result;
    result << x, y;
    return result;
}

template <class PointLike>
bool point_equals(
    const PointLike& point,
    Scalar expected_x,
    Scalar expected_y) {
    return std::abs(point[Index{0}] - expected_x) <= tolerance &&
           std::abs(point[Index{1}] - expected_y) <= tolerance;
}

// ============================================================================
// Instrumented database test double
// ============================================================================

/**
 * @brief Minimal observable implementation of the DatabaseT contract.
 *
 * AbstractMesh does not own a fixed database implementation. A concrete test
 * mesh therefore needs some DatabaseT providing:
 *
 * - `push(position, sigma)`,
 * - `read(address, position, sigma)`,
 * - `contains(sigma)`,
 * - `erase(address, sigma)`.
 *
 * This class implements only those operations in the simplest possible form.
 * It intentionally uses a vector of records so that tests remain independent
 * of the production HVDataBase implementation. Read and erase counters are
 * provided solely to verify whether AbstractMesh traverses primary addresses
 * once or accidentally revisits all secondary registrations.
 */
class InstrumentedTestDatabase {
public:
    using Scalar = ::Scalar;
    using Index = ::Index;

    struct Record {
        Sigma sigma;
        std::vector<Scalar> position;
        std::vector<Scalar> direction;
        bool deleted = false;
    };

    template <class RVector, class SigmaVector>
    [[nodiscard]] std::size_t push(
        const RVector& position,
        const SigmaVector& sigma) {
        ++push_count_;

        for (const Record& record : records_) {
            if (!record.deleted &&
                record.sigma.size() == sigma.size() &&
                std::equal(
                    record.sigma.begin(),
                    record.sigma.end(),
                    sigma.begin())) {
                return 0;
            }
        }

        Record record;
        record.sigma.assign(sigma.begin(), sigma.end());
        record.position.assign(
            position.data(),
            position.data() + position.size());
        records_.push_back(std::move(record));

        // The production database contract uses zero as "not inserted".
        // Therefore valid test addresses are one-based.
        return records_.size();
    }

    template <class RVector, class SigmaVector, class UVector>
    [[nodiscard]] std::size_t push_facet(
        const RVector& position,
        SigmaVector& sigma,
        const UVector& direction) {
        std::sort(sigma.begin(), sigma.end());
        const std::size_t address = push(position, sigma);
        if (address == 0) {
            return 0;
        }
        records_.back().direction.assign(
            direction.data(),
            direction.data() + direction.size());
        return address;
    }

    template <class RVector, class SigmaVector>
    void read(
        std::size_t address,
        RVector& position,
        SigmaVector& sigma) const {
        ++read_count_;
        last_position_destination_ =
            static_cast<const void*>(position.data());

        const Record& record = records_.at(address - 1);
        sigma.clear();

        // An empty sigma is the database signal for a deleted record.
        if (record.deleted) {
            return;
        }

        sigma.assign(record.sigma.begin(), record.sigma.end());
        std::copy(
            record.position.begin(),
            record.position.end(),
            position.data());
    }

    template <class RVector, class SigmaVector, class UVector>
    void read_facet(
        std::size_t address,
        RVector& position,
        SigmaVector& sigma,
        UVector& direction) const {
        read(address, position, sigma);
        if (sigma.empty()) {
            return;
        }
        const Record& record = records_.at(address - 1);
        std::copy(
            record.direction.begin(),
            record.direction.end(),
            direction.data());
    }

    template <class SigmaVector>
    [[nodiscard]] bool contains(
        const SigmaVector& sigma) const {
        for (const Record& record : records_) {
            if (record.deleted || record.sigma.size() != sigma.size()) {
                continue;
            }
            if (std::equal(
                    record.sigma.begin(),
                    record.sigma.end(),
                    sigma.begin())) {
                return true;
            }
        }
        return false;
    }

    template <class SigmaVector>
    bool erase(
        std::size_t address,
        const SigmaVector& sigma) {
        ++erase_count_;

        Record& record = records_.at(address - 1);
        if (record.deleted || record.sigma.size() != sigma.size()) {
            return false;
        }
        if (!std::equal(
                record.sigma.begin(),
                record.sigma.end(),
                sigma.begin())) {
            return false;
        }

        record.deleted = true;
        return true;
    }

    void reset_counters() const noexcept {
        push_count_ = 0;
        read_count_ = 0;
        erase_count_ = 0;
    }

    void reset_read_count() const noexcept {
        read_count_ = 0;
    }

    [[nodiscard]] std::size_t push_count() const noexcept {
        return push_count_;
    }

    [[nodiscard]] std::size_t read_count() const noexcept {
        return read_count_;
    }

    [[nodiscard]] std::size_t erase_count() const noexcept {
        return erase_count_;
    }

    [[nodiscard]] const void*
    last_position_destination() const noexcept {
        return last_position_destination_;
    }

    [[nodiscard]] std::size_t active_record_count() const noexcept {
        return static_cast<std::size_t>(std::count_if(
            records_.begin(),
            records_.end(),
            [](const Record& record) {
                return !record.deleted;
            }));
    }

private:
    std::vector<Record> records_;
    mutable std::size_t push_count_ = 0;
    mutable std::size_t read_count_ = 0;
    mutable std::size_t erase_count_ = 0;
    mutable const void* last_position_destination_ = nullptr;
};

// ============================================================================
// Minimal vector-like address container used to test AddressListT
// ============================================================================

/**
 * @brief Deliberately restricted vector-like append-only container.
 *
 * There are no iterators, no `begin()`, no `end()`, and no mutable
 * `operator[]`. If AbstractMesh accidentally assumes more than `size()`,
 * const `operator[]`, and `push_back()`, compilation of the corresponding test
 * fixture fails.
 */
template <class T>
class AppendOnlyVectorLike {
public:
    using value_type = T;
    using size_type = std::size_t;

    [[nodiscard]] size_type size() const noexcept {
        return data_.size();
    }

    [[nodiscard]] value_type operator[](size_type index) const {
        return data_[index];
    }

    void push_back(value_type value) {
        data_.push_back(std::move(value));
    }

private:
    std::vector<value_type> data_;
};

template <class T, class = void>
struct HasMemberBegin : std::false_type {};

template <class T>
struct HasMemberBegin<
    T,
    std::void_t<decltype(std::declval<const T&>().begin())>>
    : std::true_type {};

// ============================================================================
// Public node adapters for the three NodeAccessMode branches
// ============================================================================

/**
 * @brief Public node adapter preserving Stored access semantics.
 *
 * Public indices are translated through `public_to_internal_`, after which a
 * stable pointer into the immutable internal node storage is returned.
 */
class PublicStoredNodes final
    : public highvoronoi::StoredNodeAccess<Scalar, 2, Index> {
public:
    using Base = highvoronoi::StoredNodeAccess<Scalar, 2, Index>;

    PublicStoredNodes(
        const highvoronoi::VoronoiNodes<Scalar, 2, Index>& internal_nodes,
        const std::vector<Index>& public_to_internal)
        : Base(Index{2}),
          internal_nodes_(internal_nodes),
          public_to_internal_(public_to_internal) {}

    [[nodiscard]] Index size() const noexcept override {
        return static_cast<Index>(public_to_internal_.size());
    }

protected:
    [[nodiscard]] const Scalar*
    get_stored_node_pointer(Index public_index) const override {
        return internal_nodes_.stable_node_data(
            public_to_internal_.at(public_index));
    }

private:
    const highvoronoi::VoronoiNodes<Scalar, 2, Index>& internal_nodes_;
    const std::vector<Index>& public_to_internal_;
};


// ============================================================================
// Concrete stored mesh fixture for AbstractMesh algorithms
// ============================================================================

/**
 * @brief Minimal concrete mesh used to exercise AbstractMesh itself.
 *
 * The fixture intentionally uses only Stored node access. Computed and Hybrid
 * node semantics, including real ExtendedVoronoiNodes behaviour, are already
 * covered by the dedicated VoronoiNodes tests.
 */
template <class AddressListT = std::vector<std::size_t>>
class TestMeshT final
    : public highvoronoi::AbstractMesh<
          Scalar,
          Scalar,
          Index,
          2,
          highvoronoi::NodeAccessMode::Stored,
          InstrumentedTestDatabase,
          AddressListT> {
public:
    using Base = highvoronoi::AbstractMesh<
        Scalar,
        Scalar,
        Index,
        2,
        highvoronoi::NodeAccessMode::Stored,
        InstrumentedTestDatabase,
        AddressListT>;
    using typename Base::Address;
    using typename Base::AddressList;
    using typename Base::BoundaryType;
    using typename Base::ExtendedNodesAccess;
    using typename Base::NodesAccess;
    using typename Base::VertexPoint;

    explicit TestMeshT(Index count = Index{5})
        : Base(Index{2}),
          internal_nodes_(count),
          public_to_internal_(static_cast<std::size_t>(count)),
          internal_to_public_(static_cast<std::size_t>(count)),
          public_nodes_(internal_nodes_, public_to_internal_),
          primary_address_lists_(static_cast<std::size_t>(count)),
          secondary_address_lists_(static_cast<std::size_t>(count)) {
        for (Index index = Index{0}; index < count; ++index) {
            public_to_internal_[index] = index;
            internal_to_public_[index] = index;
            internal_nodes_.set(
                index,
                make_point(
                    static_cast<Scalar>(index),
                    Scalar{10} + static_cast<Scalar>(index)));
        }
    }

    [[nodiscard]] const InstrumentedTestDatabase&
    database() const noexcept {
        return database_;
    }

private:
    [[nodiscard]] const NodesAccess&
    nodes_impl() const noexcept override {
        return public_nodes_;
    }

    [[nodiscard]] ExtendedNodesAccess&
    extended_nodes_impl() noexcept override {
        return public_nodes_;
    }

    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept override {
        return public_nodes_;
    }

    [[nodiscard]] const BoundaryType&
    boundary_impl() const noexcept override {
        return boundary_;
    }

    void set_boundary_impl(BoundaryType boundary) override {
        boundary_ = std::move(boundary);
    }

    [[nodiscard]] InstrumentedTestDatabase&
    database_impl() noexcept override {
        return database_;
    }

    [[nodiscard]] const InstrumentedTestDatabase&
    database_impl() const noexcept override {
        return database_;
    }

    [[nodiscard]] Index
    internal_node_count_impl() const noexcept override {
        return internal_nodes_.size();
    }

    [[nodiscard]] Index
    public_node_to_internal_impl(Index public_node) const override {
        return public_to_internal_.at(public_node);
    }

    [[nodiscard]] std::optional<Index>
    internal_node_to_public_impl(Index internal_node) const override {
        const Index public_node =
            internal_to_public_.at(internal_node);
        if (public_node == deleted_marker()) {
            return std::nullopt;
        }
        return public_node;
    }

    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(
        Index internal_node) const override {
        return primary_address_lists_.at(internal_node);
    }

    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(
        Index internal_node) const override {
        return secondary_address_lists_.at(internal_node);
    }

    void register_primary_vertex_impl(
        Index internal_node,
        Address address) override {
        primary_address_lists_.at(internal_node).push_back(address);
    }

    void register_secondary_vertex_impl(
        Index internal_node,
        Address address) override {
        secondary_address_lists_.at(internal_node).push_back(address);
    }

    [[nodiscard]] const AddressList&
    infinite_edge_addresses_impl() const override {
        return infinite_edge_addresses_;
    }

    void register_infinite_edge_impl(Address address) override {
        infinite_edge_addresses_.push_back(address);
    }

    void mark_internal_node_deleted_impl(
        Index internal_node) override {
        if (internal_to_public_.at(internal_node) == deleted_marker()) {
            return;
        }

        public_to_internal_.erase(
            std::remove(
                public_to_internal_.begin(),
                public_to_internal_.end(),
                internal_node),
            public_to_internal_.end());

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
    }

    [[nodiscard]] static constexpr Index deleted_marker() noexcept {
        return std::numeric_limits<Index>::max();
    }

    highvoronoi::VoronoiNodes<Scalar, 2, Index> internal_nodes_;
    std::vector<Index> public_to_internal_;
    std::vector<Index> internal_to_public_;
    PublicStoredNodes public_nodes_;
    BoundaryType boundary_;
    InstrumentedTestDatabase database_;
    std::vector<AddressList> primary_address_lists_;
    std::vector<AddressList> secondary_address_lists_;
    AddressList infinite_edge_addresses_;
};

using TestMesh = TestMeshT<>;
using WrappedAddressList = AppendOnlyVectorLike<std::size_t>;
using WrappedTestMesh = TestMeshT<WrappedAddressList>;

// ============================================================================
// Tests
// ============================================================================

/**
 * @brief Verify only the bridge from AbstractMesh to the established
 *        VoronoiNodes access-mode rule.
 *
 * The detailed node and mirror semantics are tested in
 * hvcc_voronoi_nodes_compile_contracts_20260803.hpp and
 * hvcc_voronoi_nodes_extended_tests_20260803.hpp.
 */
void test_abstract_mesh_access_mode_bridge() {
    using namespace highvoronoi;

    explain_test(
        "Verify that AbstractMesh uses the same extended access-mode mapping as voronoi_nodes.hpp.",
        "Instantiate only the three AbstractMesh base types and inspect their compile-time aliases.",
        "Stored maps to Stored; Computed and Hybrid map to Hybrid. No node values or mirror behaviour are retested here.");

    using StoredBase = AbstractMesh<
        Scalar,
        Scalar,
        Index,
        2,
        NodeAccessMode::Stored,
        InstrumentedTestDatabase>;
    using ComputedBase = AbstractMesh<
        Scalar,
        Scalar,
        Index,
        2,
        NodeAccessMode::Computed,
        InstrumentedTestDatabase>;
    using HybridBase = AbstractMesh<
        Scalar,
        Scalar,
        Index,
        2,
        NodeAccessMode::Hybrid,
        InstrumentedTestDatabase>;

    static_assert(
        StoredBase::ExtendedNodeMode ==
        detail::ExtendedNodeAccessModeV<NodeAccessMode::Stored>);
    static_assert(
        ComputedBase::ExtendedNodeMode ==
        detail::ExtendedNodeAccessModeV<NodeAccessMode::Computed>);
    static_assert(
        HybridBase::ExtendedNodeMode ==
        detail::ExtendedNodeAccessModeV<NodeAccessMode::Hybrid>);

    static_assert(
        StoredBase::ExtendedNodeMode == NodeAccessMode::Stored);
    static_assert(
        ComputedBase::ExtendedNodeMode == NodeAccessMode::Hybrid);
    static_assert(
        HybridBase::ExtendedNodeMode == NodeAccessMode::Hybrid);

    static_assert(std::is_same_v<
        StoredBase::ExtendedNodesAccess,
        StoredNodeAccess<Scalar, 2, Index>>);
    static_assert(std::is_same_v<
        ComputedBase::ExtendedNodesAccess,
        HybridNodeAccess<Scalar, 2, Index>>);
    static_assert(std::is_same_v<
        HybridBase::ExtendedNodesAccess,
        HybridNodeAccess<Scalar, 2, Index>>);

    check(true,
          "AbstractMesh aliases follow the VoronoiNodes extended-mode rule");
}

/**
 * Verify that AddressListT really is a minimal vector-like template parameter
 * rather than an implicit std::vector dependency.
 */
void test_vector_like_address_list_template() {
    using WrappedBase = highvoronoi::AbstractMesh<
        Scalar,
        Scalar,
        Index,
        2,
        highvoronoi::NodeAccessMode::Stored,
        InstrumentedTestDatabase,
        WrappedAddressList>;

    explain_test(
        "Verify that AbstractMesh accepts an address container without iterators or mutable element access.",
        "Use AppendOnlyVectorLike, which provides only size(), const operator[](), and push_back().",
        "The custom type compiles, stores addresses, supports combined iteration, and supports primary-only global traversal.");

    static_assert(std::is_same_v<
        typename WrappedBase::AddressList,
        WrappedAddressList>);
    static_assert(!HasMemberBegin<WrappedAddressList>::value,
                  "The test container must not accidentally provide begin().");

    WrappedTestMesh mesh;

    const std::size_t primary_at_one = mesh.store_vertex(
        make_point(1.0, 1.0),
        Sigma{Index{1}, Index{3}});
    const std::size_t secondary_at_one = mesh.store_vertex(
        make_point(2.0, 2.0),
        Sigma{Index{0}, Index{1}});

    check(primary_at_one != 0 && secondary_at_one != 0,
          "store_vertex appends through AddressListT::push_back");

    std::vector<std::size_t> combined_addresses;
    for (const auto& record : mesh.vertices(Index{1})) {
        combined_addresses.push_back(record.address);
    }

    check(combined_addresses ==
              std::vector<std::size_t>{
                  primary_at_one,
                  secondary_at_one},
          "combined iteration reads primary before secondary via size and operator[]");

    mesh.database().reset_read_count();
    const std::size_t erased = mesh.erase_vertices_if(
        [](const Sigma&, const Point&) {
            return false;
        });

    check(erased == 0,
          "a false predicate leaves both database records active");
    check(mesh.database().read_count() == 2,
          "global traversal reads the two primary registrations exactly once");
}

/**
 * Verify public-to-internal signature conversion, sorting, caller-owned buffer
 * reuse, duplicate detection, and the explicitly supported in-place path.
 */
void test_store_and_contains_with_recycled_buffer() {
    explain_test(
        "Verify canonical signature conversion and allocation reuse in store_vertex and contains_vertex.",
        "Pass unsorted public signatures through one pre-reserved Sigma buffer, then repeat with the same object as input and output.",
        "The internal signature is sorted, storage is reused, contains finds permutations, and a duplicate insertion returns address zero.");

    TestMesh mesh;
    const Point position = make_point(0.25, 0.5);

    Sigma public_sigma{Index{2}, Index{0}, Index{1}};
    Sigma conversion_buffer;
    conversion_buffer.reserve(16);
    const Index* const original_storage = conversion_buffer.data();

    const std::size_t address = mesh.store_vertex(
        position,
        public_sigma,
        conversion_buffer);

    check(address == 1,
          "the first canonical signature is inserted at address 1");
    check(conversion_buffer == Sigma({Index{0}, Index{1}, Index{2}}),
          "the caller buffer contains the sorted internal signature");
    check(conversion_buffer.data() == original_storage,
          "store_vertex reuses the preallocated conversion storage");

    public_sigma = Sigma{Index{1}, Index{2}, Index{0}};
    check(mesh.contains_vertex(public_sigma, conversion_buffer),
          "contains_vertex recognizes a permutation of the stored signature");
    check(conversion_buffer.data() == original_storage,
          "contains_vertex reuses the same conversion storage");

    Sigma in_place_sigma{Index{2}, Index{1}, Index{0}};
    const std::size_t duplicate = mesh.store_vertex(
        position,
        in_place_sigma,
        in_place_sigma);

    check(duplicate == 0,
          "the database reports an already existing canonical signature with address zero");
    check(in_place_sigma == Sigma({Index{0}, Index{1}, Index{2}}),
          "the supported in-place conversion leaves a canonical signature");
}

/**
 * Verify the three erase_vertex buffer variants and make their differing
 * ownership rules observable through the test database.
 */
void test_erase_vertex_buffer_ownership() {
    explain_test(
        "Verify which scratch buffers each erase_vertex overload owns and reuses.",
        "Erase three records using caller sigma plus mesh position, fully mesh-owned buffers, and fully caller-owned buffers.",
        "Serial overloads reuse the mesh position object; the three-argument overload writes into the caller position object.");

    TestMesh mesh;

    const std::size_t first_address = mesh.store_vertex(
        make_point(0.1, 0.2),
        Sigma{Index{0}, Index{1}});
    const std::size_t second_address = mesh.store_vertex(
        make_point(0.3, 0.4),
        Sigma{Index{1}, Index{2}});
    const std::size_t third_address = mesh.store_vertex(
        make_point(0.5, 0.6),
        Sigma{Index{2}, Index{3}});

    Sigma caller_sigma;
    caller_sigma.reserve(8);
    const Index* const original_sigma_storage = caller_sigma.data();

    check(mesh.erase_vertex(first_address, caller_sigma),
          "the caller-sigma overload erases the first active record");
    const void* const first_mesh_position =
        mesh.database().last_position_destination();

    check(caller_sigma == Sigma({Index{0}, Index{1}}),
          "the caller sigma receives the first record signature");
    check(caller_sigma.data() == original_sigma_storage,
          "the caller sigma allocation is reused");

    check(mesh.erase_vertex(second_address),
          "the convenience overload erases the second active record");
    const void* const second_mesh_position =
        mesh.database().last_position_destination();

    check(first_mesh_position == second_mesh_position,
          "both serial overloads use the same mesh-owned position buffer");

    Point caller_position;
    check(mesh.erase_vertex(
              third_address,
              caller_sigma,
              caller_position),
          "the fully caller-owned overload erases the third active record");
    check(mesh.database().last_position_destination() ==
              static_cast<const void*>(caller_position.data()),
          "the three-argument overload writes into the caller position buffer");
    check(mesh.database().active_record_count() == 0,
          "all three database records are tombstoned");
}

struct LvalueNodePredicate {
    std::size_t* calls = nullptr;

    bool operator()(Index public_index, const Point&) {
        ++(*calls);
        return public_index == Index{1};
    }
};

/** Verify that forwarding-reference predicates accept both lvalues and temporaries. */
void test_forwarding_predicates_accept_lvalues_and_temporaries() {
    explain_test(
        "Verify the NodePredicate&& interface without requiring callers to std::move named predicates.",
        "Call erase_nodes_if once with a named function object and once with a temporary lambda.",
        "Both callable forms are invoked over all public nodes and delete exactly their selected node.");

    TestMesh lvalue_mesh(Index{4});
    std::size_t calls = 0;
    LvalueNodePredicate predicate{&calls};
    Sigma deleted_nodes;

    const std::size_t deleted = lvalue_mesh.erase_nodes_if(
        predicate,
        deleted_nodes);

    check(calls == 4,
          "the named lvalue predicate is called once for every public node");
    check(deleted == 1 && deleted_nodes == Sigma({Index{1}}),
          "the lvalue predicate deletes public node 1");

    TestMesh temporary_mesh(Index{4});
    const std::size_t temporary_deleted =
        temporary_mesh.erase_nodes_if(
            [](Index public_index, const Point&) {
                return public_index == Index{2};
            });

    check(temporary_deleted == 1,
          "a temporary lambda directly deletes public node 2");
}

/** Verify primary, secondary, and combined range membership and order. */
void test_primary_secondary_and_combined_ranges() {
    explain_test(
        "Verify the ownership convention for per-node address lists and the three public range functions.",
        "At public node 1 store one vertex whose smallest index is 1 and one whose smallest index is 0.",
        "primary_vertices returns the first, secondary_vertices returns the second, and vertices returns both in primary-first order.");

    TestMesh mesh;

    const std::size_t primary_address = mesh.store_vertex(
        make_point(1.0, 0.0),
        Sigma{Index{1}, Index{2}});
    const std::size_t secondary_address = mesh.store_vertex(
        make_point(2.0, 0.0),
        Sigma{Index{0}, Index{1}});

    check(primary_address == 1 && secondary_address == 2,
          "the fixture creates one primary and one secondary registration at node 1");

    std::vector<std::size_t> primary_addresses;
    std::vector<Sigma> primary_signatures;
    for (const auto& record : mesh.primary_vertices(Index{1})) {
        primary_addresses.push_back(record.address);
        primary_signatures.push_back(record.sigma);
    }

    std::vector<std::size_t> secondary_addresses;
    std::vector<Sigma> secondary_signatures;
    for (const auto& record : mesh.secondary_vertices(Index{1})) {
        secondary_addresses.push_back(record.address);
        secondary_signatures.push_back(record.sigma);
    }

    std::vector<std::size_t> combined_addresses;
    for (const auto& record : mesh.vertices(Index{1})) {
        combined_addresses.push_back(record.address);
    }

    check(primary_addresses ==
              std::vector<std::size_t>({primary_address}) &&
              primary_signatures ==
              std::vector<Sigma>({Sigma{Index{1}, Index{2}}}),
          "primary_vertices returns only the vertex owned by node 1");
    check(secondary_addresses ==
              std::vector<std::size_t>({secondary_address}) &&
              secondary_signatures ==
              std::vector<Sigma>({Sigma{Index{0}, Index{1}}}),
          "secondary_vertices returns only the non-owning registration at node 1");
    check(combined_addresses ==
              std::vector<std::size_t>({
                  primary_address,
                  secondary_address}),
          "vertices concatenates the primary list before the secondary list");
}

/** Verify that deleting a node follows both of its address lists. */
void test_erase_nodes_uses_primary_and_secondary_lists() {
    explain_test(
        "Verify that deleting one node removes every touching vertex, regardless of address ownership.",
        "At node 2 create one vertex registered secondarily, one registered primarily, and one unrelated vertex.",
        "Deleting node 2 tombstones the first two records and leaves exactly the unrelated record active.");

    TestMesh mesh;

    const std::size_t secondary_at_deleted_node = mesh.store_vertex(
        make_point(1.0, 0.0),
        Sigma{Index{0}, Index{2}});
    const std::size_t primary_at_deleted_node = mesh.store_vertex(
        make_point(2.0, 0.0),
        Sigma{Index{2}, Index{3}});
    const std::size_t unaffected = mesh.store_vertex(
        make_point(3.0, 0.0),
        Sigma{Index{0}, Index{1}});

    check(secondary_at_deleted_node == 1 &&
              primary_at_deleted_node == 2 &&
              unaffected == 3,
          "the fixture contains secondary, primary, and unaffected records");

    const std::size_t deleted = mesh.erase_nodes_if(
        [](Index public_index, const Point&) {
            return public_index == Index{2};
        });

    check(deleted == 1,
          "erase_nodes_if removes exactly public node 2");
    check(mesh.database().active_record_count() == 1,
          "only the vertex unrelated to node 2 remains active");

    std::vector<std::size_t> surviving_addresses;
    for (const auto& record : mesh.vertices(Index{0})) {
        surviving_addresses.push_back(record.address);
    }
    check(surviving_addresses == std::vector<std::size_t>{unaffected},
          "the surviving public range contains exactly the unrelated address");
}

/** Verify that global vertex traversal visits only unique primary addresses. */
void test_erase_vertices_visits_primary_lists_only() {
    explain_test(
        "Verify that erase_vertices_if evaluates every database record once rather than once per node registration.",
        "Store three vertices with three generating nodes each, creating three primary and six secondary registrations.",
        "The predicate and database read counter are both exactly three, while no record is erased.");

    TestMesh mesh;

    const std::size_t first = mesh.store_vertex(
        make_point(0.1, 0.1),
        Sigma{Index{0}, Index{1}, Index{2}});
    const std::size_t second = mesh.store_vertex(
        make_point(0.2, 0.2),
        Sigma{Index{1}, Index{2}, Index{3}});
    const std::size_t third = mesh.store_vertex(
        make_point(0.3, 0.3),
        Sigma{Index{0}, Index{3}, Index{4}});

    check(first == 1 && second == 2 && third == 3,
          "three distinct records are stored before global traversal");

    mesh.database().reset_read_count();
    std::size_t predicate_calls = 0;

    const std::size_t erased = mesh.erase_vertices_if(
        [&predicate_calls](const Sigma&, const Point&) {
            ++predicate_calls;
            return false;
        });

    check(erased == 0,
          "a predicate returning false erases no record");
    check(predicate_calls == 3,
          "the predicate is invoked once for each unique primary record");
    check(mesh.database().read_count() == 3,
          "the database is read three times, not once per nine registrations");
    check(mesh.database().active_record_count() == 3,
          "all records remain active after the false predicate");
}

/** Verify the coordinated node-and-vertex filtering contract. */
void test_filter_receives_sorted_deleted_public_nodes() {
    explain_test(
        "Verify filter ordering, predicate arguments, automatic dependent deletion, and public renumbering.",
        "Delete old public nodes 1 and 3; additionally select one independent vertex; leave one vertex to survive renumbering.",
        "The vertex predicate sees old public numbering and the sorted deletion list, two vertices are deleted, and the survivor is exposed under new numbering.");

    TestMesh mesh;

    // A touches deleted node 1 and must be removed automatically.
    const std::size_t address_a = mesh.store_vertex(
        make_point(1.0, 0.0),
        Sigma{Index{0}, Index{1}});

    // B touches no deleted node but is selected by the vertex predicate.
    const std::size_t address_b = mesh.store_vertex(
        make_point(2.0, 0.0),
        Sigma{Index{2}, Index{4}});

    // C survives. After deleting old public nodes 1 and 3, old signature
    // {0, 4} becomes new public signature {0, 2}.
    const std::size_t address_c = mesh.store_vertex(
        make_point(3.0, 0.0),
        Sigma{Index{0}, Index{4}});

    check(address_a == 1 && address_b == 2 && address_c == 3,
          "the fixture stores dependent, predicate-selected, and surviving vertices");

    Sigma deleted_public_nodes;
    deleted_public_nodes.reserve(8);
    std::size_t vertex_predicate_calls = 0;
    bool every_call_received_sorted_nodes = true;
    bool saw_old_public_signature = false;

    const auto result = mesh.filter(
        [](Index public_index, const Point&) {
            return public_index == Index{1} ||
                   public_index == Index{3};
        },
        [&](const Sigma& deleted_nodes,
            const Sigma& public_sigma,
            const Point&) {
            ++vertex_predicate_calls;
            every_call_received_sorted_nodes =
                every_call_received_sorted_nodes &&
                deleted_nodes == Sigma({Index{1}, Index{3}});

            if (public_sigma == Sigma({Index{2}, Index{4}})) {
                saw_old_public_signature = true;
                return true;
            }
            return false;
        },
        deleted_public_nodes);

    check(deleted_public_nodes == Sigma({Index{1}, Index{3}}),
          "the caller buffer contains sorted pre-filter public node indices");
    check(vertex_predicate_calls == 3,
          "the vertex predicate sees every active vertex exactly once");
    check(every_call_received_sorted_nodes,
          "every vertex-predicate call receives the same deletion context");
    check(saw_old_public_signature,
          "the vertex predicate sees signature {2,4} before public compaction");
    check(result.first == 2,
          "filter reports two deleted nodes");
    check(result.second == 2,
          "filter reports one dependent and one predicate-selected vertex deletion");
    check(mesh.size() == Index{3},
          "public node numbering is compacted from five nodes to three");
    check(mesh.database().active_record_count() == 1,
          "exactly the intended survivor remains active");

    std::size_t surviving_records = 0;
    for (const auto& record : mesh.vertices(Index{0})) {
        ++surviving_records;
        check(record.address == address_c,
              "the surviving range returns the expected database address");
        check(record.sigma == Sigma({Index{0}, Index{2}}),
              "the surviving vertex is converted to the new public numbering");
        check(point_equals(record.position, 3.0, 0.0),
              "the surviving vertex retains its position");
    }
    check(surviving_records == 1,
          "public node 0 exposes exactly one surviving vertex");
}

} // namespace

int main() {
    std::cout
        << "Running AbstractMesh tests\n"
        << "==========================\n"
        << "\nScope note\n"
        << "----------\n"
        << "Ordinary and extended VoronoiNodes semantics are tested by the\n"
        << "modular VoronoiNodes test suite. This executable tests only the\n"
        << "AbstractMesh access-mode bridge and AbstractMesh algorithms.\n"
        << "\nDatabase fixture note\n"
        << "---------------------\n"
        << "InstrumentedTestDatabase is a local test double for DatabaseT.\n"
        << "It makes AbstractMesh reads and erases observable; it neither\n"
        << "replaces nor tests the production HVDataBase implementation.\n";

    run_test(
        "AbstractMesh access-mode bridge",
        test_abstract_mesh_access_mode_bridge);
    run_test(
        "minimal vector-like AddressList template",
        test_vector_like_address_list_template);
    run_test(
        "store/contains signature conversion and buffer reuse",
        test_store_and_contains_with_recycled_buffer);
    run_test(
        "erase_vertex buffer ownership",
        test_erase_vertex_buffer_ownership);
    run_test(
        "forwarding predicate parameters",
        test_forwarding_predicates_accept_lvalues_and_temporaries);
    run_test(
        "primary/secondary/combined vertex ranges",
        test_primary_secondary_and_combined_ranges);
    run_test(
        "node deletion across primary and secondary lists",
        test_erase_nodes_uses_primary_and_secondary_lists);
    run_test(
        "unique global traversal through primary lists",
        test_erase_vertices_visits_primary_lists_only);
    run_test(
        "coordinated node and vertex filter",
        test_filter_receives_sorted_deleted_public_nodes);

    std::cout << "\nSummary\n"
              << "=======\n"
              << "Checks:   " << performed_checks << '\n'
              << "Failures: " << failed_checks << '\n';

    if (failed_checks == 0) {
        std::cout << "All AbstractMesh tests passed.\n";
        return 0;
    }

    std::cerr << "AbstractMesh tests failed.\n";
    return 1;
}
