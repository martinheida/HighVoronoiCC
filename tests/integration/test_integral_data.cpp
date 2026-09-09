#include <highvoronoi/integrals.hpp>
#include <highvoronoi/storage/neighbour/neighbour_database.hpp>
#include <highvoronoi/core/detail/locks.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {
using Lock = highvoronoi::detail::EmptyLock;
using Index = std::uint32_t;
using Area = double;
using Integral = double;
using NeighbourDB = highvoronoi::detail::NeighbourDatabase<Lock, Index>;
using Data = highvoronoi::IntegralData<Lock, Index, Area, Integral, NeighbourDB>;
std::size_t checks = 0, failures = 0;
void check(bool ok, const char* text) {
    ++checks; if (!ok) { ++failures; std::cerr << "[FAIL] " << text << '\n'; }
    else std::cout << "[OK]   " << text << '\n';
}

template<class Exception, class F>
void check_throws(F&& f, const char* text) {
    bool ok = false;
    try { f(); } catch (const Exception&) { ok = true; } catch (...) {}
    check(ok, text);
}

struct FakeMeshEngine {
    using Index = std::uint32_t;
    explicit FakeMeshEngine(Index count) : count_(count) {}
    Index node_count() const noexcept { return count_; }
    Index count_;
};

class FakeIntegralEngine final
    : public highvoronoi::IntegralEngine<FakeMeshEngine, Area, Integral> {
public:
    using Base = highvoronoi::IntegralEngine<FakeMeshEngine, Area, Integral>;
    explicit FakeIntegralEngine(std::shared_ptr<FakeMeshEngine> mesh)
        : Base(std::move(mesh)) {}

    bool read_volume(Index node, Area& volume) const override {
        if (node >= node_count()) return false;
        volume = node == 0 ? 4.0 : 5.0;
        return true;
    }
    Address area_address_capacity() const noexcept override { return 2; }
    bool read_area(Address a, AreaBuffer& out) const override {
        if (a == 0) { out = {0.5, 0.75, 1.0}; return true; }
        if (a == 1) { out = {2.0}; return true; }
        out.clear(); return false;
    }
    std::optional<Address> area_address(Index node) const override {
        return node < 2 ? std::optional<Address>{node} : std::nullopt;
    }
    std::size_t integral_components() const noexcept override { return 2; }
    Address integral_address_capacity() const noexcept override { return 4; }
    bool read_integral(Address a, IntegralBuffer& out) const override {
        switch (a) {
            case 0: out={2.0,3.0}; return true;
            case 1: out={10,11,20,21,30,31}; return true;
            case 2: out={6.0,7.0}; return true;
            case 3: out={40,41}; return true;
            default: out.clear(); return false;
        }
    }
    std::optional<Address> bulk_integral_address(Index node) const override {
        if (node == 0) return Address{0};
        if (node == 1) return Address{2};
        return std::nullopt;
    }
    std::optional<Address> interface_integral_address(Index node) const override {
        if (node == 0) return Address{1};
        if (node == 1) return Address{3};
        return std::nullopt;
    }
};
}

int main() {
    NeighbourDB neighbours(16);
    const auto naddr0 = neighbours.push(std::vector<Index>{1,1,3});
    const auto naddr1 = neighbours.push(std::vector<Index>{0});

    // ------------------------------------------------------------------
    // Full stored layout.
    // ------------------------------------------------------------------
    Data full(neighbours, 2, 2);
    full.set_neighbour_address(Index{0}, naddr0);

    Data::CellData cell;
    cell.reserve(8, 2);
    check(full.prepare_cell(Index{0}, cell) &&
              cell.neighbours() == std::vector<Index>({1,1,3}),
          "prepare_cell loads the read-only neighbour snapshot including duplicates");

    cell.set_volume(4.0);
    cell.area() = {0.5, 0.75, 1.0};
    cell.bulk_integral() = {2.0, 3.0};
    cell.interface_integral() = {10,11,20,21,30,31};
    full.write_cell(Index{0}, cell);

    Data::CellData loaded;
    check(full.read_cell(Index{0}, loaded),
          "read_cell reports a complete fully enabled cell");
    check(loaded.volume() == 4.0,
          "volume is stored in its separate per-cell array");
    check(loaded.neighbours() == std::vector<Index>({1,1,3}),
          "neighbour order/multiplicity comes from the linked neighbour database");
    check(loaded.area() == std::vector<Area>({0.5,0.75,1.0}),
          "area ordinal k corresponds exactly to neighbour ordinal k");
    check(loaded.bulk_integral() == std::vector<Integral>({2,3}),
          "bulk integral supports vector-valued data");
    check(loaded.interface_integral() ==
              std::vector<Integral>({10,11,20,21,30,31}),
          "interface integral is neighbour-major and preserves duplicate-neighbour occurrences");
    check(full.validate_layout(Index{0}),
          "validate_layout enforces the neighbour/area/interface contract");

    // stale CellData must not be published against another neighbour snapshot.
    full.set_neighbour_address(Index{0}, naddr1);
    check_throws<std::logic_error>(
        [&] { full.write_cell(Index{0}, cell); },
        "write_cell rejects a CellData prepared from an older neighbour address");

    // ------------------------------------------------------------------
    // Optional storage: only neighbours + bulk integral.
    // ------------------------------------------------------------------
    highvoronoi::IntegralDataOptions sparse_options;
    sparse_options.volume = false;
    sparse_options.area = false;
    sparse_options.bulk_integral = true;
    sparse_options.interface_integral = false;

    Data sparse(neighbours, 3, 2, sparse_options);
    sparse.set_neighbour_address(Index{0}, naddr0);

    check(sparse.volume_slot_count() == 0 &&
              sparse.area_address_slot_count() == 0 &&
              sparse.interface_integral_address_slot_count() == 0 &&
              sparse.bulk_integral_address_slot_count() == 3,
          "disabled per-cell arrays are not allocated at all");
    check(sparse.area_database() == nullptr && sparse.integral_database() != nullptr,
          "disabled area storage does not even construct an area database");

    Data::CellData sparse_cell;
    check(sparse.prepare_cell(Index{0}, sparse_cell),
          "sparse CellData still receives its mandatory neighbour snapshot");
    sparse_cell.set_volume(999.0);                 // must be ignored
    sparse_cell.area() = {8.0,8.0,8.0};           // must be ignored
    sparse_cell.bulk_integral() = {7.0, 8.0};
    sparse_cell.interface_integral() = {1,2,3,4,5,6}; // must be ignored
    sparse.write_cell(Index{0}, sparse_cell);

    Data::CellData sparse_loaded;
    check(sparse.read_cell(Index{0}, sparse_loaded),
          "enabled bulk-only layout is complete after write");
    check(sparse_loaded.volume() == 0.0 && sparse_loaded.area().empty() &&
              sparse_loaded.interface_integral().empty(),
          "disabled CellData fields read as zero/empty regardless of caller input");
    check(sparse_loaded.bulk_integral() == std::vector<Integral>({7,8}),
          "enabled data remains writable when all unrelated fields are disabled");

    // Alignment checks are semantic and live at IntegralData level, not DB level.
    full.set_neighbour_address(Index{0}, naddr0);
    Data::CellData bad;
    check(full.prepare_cell(Index{0}, bad),
          "alignment-error fixture starts from the current neighbour snapshot");
    bad.set_volume(1.0);
    bad.area() = {1.0, 2.0}; // should be 3
    bad.bulk_integral() = {1.0,2.0};
    bad.interface_integral() = {1,2,3,4,5,6};
    check_throws<std::invalid_argument>(
        [&] { full.write_cell(Index{0}, bad); },
        "write_cell rejects area count that is not exactly neighbour count");

    bad.area() = {1.0,2.0,3.0};
    bad.interface_integral() = {1,2,3,4}; // should be 3*2=6
    check_throws<std::invalid_argument>(
        [&] { full.write_cell(Index{0}, bad); },
        "write_cell rejects interface layout not aligned with neighbour ordinals");

    // ------------------------------------------------------------------
    // Retarget one historical neighbour snapshot to a new one.
    // Duplicate neighbours are matched occurrence-by-occurrence.
    // ------------------------------------------------------------------
    const auto naddr2 = neighbours.push(std::vector<Index>{1,1,2,3});
    Data reconcile(neighbours, 1, 2);
    reconcile.set_neighbour_address(Index{0}, naddr0);

    Data::CellData old_cell;
    check(reconcile.prepare_cell(Index{0}, old_cell),
          "retarget fixture loads the old neighbour snapshot");
    old_cell.set_volume(4.0);
    old_cell.area() = {0.5, 0.75, 1.0};
    old_cell.bulk_integral() = {2.0, 3.0};
    old_cell.interface_integral() = {10,11,20,21,30,31};
    reconcile.write_cell(Index{0}, old_cell);

    Data::CellData retargeted;
    check(reconcile.read_cell(Index{0}, retargeted),
          "retarget fixture has one complete old integral state");
    std::vector<std::uint8_t> matched;
    std::vector<Area> area_scratch;
    std::vector<Integral> interface_scratch;
    reconcile.retarget_cell(
        retargeted,
        naddr2,
        std::vector<Index>{1,1,2,3},
        matched,
        area_scratch,
        interface_scratch);

    check(retargeted.source_neighbour_address() == naddr0 &&
              retargeted.neighbour_address() == naddr2,
          "retarget keeps the old source address and records the new target address");
    check(matched == std::vector<std::uint8_t>({1,1,0,1}),
          "duplicate neighbour occurrences are matched one-to-one in ordinal order");
    check(retargeted.area() == std::vector<Area>({0.5,0.75,0.0,1.0}),
          "retarget preserves surviving area entries and zero-initializes a new interface");
    check(retargeted.interface_integral() ==
              std::vector<Integral>({10,11,20,21,0,0,30,31}),
          "retarget preserves surviving interface-integral blocks with identical ordinal alignment");

    retargeted.set_volume(9.0);
    retargeted.bulk_integral() = {8.0, 9.0};
    reconcile.write_cell(Index{0}, retargeted);
    Data::CellData committed_retarget;
    check(reconcile.read_cell(Index{0}, committed_retarget) &&
              reconcile.neighbour_address(Index{0}) == naddr2 &&
              committed_retarget.neighbours() == std::vector<Index>({1,1,2,3}),
          "write_cell atomically publishes the new neighbour snapshot with its aligned data");

    // ------------------------------------------------------------------
    // One coherent engine, automatically adapted to both hybrid DBs.
    // ------------------------------------------------------------------
    auto mesh = std::make_shared<FakeMeshEngine>(2);
    auto engine = std::make_shared<FakeIntegralEngine>(mesh);
    const auto registration = full.register_engine(engine);
    full.publish_engine_cell(
        Index{1}, *engine, registration, Index{0}, naddr0);

    Data::CellData engine_cell;
    check(full.read_cell(Index{1}, engine_cell),
          "one coherent IntegralEngine publishes through both hybrid databases");
    check(engine_cell.volume() == 4.0 &&
              engine_cell.area() == std::vector<Area>({0.5,0.75,1.0}) &&
              engine_cell.bulk_integral() == std::vector<Integral>({2,3}) &&
              engine_cell.interface_integral() ==
                  std::vector<Integral>({10,11,20,21,30,31}),
          "engine-backed CellData has the same external layout as stored CellData");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
