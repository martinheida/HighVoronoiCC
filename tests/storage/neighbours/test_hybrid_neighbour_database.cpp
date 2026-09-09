#include <highvoronoi/storage/neighbour/hybrid_neighbour_database.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using Index = std::uint32_t;
using Sigma = std::vector<Index>;
using Database = highvoronoi::detail::HybridNeighbourDatabase<
    highvoronoi::detail::EmptyLock,
    Index>;

std::size_t checks = 0;
std::size_t failures = 0;

void check(bool ok, std::string_view description) {
    ++checks;
    if (ok) {
        std::cout << "    [OK]   " << description << '\n';
    } else {
        ++failures;
        std::cerr << "    [FAIL] " << description << '\n';
    }
}

class FakeEngine final {
public:
    using Index = ::Index;
    using Sigma = ::Sigma;

    [[nodiscard]] bool provides_neighbours() const noexcept { return true; }
    [[nodiscard]] Index node_count() const noexcept { return Index{3}; }

    [[nodiscard]] bool read_neighbours(Index cell, Sigma& neighbours) const {
        neighbours.clear();
        const Index boundary = (std::numeric_limits<Index>::max)() - Index{1};
        switch (cell) {
        case Index{0}:
            neighbours = Sigma{Index{1}, Index{2}};
            return true;
        case Index{1}:
            neighbours = Sigma{Index{0}, Index{0}, Index{2}, boundary};
            return true;
        case Index{2}:
            neighbours = Sigma{Index{0}, Index{1}};
            return true;
        default:
            return false;
        }
    }
};

void test_stored_engine_stored_routing() {
    std::cout << "\n[TEST] HybridNeighbourDatabase stored -> engine -> stored\n";

    Database db(64);
    const auto a = db.push(Sigma{Index{40}, Index{41}});
    check(a == 1, "first stored record gets logical address 1");

    auto engine = std::make_shared<FakeEngine>();
    const auto reg = db.register_engine(engine, Index{10});
    check(reg.first_address == 2, "engine range starts after first stored record");
    check(reg.address_capacity == 3, "engine reserves one address per local cell");
    check(reg.end_address() == 5, "engine range is [2,5)");

    const auto b = db.push(Sigma{Index{50}, Index{51}, Index{51}});
    check(b == 5, "later stored record continues after virtual engine range");
    check(db.next_logical_address() == 6, "next logical address follows all segments");

    Sigma result{Index{999}};
    check(!db.read(0, result) && result.empty(), "address zero means no record");

    check(db.read(a, result) && result == Sigma({Index{40}, Index{41}}),
          "stored record before engine reads unchanged");

    check(db.read(reg.first_address + 0, result) &&
              result == Sigma({Index{11}, Index{12}}),
          "engine local neighbours are translated by node offset");

    const Index boundary = (std::numeric_limits<Index>::max)() - Index{1};
    check(db.read(reg.first_address + 1, result) &&
              result == Sigma({Index{10}, Index{10}, Index{12}, boundary}),
          "multiplicity survives and high-end boundary encoding is unchanged");

    check(db.read(b, result) &&
              result == Sigma({Index{50}, Index{51}, Index{51}}),
          "stored record after engine reads unchanged with duplicates");
}

void test_registration_requires_authority() {
    std::cout << "\n[TEST] Registration requires authoritative neighbour support\n";

    class NoNeighbours final {
    public:
        using Index = ::Index;
        using Sigma = ::Sigma;
        [[nodiscard]] bool provides_neighbours() const noexcept { return false; }
        [[nodiscard]] Index node_count() const noexcept { return Index{2}; }
        [[nodiscard]] bool read_neighbours(Index, Sigma&) const { return false; }
    };

    Database db;
    bool threw = false;
    try {
        (void)db.register_engine(std::make_shared<NoNeighbours>(), Index{0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "engine without neighbour authority cannot reserve virtual records");
}

} // namespace

int main() {
    test_stored_engine_stored_routing();
    test_registration_requires_authority();

    std::cout << "\nchecks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
