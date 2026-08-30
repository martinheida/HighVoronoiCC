#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/detail/hybrid_database.hpp>
#include <highvoronoi/geometry/cuboid_mesh_engine.hpp>
#include <highvoronoi/parameters.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
using Params = highvoronoi::DataBaseParams<Scalar, Index>;
using StoredDatabase = highvoronoi::HVDataBase<highvoronoi::EmptyLock, Params>;
using CuboidEngine = highvoronoi::CuboidMeshEngine<Scalar, Scalar, Index, 3>;
using Database = highvoronoi::HybridDataBase<highvoronoi::EmptyLock, Params>;
using Point = CuboidEngine::VertexPoint;
using CountPoint = CuboidEngine::CountPoint;
using Sigma = std::vector<Index>;

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

Point point(Scalar x, Scalar y, Scalar z) {
    Point p;
    p << x, y, z;
    return p;
}

CountPoint counts(Index x, Index y, Index z) {
    CountPoint p;
    p << x, y, z;
    return p;
}

void test_stored_engine_stored_address_space() {
    std::cout << "\n[TEST] HybridDataBase stored -> engine -> stored routing\n";

    static_assert(std::is_same_v<typename Database::StoreDataBaseType, StoredDatabase>,
                  "default HybridDataBase backend must be HVDataBase<Lock, Params>.");

    Database database(
        4096,
        Params{highvoronoi::DirectHash{4096}});

    const Sigma sigma_a{Index{100}, Index{101}, Index{102}, Index{103}};
    const Sigma sigma_b{Index{110}, Index{111}, Index{112}, Index{113}};
    const auto address_a = database.push(point(-10, 0, 0), sigma_a);
    const auto address_b = database.push(point(-20, 0, 0), sigma_b);
    check(address_a == 1 && address_b == 2,
          "stored records before engine occupy logical addresses 1 and 2");

    auto concrete = std::make_shared<CuboidEngine>(
        point(0, 0, 0),
        point(2, 2, 2),
        counts(3, 3, 2));
    const auto registration = database.register_engine(concrete, Index{20});
    check(registration.first_address == 3,
          "engine range begins immediately after earlier stored records");
    check(registration.address_capacity == 4,
          "3x3x2 grid reserves four finite engine-vertex addresses");

    const Sigma sigma_c{Index{120}, Index{121}, Index{122}, Index{123}};
    const Sigma sigma_d{Index{130}, Index{131}, Index{132}, Index{133}};
    const auto address_c = database.push(point(20, 0, 0), sigma_c);
    const auto address_d = database.push(point(30, 0, 0), sigma_d);
    check(address_c == 7 && address_d == 8,
          "stored records after engine continue after its full reserved range");

    Point loaded;
    Sigma loaded_sigma;

    database.read(address_a, loaded, loaded_sigma);
    check(loaded_sigma == sigma_a && loaded[0] == -10,
          "first stored logical record still reads correctly");

    database.read(address_d, loaded, loaded_sigma);
    check(loaded_sigma == sigma_d && loaded[0] == 30,
          "later stored logical record still reads correctly");

    // Engine local vertex 0 is at logical first_address and all local ordinary
    // node indices are shifted by node_offset=20 in the HybridDataBase view.
    database.read(registration.first_address, loaded, loaded_sigma);
    const Sigma expected_engine_sigma{
        Index{20}, Index{21}, Index{23}, Index{24},
        Index{29}, Index{30}, Index{32}, Index{33}};
    check(loaded_sigma == expected_engine_sigma,
          "engine vertex sigma is translated to global node indices on read");
    check(loaded[0] == 1 && loaded[1] == 1 && loaded[2] == 1,
          "engine vertex position is read through the same database API");

    check(database.contains(expected_engine_sigma),
          "registered engine signature participates in the global hash");
    check(database.push(point(99, 99, 99), expected_engine_sigma) == 0,
          "stored duplicate insertion sees an existing computed-engine vertex");

    check(database.erase(registration.first_address, expected_engine_sigma),
          "engine vertex deletion is routed through HybridDataBase");
    loaded_sigma = Sigma{Index{999}};
    database.read(registration.first_address, loaded, loaded_sigma);
    check(loaded_sigma.empty(),
          "deleted engine vertex reads as a tombstone through HybridDataBase");
    check(!database.contains(expected_engine_sigma),
          "deleted engine signature is removed from the shared hash");
}

void test_registration_collision_is_transactional() {
    std::cout << "\n[TEST] HybridDataBase engine registration collision rollback\n";

    Database database(
        4096,
        Params{highvoronoi::DirectHash{4096}});

    // This equals local cuboid vertex-0 sigma after node_offset 20.
    const Sigma collision_sigma{
        Index{20}, Index{21}, Index{23}, Index{24},
        Index{29}, Index{30}, Index{32}, Index{33}};
    const auto stored_address = database.push(point(-1, -1, -1), collision_sigma);
    check(stored_address == 1,
          "conflicting stored vertex is inserted before engine registration");

    auto concrete = std::make_shared<CuboidEngine>(
        point(0, 0, 0),
        point(2, 2, 2),
        counts(3, 3, 2));
    bool threw = false;
    try {
        (void)database.register_engine(concrete, Index{20});
    } catch (const std::logic_error&) {
        threw = true;
    }

    check(threw,
          "engine registration rejects a signature already present in stored data");
    check(database.next_logical_address() == 2,
          "failed registration consumes no logical address range");
    check(database.contains(collision_sigma),
          "failed registration preserves the pre-existing stored hash entry");
}

} // namespace

int main() {
    test_stored_engine_stored_address_space();
    test_registration_collision_is_transactional();
    std::cout << "\nperformed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    return failed_checks == 0 ? 0 : 1;
}
