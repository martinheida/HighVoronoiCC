
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/core/point.hpp>
#include <highvoronoi/parameters.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using size_t = std::size_t;

/**
 * @brief Assert that two vector-like objects contain the same values.
 */
template<class LeftVector, class RightVector>
void assert_equal_values(
    const LeftVector& left,
    const RightVector& right)
{
    assert(left.size() == right.size());

    for (size_t i = 0; i < left.size(); ++i) {
        assert(left[i] == right[i]);
    }
}

/**
 * @brief Test insertion, duplicate detection, reading, deletion, and reinsertion.
 */
template<class Database>
void test_push_and_read(Database& database)
{
    using Scalar = typename Database::Scalar;
    using Index = typename Database::Index;

    highvoronoi::StaticPoint<Scalar, 3> r;
    r[0] = Scalar{1.25};
    r[1] = Scalar{-2.5};
    r[2] = Scalar{3.75};

    std::vector<Index> sigma{
        Index{10},
        Index{20},
        Index{30}
    };

    const size_t address = database.push(r, sigma);

    assert(address == 1);
    assert(database.contains(sigma));
    assert(database.push(r, sigma) == 0);

    highvoronoi::StaticPoint<Scalar, 3> read_r;
    read_r[0] = Scalar{0};
    read_r[1] = Scalar{0};
    read_r[2] = Scalar{0};

    // read() resizes the dynamic signature vector to the stored length.
    std::vector<Index> read_sigma{Index{99}};

    database.read(address, read_r, read_sigma);

    assert_equal_values(read_r, r);
    assert_equal_values(read_sigma, sigma);

    assert(database.erase(address, sigma));
    assert(!database.contains(sigma));
    assert(!database.erase(address, sigma));

    read_r[0] = Scalar{91};
    read_r[1] = Scalar{92};
    read_r[2] = Scalar{93};

    read_sigma = {
        Index{81},
        Index{82},
        Index{83}
    };

    const auto unchanged_r = read_r;
    const auto unchanged_sigma = read_sigma;

    database.read(address, read_r, read_sigma);

    // Reading a deleted record does not modify either destination.
    assert_equal_values(read_r, unchanged_r);
    assert(read_sigma.empty());

    // erase() removes the signature from the hash structure, so it can be
    // inserted again at a new append-only address.
    const size_t second_address = database.push(r, sigma);

    assert(second_address != 0);
    assert(second_address != address);
    assert(database.contains(sigma));
}

/**
 * @brief Test facet insertion, in-place sorting, reading, and deletion.
 */
template<class Database>
void test_push_and_read_facet(Database& database)
{
    using Scalar = typename Database::Scalar;
    using Index = typename Database::Index;

    typename Database::VertexPoint r;
    r[0] = Scalar{4};
    r[1] = Scalar{5};
    r[2] = Scalar{6};

    typename Database::VertexPoint u;
    u[0] = Scalar{-1};
    u[1] = Scalar{-2};
    u[2] = Scalar{-3};

    std::vector<Index> sigma{
        Index{300},
        Index{100},
        Index{200}
    };

    const std::vector<Index> sorted_sigma{
        Index{100},
        Index{200},
        Index{300}
    };

    const size_t address = database.push_facet(r, sigma, u);

    assert(address != 0);

    // push_facet() sorts the supplied signature directly.
    assert_equal_values(sigma, sorted_sigma);
    assert(database.contains(sigma));
    assert(database.push_facet(r, sigma, u) == 0);

    typename Database::VertexPoint read_r;
    read_r[0] = Scalar{0};
    read_r[1] = Scalar{0};
    read_r[2] = Scalar{0};

    typename Database::VertexPoint read_u;
    read_u[0] = Scalar{0};
    read_u[1] = Scalar{0};
    read_u[2] = Scalar{0};

    std::vector<Index> read_sigma;

    database.read_facet(
        address,
        read_r,
        read_sigma,
        read_u);

    assert_equal_values(read_r, r);
    assert_equal_values(read_sigma, sorted_sigma);
    assert_equal_values(read_u, u);

    assert(database.erase(address, sigma));
    assert(!database.contains(sigma));

    read_r[0] = Scalar{71};
    read_r[1] = Scalar{72};
    read_r[2] = Scalar{73};

    read_sigma = {
        Index{61},
        Index{62},
        Index{63}
    };

    read_u[0] = Scalar{51};
    read_u[1] = Scalar{52};
    read_u[2] = Scalar{53};

    const auto unchanged_r = read_r;
    const auto unchanged_sigma = read_sigma;
    const auto unchanged_u = read_u;

    database.read_facet(
        address,
        read_r,
        read_sigma,
        read_u);

    // Reading a deleted facet does not modify any destination.
    assert_equal_values(read_r, unchanged_r);
    assert(read_sigma.empty());
    assert_equal_values(read_u, unchanged_u);

    const size_t second_address =
        database.push_facet(r, sigma, u);

    assert(second_address != 0);
    assert(second_address != address);
    assert(database.contains(sigma));
}

/**
 * @brief Test reading and writing arrays that cross several block boundaries.
 */
template<class Database>
void test_block_crossing(Database& database)
{
    using Scalar = typename Database::Scalar;
    using Index = typename Database::Index;

    std::vector<Index> sigma{
        Index{10},
        Index{20},
        Index{30},
        Index{40},
        Index{50},
        Index{60},
        Index{70},
        Index{80}
    };

    typename Database::VertexPoint r;
    r << Scalar{1}, Scalar{2}, Scalar{3};

    const size_t address =
        database.push(r, sigma);

    assert(address != 0);

    std::vector<Index> read_sigma;
    typename Database::VertexPoint read_r;

    database.read(
        address,
        read_r,
        read_sigma);

    assert_equal_values(read_sigma, sigma);
    assert_equal_values(read_r, r);
}

/**
 * @brief Test facet arrays that cross several block boundaries.
 */
template<class Database>
void test_facet_block_crossing(Database& database)
{
    using Scalar = typename Database::Scalar;
    using Index = typename Database::Index;

    std::vector<Index> sigma{
        Index{80},
        Index{20},
        Index{65},
        Index{40},
        Index{10},
        Index{70},
        Index{30},
        Index{50}
    };

    typename Database::VertexPoint r;
    r << Scalar{1}, Scalar{2}, Scalar{3};

    typename Database::VertexPoint u;
    u << Scalar{-1}, Scalar{-2}, Scalar{-3};

    const size_t address =
        database.push_facet(r, sigma, u);

    assert(address != 0);

    std::vector<Index> read_sigma;
    typename Database::VertexPoint read_r;
    typename Database::VertexPoint read_u;

    database.read_facet(
        address,
        read_r,
        read_sigma,
        read_u);

    assert_equal_values(read_sigma, sigma);
    assert_equal_values(read_r, r);
    assert_equal_values(read_u, u);
}
/**
 * @brief Run all database tests for one parameter configuration.
 */
template<class Params>
void run_database_configuration(
    std::string_view name,
    const Params& parameters)
{
    using Database = highvoronoi::HVDataBase<
        highvoronoi::ReadWriteLock,
        Params,
        3>;

    std::cout << "[HVDataBase] " << name << '\n';

    // Five UInt16 units deliberately force Index and Scalar ranges to cross
    // several DataUnit boundaries.
    Database database(5, parameters);

    assert(database.block_unit_length() == 5);
    assert(database.block_count() == 0);
    assert(database.reserved_unit_count() == 0);

    test_push_and_read(database);
    test_push_and_read_facet(database);
    test_block_crossing(database);
    test_facet_block_crossing(database);

    assert(database.block_count() > 1);
    assert(database.reserved_unit_count() > 0);

    const size_t expected_blocks =
        (database.reserved_unit_count() +
         database.block_unit_length() - 1) /
        database.block_unit_length();

    assert(database.block_count() == expected_blocks);
}

/**
 * @brief Test all supported QueueHash container and table combinations.
 */
void test_all_database_hash_configurations()
{
    using DirectModernParams = highvoronoi::DataBaseParams<
        float,
        std::uint32_t,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::DirectHash,
        highvoronoi::QueueTable>;

    using StaticModernParams = highvoronoi::DataBaseParams<
        float,
        std::uint32_t,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::StaticHash<4>,
        highvoronoi::QueueTable>;

    using DynamicModernParams = highvoronoi::DataBaseParams<
        float,
        std::uint32_t,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::DynamicHash<>,
        highvoronoi::QueueTable>;

    using DirectClassicParams = highvoronoi::DataBaseParams<
        float,
        std::uint32_t,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::DirectHash,
        highvoronoi::ClassicQueueTable>;

    using StaticClassicParams = highvoronoi::DataBaseParams<
        float,
        std::uint32_t,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::StaticHash<4>,
        highvoronoi::ClassicQueueTable>;

    using DynamicClassicParams = highvoronoi::DataBaseParams<
        float,
        std::uint32_t,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::DynamicHash<>,
        highvoronoi::ClassicQueueTable>;

    run_database_configuration(
        "DirectHash + QueueTable",
        DirectModernParams{highvoronoi::DirectHash{32}});

    run_database_configuration(
        "StaticHash<4> + QueueTable",
        StaticModernParams{highvoronoi::StaticHash<4>{8}});

    run_database_configuration(
        "DynamicHash + QueueTable",
        DynamicModernParams{highvoronoi::DynamicHash<>{1, 10, 8}});

    run_database_configuration(
        "DirectHash + ClassicQueueTable",
        DirectClassicParams{highvoronoi::DirectHash{32}});

    run_database_configuration(
        "StaticHash<4> + ClassicQueueTable",
        StaticClassicParams{highvoronoi::StaticHash<4>{8}});

    run_database_configuration(
        "DynamicHash + ClassicQueueTable",
        DynamicClassicParams{highvoronoi::DynamicHash<>{1, 10, 8}});
}

/**
 * @brief Exercise four-unit Scalar and Index values across block boundaries.
 */
void test_wide_scalar_and_index_types()
{
    using Params = highvoronoi::DataBaseParams<
        double,
        std::uint64_t,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::DirectHash,
        highvoronoi::QueueTable>;

    run_database_configuration(
        "DirectHash + QueueTable (double, uint64_t)",
        Params{highvoronoi::DirectHash{32}});
}

} // namespace

int main()
{
    test_all_database_hash_configurations();
    test_wide_scalar_and_index_types();

    std::cout << "ALL HVDATABASE TESTS PASSED\n";
    return 0;
}


