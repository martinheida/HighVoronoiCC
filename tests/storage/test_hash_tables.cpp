
#include "highvoronoi/storage/hash/queue_hash_table.hpp"
#include "highvoronoi/storage/hash/edge_hash_table.hpp"
#include "highvoronoi/storage/hash/hash_table_containers.hpp"
#include "highvoronoi/storage/hash/hash_types.hpp"
#include "highvoronoi/parameters.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace highvoronoi::detail;

namespace {

void check(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
    std::cout << "  [OK] " << message << '\n';
}


template<class Actual, class Expected>
void check_same_type(const std::string& message)
{
    static_assert(
        std::is_same_v<Actual, Expected>,
        "The parameter adapter produced an unexpected type"
    );

    check(std::is_same_v<Actual, Expected>, message);
}

template<class View>
void print_view(const View& view)
{
    std::cout << "  [";
    for (std::size_t i = 0; i < view.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << view[i];
    }
    std::cout << "]\n";
}

template<class Hash>
void test_hash64(
    const std::string& name,
    const UInt64View<std::array<std::int32_t, 3>>& key1_i32,
    const UInt64View<std::array<std::uint64_t, 3>>& key1_u64,
    const UInt64View<std::array<std::uint64_t, 3>>& key2)
{
    const Hash hash{};

    const std::uint64_t h1_i32 = hash(key1_i32);
    const std::uint64_t h1_u64 = hash(key1_u64);
    const std::uint64_t h2 = hash(key2);

    std::cout << "\n[" << name << "]\n";
    std::cout << "  hash({1,2,3}) = 0x"
              << std::hex << h1_u64 << std::dec << '\n';
    std::cout << "  hash({2,3,4}) = 0x"
              << std::hex << h2 << std::dec << '\n';

    check(h1_i32 == h1_u64,
          name + ": Int32 and UInt64 views produce the same hash");
    check(h1_u64 != h2,
          name + ": {1,2,3} and {2,3,4} produce different hashes");
}

template<class Generator, class Key1, class Key2>
void test_generator(
    const std::string& name,
    const Key1& key1,
    const Key2& key2)
{
    std::cout << "\n[" << name << "]\n";

    Generator empty_generator;
    (void)empty_generator;
    check(true, name + ": default construction");

    const Generator g1(key1);
    const Generator g2(key2);

    check(g1 != g2, name + ": different keys produce different fingerprints");

    constexpr std::size_t capacity = 16;
    constexpr std::size_t mask = capacity - 1;

    std::array<bool, capacity> seen{};
    bool all_inside = true;
    bool all_unique = true;

    std::cout << "  probe sequence:";
    for (std::size_t i = 0; i < capacity; ++i) {
        const std::size_t idx = g1.index(mask, i);
        std::cout << ' ' << idx;

        if (idx >= capacity) {
            all_inside = false;
            continue;
        }
        if (seen[idx]) {
            all_unique = false;
        }
        seen[idx] = true;
    }
    std::cout << '\n';

    check(all_inside, name + ": all probe indices are inside the table");
    check(all_unique, name + ": probe sequence contains no duplicate index");

    bool all_seen = true;
    for (bool value : seen) {
        all_seen = all_seen && value;
    }
    check(all_seen, name + ": probe sequence visits every slot exactly once");
}

void test_uint64_view()
{
    std::cout << "\n============================================================\n";
    std::cout << "UInt64View\n";
    std::cout << "============================================================\n";

    const std::array<std::int32_t, 3> key_i32{1, 2, 3};
    const std::array<std::uint64_t, 3> key_u64{1, 2, 3};

    const UInt64View view_i32(key_i32);
    const UInt64View view_u64(key_u64);

    std::cout << "Int32  view:";
    print_view(view_i32);

    std::cout << "UInt64 view:";
    print_view(view_u64);

    check(view_i32.size() == 3, "Int32 view has size 3");
    check(view_u64.size() == 3, "UInt64 view has size 3");

    bool values_ok = true;
    for (std::size_t i = 0; i < 3; ++i) {
        values_ok =
            values_ok &&
            view_i32[i] == i + 1 &&
            view_u64[i] == i + 1 &&
            view_i32[i] == view_u64[i];
    }
    check(values_ok, "both views return {1,2,3} through operator[]");
}

void test_hash_functions()
{
    std::cout << "\n============================================================\n";
    std::cout << "Hash functions\n";
    std::cout << "============================================================\n";

    const std::array<std::int32_t, 3> key1_i32{1, 2, 3};
    const std::array<std::uint64_t, 3> key1_u64{1, 2, 3};
    const std::array<std::uint64_t, 3> key2_u64{2, 3, 4};

    const UInt64View view1_i32(key1_i32);
    const UInt64View view1_u64(key1_u64);
    const UInt64View view2_u64(key2_u64);

    test_hash64<FNV1a64<>>(
        "FNV1a64", view1_i32, view1_u64, view2_u64);

    test_hash64<XXHash64<>>(
        "XXHash64", view1_i32, view1_u64, view2_u64);

    test_hash64<MurmurHash64<>>(
        "MurmurHash64", view1_i32, view1_u64, view2_u64);

    test_hash64<SipHash64<>>(
        "SipHash64", view1_i32, view1_u64, view2_u64);

    std::cout << "\n[MurmurHash_x64_128]\n";

    const MurmurHash_x64_128<> murmur128{};
    const auto h1_i32 = murmur128(view1_i32);
    const auto h1_u64 = murmur128(view1_u64);
    const auto h2 = murmur128(view2_u64);

    std::cout << "  hash({1,2,3}) = {0x"
              << std::hex << h1_u64.first
              << ", 0x" << h1_u64.second
              << "}\n";
    std::cout << "  hash({2,3,4}) = {0x"
              << h2.first
              << ", 0x" << h2.second
              << "}" << std::dec << '\n';

    check(h1_i32 == h1_u64,
          "MurmurHash_x64_128: Int32 and UInt64 views produce the same hash");
    check(h1_u64 != h2,
          "MurmurHash_x64_128: different keys produce different hashes");
}

void test_hash_generators()
{
    std::cout << "\n============================================================\n";
    std::cout << "Hash generators\n";
    std::cout << "============================================================\n";

    const std::array<std::uint64_t, 3> key1{1, 2, 3};
    const std::array<std::uint64_t, 3> key2{2, 3, 4};

    using Multi64 = UInt64HashGenerator<
        XXHash64<1>,
        MurmurHash64<2>,
        FNV1a64<>,
        SipHash64<3, 5>
    >;

    using Murmur128 = Murmur128HashGenerator<7>;

    using Extended = ExtendedHashGenerator<
        Murmur128,
        XXHash64<11>,
        FNV1a64<>
    >;

    using LegacyFNV = FNV64_128HashGenerator;

    test_generator<Multi64>(
        "UInt64HashGenerator", key1, key2);

    test_generator<Murmur128>(
        "Murmur128HashGenerator", key1, key2);

    test_generator<Extended>(
        "ExtendedHashGenerator", key1, key2);

    test_generator<LegacyFNV>(
        "FNV64_128HashGenerator", key1, key2);

    std::cout << "\n[ExtendedHashGenerator delegation]\n";

    const Murmur128 base(key1);
    const Extended extended(key1);

    bool same_indices = true;
    for (std::size_t i = 0; i < 16; ++i) {
        same_indices =
            same_indices &&
            base.index(15, i) == extended.index(15, i);
    }
    check(
        same_indices,
        "ExtendedHashGenerator uses the hosted generator's probe sequence");
}

void test_queue_hash_table()
{
    std::cout << "\n============================================================\n";
    std::cout << "QueueHashTable\n";
    std::cout << "============================================================\n";

    using Generator = UInt64HashGenerator<
        XXHash64<1>,
        MurmurHash64<2>,
        FNV1a64<>
    >;

    QueueHashTable<EmptyLock, Generator> queue(4);

    const std::size_t initial_capacity = queue.capacity();
    std::cout << "  initial capacity = " << initial_capacity << '\n';
    check(initial_capacity == 4, "queue starts with capacity 4");

    bool all_inserted_as_new = true;
    bool all_found = true;
    bool all_read_queries_found = true;

    for (std::uint64_t n = 0; n < 12; ++n) {
        const std::array<std::uint64_t, 3> key{n, n + 1, n + 2};

        all_inserted_as_new =
            all_inserted_as_new && !queue.pushqueue(key);
        all_found =
            all_found && queue.contains(key);
        all_read_queries_found =
            all_read_queries_found && queue.pushqueue(key, false);
    }

    check(all_inserted_as_new, "12 different queue keys are inserted as new");
    check(all_found, "all 12 inserted queue keys can be found");
    check(all_read_queries_found,
          "read-only pushqueue finds all 12 existing keys");

    const std::size_t grown_capacity = queue.capacity();
    std::cout << "  capacity after insertions = "
              << grown_capacity << '\n';

    check(
        grown_capacity > initial_capacity,
        "queue internal storage was extended");

    const std::array<std::uint64_t, 3> deleted_key{4, 5, 6};

    check(queue.contains(deleted_key),
          "key selected for deletion exists");

    check(queue.erase(deleted_key),
          "erase returns true for existing key");

    check(!queue.contains(deleted_key),
          "deleted key is no longer found");

    check(!queue.erase(deleted_key),
          "erase returns false when key is already absent");

    check(!queue.pushqueue(deleted_key),
          "deleted key can be inserted again");

    check(queue.contains(deleted_key),
          "reinserted key can be found");

    queue.clear();
    check(!queue.contains(deleted_key),
          "clear removes queue entries");

    std::cout << "  final capacity = "
              << queue.capacity() << '\n';
}

void test_edge_hash_table()
{
    std::cout << "\n============================================================\n";
    std::cout << "EdgeHashTable\n";
    std::cout << "============================================================\n";

    using Generator = ExtendedHashGenerator<
        Murmur128HashGenerator<7>,
        XXHash64<11>,
        SipHash64<13, 17>
    >;

    EdgeHashTable<EmptyLock, Generator> edges(4);

    const std::size_t initial_capacity = edges.capacity();
    std::cout << "  initial capacity = " << initial_capacity << '\n';
    check(initial_capacity == 4, "edge table starts with capacity 4");

    bool all_edges_inserted_as_new = true;

    for (std::uint64_t n = 0; n < 12; ++n) {
        const std::array<std::uint64_t, 3> key{
            100 + n,
            200 + n,
            300 + n
        };

        all_edges_inserted_as_new =
            all_edges_inserted_as_new && !edges.pushedge(key, n);
    }

    check(all_edges_inserted_as_new,
          "12 different edge keys are inserted as new");

    const std::size_t grown_capacity = edges.capacity();
    std::cout << "  capacity after insertions = "
              << grown_capacity << '\n';

    check(
        grown_capacity > initial_capacity,
        "edge table internal storage was extended");

    const std::array<std::uint64_t, 3> edge_key{999, 1000, 1001};

    check(!edges.pushedge(edge_key, 10),
          "first cell of an edge is inserted");

    check(!edges.pushedge(edge_key, 11),
          "second cell of the same edge is inserted");

    check(edges.pushedge(edge_key, 12),
          "third occurrence reports that the edge already has two cells");

    edges.clear();

    check(!edges.pushedge(edge_key, 20),
          "edge table accepts the key again after clear");

    std::cout << "  final capacity = "
              << edges.capacity() << '\n';

    std::cout << "  Note: EdgeHashTable currently has no contains() or erase().\n";
}


void test_parameter_types()
{
    std::cout << "\n============================================================\n";
    std::cout << "Parameter types and compile-time adapters\n";
    std::cout << "============================================================\n";

    using Generator = highvoronoi::UInt64HashGenerator<
        highvoronoi::XXHash64<1>,
        highvoronoi::MurmurHash64<2>,
        highvoronoi::FNV1a64<>
    >;

    // ------------------------------------------------------------------------
    // Public threading aliases
    // ------------------------------------------------------------------------

    check_same_type<
        typename highvoronoi::SingleThread::RWLock,
        highvoronoi::EmptyLock
    >("SingleThread selects EmptyLock");

    check_same_type<
        typename highvoronoi::MultiThread::RWLock,
        highvoronoi::ReadWriteLock
    >("MultiThread selects ReadWriteLock");

    const highvoronoi::SingleThread single_thread;
    const highvoronoi::MultiThread multi_thread(4);

    check(single_thread.thread_count() == 1,
          "SingleThread reports one thread");
    check(multi_thread.thread_count() == 4,
          "MultiThread stores the requested thread count");

    // ------------------------------------------------------------------------
    // Default database parameters
    // ------------------------------------------------------------------------

    using DefaultDatabaseParams = highvoronoi::DataBaseParams<>;

    check_same_type<
        typename DefaultDatabaseParams::Scalar,
        double
    >("DataBaseParams default Scalar is double");

    check_same_type<
        typename DefaultDatabaseParams::Index,
        std::uint32_t
    >("DataBaseParams default Index is uint32_t");

    check_same_type<
        typename DefaultDatabaseParams::HashGenerator,
        highvoronoi::FNV64_128HashGenerator
    >("DataBaseParams default hash generator is FNV64_128HashGenerator");

    check_same_type<
        typename DefaultDatabaseParams::ContainerMode,
        highvoronoi::DirectHash
    >("DataBaseParams default container mode is DirectHash");

    using DefaultQueueFromParams =
        highvoronoi::detail::QueueHashFromParams_t<
            highvoronoi::EmptyLock,
            DefaultDatabaseParams
        >;

    using ExpectedDefaultQueue =
        highvoronoi::detail::QueueHashTable_2<
            highvoronoi::EmptyLock,
            highvoronoi::FNV64_128HashGenerator
        >;

    check_same_type<
        DefaultQueueFromParams,
        ExpectedDefaultQueue
    >("default DataBaseParams produce the default direct QueueHashTable_2");

    // ------------------------------------------------------------------------
    // Explicit database parameters: direct, static and dynamic
    // ------------------------------------------------------------------------

    using DirectDatabaseParams = highvoronoi::DataBaseParams<
        float,
        std::uint32_t,
        Generator,
        highvoronoi::DirectHash,
        highvoronoi::QueueTable
    >;

    check_same_type<
        typename DirectDatabaseParams::Scalar,
        float
    >("explicit DataBaseParams preserve Scalar");

    check_same_type<
        typename DirectDatabaseParams::Index,
        std::uint32_t
    >("explicit DataBaseParams preserve Index");

    check_same_type<
        typename DirectDatabaseParams::HashGenerator,
        Generator
    >("explicit DataBaseParams preserve HashGenerator");

    using DirectQueueFromParams =
        highvoronoi::detail::QueueHashFromParams_t<
            highvoronoi::EmptyLock,
            DirectDatabaseParams
        >;

    using ExpectedDirectQueue =
        highvoronoi::detail::QueueHashTable_2<
            highvoronoi::EmptyLock,
            Generator
        >;

    check_same_type<
        DirectQueueFromParams,
        ExpectedDirectQueue
    >("DirectHash produces one direct QueueHashTable_2");

    using StaticDatabaseParams = highvoronoi::DataBaseParams<
        double,
        std::uint64_t,
        Generator,
        highvoronoi::StaticHash<4>,
        highvoronoi::QueueTable
    >;

    using StaticQueueFromParams =
        highvoronoi::detail::QueueHashFromParams_t<
            highvoronoi::ReadWriteLock,
            StaticDatabaseParams
        >;

    using ExpectedStaticQueue =
        highvoronoi::detail::StaticQueueHashContainer<
            highvoronoi::detail::QueueHashTable_2<
                highvoronoi::ReadWriteLock,
                Generator
            >,
            4
        >;

    check_same_type<
        StaticQueueFromParams,
        ExpectedStaticQueue
    >("StaticHash<4> produces a four-table static queue container");

    using DynamicDatabaseParams = highvoronoi::DataBaseParams<
        double,
        std::uint64_t,
        Generator,
        highvoronoi::DynamicHash<>,
        highvoronoi::QueueTable
    >;

    using DynamicQueueFromParams =
        highvoronoi::detail::QueueHashFromParams_t<
            highvoronoi::ReadWriteLock,
            DynamicDatabaseParams
        >;

    using ExpectedDynamicQueue =
        highvoronoi::detail::DynamicQueueHashContainer<
            highvoronoi::detail::QueueHashTable_2<
                highvoronoi::ReadWriteLock,
                Generator
            >,
            highvoronoi::ReadWriteLock
        >;

    check_same_type<
        DynamicQueueFromParams,
        ExpectedDynamicQueue
    >("DynamicHash<> produces a dynamic queue container with the selected lock");

    // ------------------------------------------------------------------------
    // Edge-buffer parameters: direct, static and dynamic
    // ------------------------------------------------------------------------

    using DefaultEdgeParams = highvoronoi::EdgeBufferParams<>;

    using DefaultEdgeFromParams =
        highvoronoi::detail::EdgeHashFromParams_t<
            highvoronoi::EmptyLock,
            DefaultEdgeParams
        >;

    using ExpectedDefaultEdge =
        highvoronoi::detail::EdgeHashTable<
            highvoronoi::EmptyLock,
            highvoronoi::FNV64_128HashGenerator
        >;

    check_same_type<
        DefaultEdgeFromParams,
        ExpectedDefaultEdge
    >("default EdgeBufferParams produce the default direct EdgeHashTable");

    using DirectEdgeParams = highvoronoi::EdgeBufferParams<
        Generator,
        highvoronoi::DirectHash,
        highvoronoi::EdgeTable
    >;

    using DirectEdgeFromParams =
        highvoronoi::detail::EdgeHashFromParams_t<
            highvoronoi::EmptyLock,
            DirectEdgeParams
        >;

    using ExpectedDirectEdge =
        highvoronoi::detail::EdgeHashTable<
            highvoronoi::EmptyLock,
            Generator
        >;

    check_same_type<
        DirectEdgeFromParams,
        ExpectedDirectEdge
    >("DirectHash produces one direct EdgeHashTable");

    using StaticEdgeParams = highvoronoi::EdgeBufferParams<
        Generator,
        highvoronoi::StaticHash<4>,
        highvoronoi::EdgeTable
    >;

    using StaticEdgeFromParams =
        highvoronoi::detail::EdgeHashFromParams_t<
            highvoronoi::ReadWriteLock,
            StaticEdgeParams
        >;

    using ExpectedStaticEdge =
        highvoronoi::detail::StaticEdgeHashContainer<
            highvoronoi::detail::EdgeHashTable<
                highvoronoi::ReadWriteLock,
                Generator
            >,
            4
        >;

    check_same_type<
        StaticEdgeFromParams,
        ExpectedStaticEdge
    >("StaticHash<4> produces a four-table static edge container");

    using DynamicEdgeParams = highvoronoi::EdgeBufferParams<
        Generator,
        highvoronoi::DynamicHash<>,
        highvoronoi::EdgeTable
    >;

    using DynamicEdgeFromParams =
        highvoronoi::detail::EdgeHashFromParams_t<
            highvoronoi::ReadWriteLock,
            DynamicEdgeParams
        >;

    using ExpectedDynamicEdge =
        highvoronoi::detail::DynamicEdgeHashContainer<
            highvoronoi::detail::EdgeHashTable<
                highvoronoi::ReadWriteLock,
                Generator
            >,
            highvoronoi::ReadWriteLock
        >;

    check_same_type<
        DynamicEdgeFromParams,
        ExpectedDynamicEdge
    >("DynamicHash<> produces a dynamic edge container with the selected lock");
}


void test_static_queue_hash_container()
{
    std::cout << "\n============================================================\n";
    std::cout << "StaticQueueHashContainer through DataBaseParams\n";
    std::cout << "============================================================\n";

    using Generator = highvoronoi::UInt64HashGenerator<
        highvoronoi::XXHash64<1>,
        highvoronoi::MurmurHash64<2>,
        highvoronoi::FNV1a64<>
    >;

    using Params = highvoronoi::DataBaseParams<
        double,
        std::uint64_t,
        Generator,
        highvoronoi::StaticHash<4>
    >;

    using Container =
        highvoronoi::detail::QueueHashFromParams_t<
            highvoronoi::EmptyLock,
            Params
        >;

    Container queues(2);

    check(queues.table_count() == 4,
          "static queue container contains exactly four hash tables");

    std::array<std::array<std::uint64_t, 3>, 8> keys{};

    bool inserted = true;
    for (std::uint64_t i = 0; i < keys.size(); ++i) {
        keys[i] = {i, 100 + i, 200 + i};
        inserted = inserted && !queues.pushqueue(keys[i]);
    }
    check(inserted,
          "static queue container inserts keys routed by key[0] % 4");

    bool found = true;
    for (const auto& key : keys) {
        found = found && queues.contains(key);
    }
    check(found,
          "static queue container finds all inserted keys");

    bool reported_as_present = true;
    for (const auto& key : keys) {
        reported_as_present =
            reported_as_present && queues.pushqueue(key, false);
    }
    check(reported_as_present,
          "static queue container read queries report all existing keys");

    check(queues.erase(keys[5]),
          "static queue container erases an existing key");
    check(!queues.contains(keys[5]),
          "erased static-container key is no longer found");
    check(!queues.erase(keys[5]),
          "erasing the same static-container key twice returns false");
    check(!queues.pushqueue(keys[5]),
          "erased static-container key can be inserted again");

    queues.clear();

    bool all_cleared = true;
    for (const auto& key : keys) {
        all_cleared = all_cleared && !queues.contains(key);
    }
    check(all_cleared,
          "clear removes entries from every static queue table");
}


void test_dynamic_queue_hash_container()
{
    std::cout << "\n============================================================\n";
    std::cout << "DynamicQueueHashContainer through DataBaseParams\n";
    std::cout << "============================================================\n";

    using Generator = highvoronoi::Murmur128HashGenerator<7>;

    using Params = highvoronoi::DataBaseParams<
        double,
        std::uint64_t,
        Generator,
        highvoronoi::DynamicHash<>
    >;

    using Container =
        highvoronoi::detail::QueueHashFromParams_t<
            highvoronoi::EmptyLock,
            Params
        >;

    Container queues(
        1,   // initially one contained hash table
        10,  // table index = key[0] / 10
        2    // initial capacity of each contained hash table
    );

    check(queues.table_count() == 1,
          "dynamic queue container starts with one hash table");
    check(queues.block_size() == 10,
          "dynamic queue container stores its block size");

    const std::array<std::uint64_t, 3> key0{5, 100, 200};
    const std::array<std::uint64_t, 3> key1{15, 101, 201};
    const std::array<std::uint64_t, 3> key3{35, 103, 203};
    const std::array<std::uint64_t, 3> absent_key5{55, 105, 205};

    check(!queues.pushqueue(key0),
          "dynamic queue inserts a key into table 0");
    check(!queues.pushqueue(key1),
          "dynamic queue creates and uses table 1");
    check(!queues.pushqueue(key3),
          "dynamic queue grows far enough to use table 3");

    check(queues.table_count() == 4,
          "dynamic queue grows from one to four contained tables");

    check(queues.contains(key0) &&
          queues.contains(key1) &&
          queues.contains(key3),
          "dynamic queue finds keys in different contained tables");

    check(!queues.contains(absent_key5),
          "lookup in a missing dynamic table returns false");
    check(queues.table_count() == 4,
          "read-only lookup does not grow the dynamic container");

    check(queues.erase(key1),
          "dynamic queue erases an existing routed key");
    check(!queues.contains(key1),
          "erased dynamic-container key is no longer found");

    check(!queues.pushqueue(key1),
          "erased dynamic-container key can be inserted again");

    queues.clear();

    check(!queues.contains(key0) &&
          !queues.contains(key1) &&
          !queues.contains(key3),
          "clear removes entries from every dynamic queue table");
    check(queues.table_count() == 4,
          "clear does not shrink the dynamic queue container");
}


void test_static_edge_hash_container()
{
    std::cout << "\n============================================================\n";
    std::cout << "StaticEdgeHashContainer through EdgeBufferParams\n";
    std::cout << "============================================================\n";

    using Generator = highvoronoi::ExtendedHashGenerator<
        highvoronoi::Murmur128HashGenerator<7>,
        highvoronoi::XXHash64<11>
    >;

    using Params = highvoronoi::EdgeBufferParams<
        Generator,
        highvoronoi::StaticHash<4>
    >;

    using Container =
        highvoronoi::detail::EdgeHashFromParams_t<
            highvoronoi::EmptyLock,
            Params
        >;

    Container edges(2);

    check(edges.table_count() == 4,
          "static edge container contains exactly four hash tables");

    const std::array<std::uint64_t, 3> edge0{0, 100, 200};
    const std::array<std::uint64_t, 3> edge1{1, 101, 201};
    const std::array<std::uint64_t, 3> edge2{2, 102, 202};
    const std::array<std::uint64_t, 3> edge3{3, 103, 203};

    check(!edges.pushedge(edge0, 10) &&
          !edges.pushedge(edge1, 11) &&
          !edges.pushedge(edge2, 12) &&
          !edges.pushedge(edge3, 13),
          "static edge container inserts keys into all four routes");

    check(!edges.pushedge(edge1, 21),
          "static edge container stores a second cell");
    check(edges.pushedge(edge1, 31),
          "static edge container reports a third occurrence");

    edges.clear();

    check(!edges.pushedge(edge1, 41),
          "static edge container accepts an edge again after clear");
}


void test_dynamic_edge_hash_container()
{
    std::cout << "\n============================================================\n";
    std::cout << "DynamicEdgeHashContainer through EdgeBufferParams\n";
    std::cout << "============================================================\n";

    using Generator = highvoronoi::Murmur128HashGenerator<13>;

    using Params = highvoronoi::EdgeBufferParams<
        Generator,
        highvoronoi::DynamicHash<>
    >;

    using Container =
        highvoronoi::detail::EdgeHashFromParams_t<
            highvoronoi::EmptyLock,
            Params
        >;

    Container edges(
        1,   // initially one contained hash table
        10,  // table index = key[0] / 10
        2    // initial capacity of each contained hash table
    );

    check(edges.table_count() == 1,
          "dynamic edge container starts with one hash table");
    check(edges.block_size() == 10,
          "dynamic edge container stores its block size");

    const std::array<std::uint64_t, 3> edge0{5, 100, 200};
    const std::array<std::uint64_t, 3> edge3{35, 103, 203};

    check(!edges.pushedge(edge0, 10),
          "dynamic edge container inserts into table 0");
    check(!edges.pushedge(edge3, 20),
          "dynamic edge container grows and inserts into table 3");

    check(edges.table_count() == 4,
          "dynamic edge container grows from one to four contained tables");

    check(!edges.pushedge(edge3, 21),
          "dynamic edge container stores a second cell");
    check(edges.pushedge(edge3, 22),
          "dynamic edge container reports a third occurrence");

    edges.clear();

    check(!edges.pushedge(edge3, 30),
          "dynamic edge container accepts an edge again after clear");
    check(edges.table_count() == 4,
          "clear does not shrink the dynamic edge container");
}

} // namespace

void test_parameter_runtime_forwarding()
{
    std::cout << "\n============================================================\n";
    std::cout << "Parameter runtime forwarding\n";
    std::cout << "============================================================\n";

    // DirectHash
    {
        using Params = highvoronoi::DataBaseParams<>;

        const Params params{
            highvoronoi::DirectHash{32}
        };

        auto queue =
            highvoronoi::detail::make_queue_hash<
                highvoronoi::EmptyLock
            >(params);

        check(
            queue.capacity() == 32,
            "DirectHash forwards hash_capacity"
        );
    }

    // StaticHash
    {
        using Params = highvoronoi::DataBaseParams<
            double,
            std::int64_t,
            highvoronoi::FNV64_128HashGenerator,
            highvoronoi::StaticHash<4>
        >;

        const Params params{
            highvoronoi::StaticHash<4>{16}
        };

        auto queues =
            highvoronoi::detail::make_queue_hash<
                highvoronoi::EmptyLock
            >(params);

        check(
            queues.table_count() == 4,
            "StaticHash forwards the compile-time table count"
        );
    }

    // DynamicHash
    {
        using Params = highvoronoi::DataBaseParams<
            double,
            std::int64_t,
            highvoronoi::FNV64_128HashGenerator,
            highvoronoi::DynamicHash<>
        >;

        const Params params{
            highvoronoi::DynamicHash<>{
                3,
                100,
                16
            }
        };

        auto queues =
            highvoronoi::detail::make_queue_hash<
                highvoronoi::EmptyLock
            >(params);

        check(
            queues.table_count() == 3,
            "DynamicHash forwards initial_table_count"
        );

        check(
            queues.block_size() == 100,
            "DynamicHash forwards block_size"
        );
    }
}

int main()
{
    try {
        std::cout << "HighVoronoi hashing tests\n";

        test_uint64_view();
        test_hash_functions();
        test_hash_generators();
        test_queue_hash_table();
        test_edge_hash_table();

        test_parameter_types();
        test_static_queue_hash_container();
        test_dynamic_queue_hash_container();
        test_static_edge_hash_container();
        test_dynamic_edge_hash_container();
        test_parameter_runtime_forwarding();

        std::cout << "\n============================================================\n";
        std::cout << "ALL HASHING TESTS PASSED\n";
        std::cout << "============================================================\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "\n============================================================\n";
        std::cerr << "TEST FAILED: " << error.what() << '\n';
        std::cerr << "============================================================\n";
        return 1;
    }
}

