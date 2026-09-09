#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/storage/hybrid_database.hpp>
#include <highvoronoi/mesh/engine/compute_mesh.hpp>
#include <highvoronoi/mesh/engine/cuboid_mesh_engine.hpp>
#include <highvoronoi/parameters.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <vector>

namespace {

std::atomic<std::size_t> allocation_count{0};
std::atomic<bool> count_allocations{false};

void note_allocation() noexcept {
    if (count_allocations.load(std::memory_order_relaxed)) {
        allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
}

class AllocationScope final {
public:
    AllocationScope() {
        allocation_count.store(0, std::memory_order_relaxed);
        count_allocations.store(true, std::memory_order_relaxed);
    }

    ~AllocationScope() {
        count_allocations.store(false, std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t count() const noexcept {
        return allocation_count.load(std::memory_order_relaxed);
    }
};

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dim = 3;
using Engine = highvoronoi::CuboidMeshEngine<Scalar, Scalar, Index, Dim>;
using Point = Engine::VertexPoint;
using Sigma = Engine::Sigma;
using Params = highvoronoi::DataBaseParams<Scalar, Index>;

Engine make_engine() {
    Engine::NodePoint start;
    start << 0.0, 0.0, 0.0;
    Engine::NodePoint spacing;
    spacing << 1.0, 1.0, 1.0;
    Engine::CountPoint repetitions;
    repetitions << Index{4}, Index{4}, Index{4};
    return Engine(start, spacing, repetitions);
}

bool test_hvdatabase_read_no_allocations() {
    using Database = highvoronoi::HVDataBase<
        highvoronoi::EmptyLock,
        Params,
        Dim>;

    Database database(4096, Params{highvoronoi::DirectHash{1024}});
    Point stored_position;
    stored_position << 0.5, 0.5, 0.5;
    Sigma stored_sigma{Index{0}, Index{1}, Index{4}, Index{16}};
    const auto address = database.push(stored_position, stored_sigma);
    if (address == 0) {
        std::cerr << "HVDataBase setup unexpectedly produced a duplicate.\n";
        return false;
    }

    Point position;
    Sigma sigma;
    sigma.reserve(stored_sigma.size());
    database.read(address, position, sigma);

    AllocationScope scope;
    for (std::size_t i = 0; i < 1000; ++i) {
        database.read(address, position, sigma);
    }
    const auto allocations = scope.count();
    if (allocations != 0) {
        std::cerr << "HVDataBase::read allocated " << allocations
                  << " times after warm-up.\n";
        return false;
    }
    return true;
}

bool test_compute_mesh_read_no_allocations() {
    Engine engine = make_engine();
    highvoronoi::detail::ComputeMeshDatabaseAdapter<Engine> adapter(engine);

    Point position;
    Sigma sigma;
    sigma.reserve(std::size_t{1} << Dim);
    adapter.read(std::size_t{1}, position, sigma);
    if (!adapter.contains(sigma)) {
        std::cerr << "ComputeMesh adapter setup signature was not found.\n";
        return false;
    }

    AllocationScope scope;
    for (std::size_t i = 0; i < 1000; ++i) {
        adapter.read(std::size_t{1}, position, sigma);
        if (!adapter.contains(sigma)) {
            std::cerr << "ComputeMesh adapter lost its warmed-up signature.\n";
            return false;
        }
    }
    const auto allocations = scope.count();
    if (allocations != 0) {
        std::cerr << "ComputeMesh read/contains allocated " << allocations
                  << " times after warm-up.\n";
        return false;
    }
    return true;
}

bool test_hybrid_engine_read_no_allocations() {
    using Database = highvoronoi::HybridDataBase<
        highvoronoi::EmptyLock,
        Params,
        Dim>;

    auto engine = std::make_shared<Engine>(make_engine());
    Database database(4096, Params{highvoronoi::DirectHash{1024}});
    const auto registration = database.register_engine(engine, Index{0});
    if (registration.address_capacity == 0) {
        std::cerr << "Cuboid engine unexpectedly has no finite vertices.\n";
        return false;
    }

    Point position;
    Sigma sigma;
    sigma.reserve(std::size_t{1} << Dim);
    database.read(registration.first_address, position, sigma);

    AllocationScope scope;
    for (std::size_t i = 0; i < 1000; ++i) {
        database.read(registration.first_address, position, sigma);
    }
    const auto allocations = scope.count();
    if (allocations != 0) {
        std::cerr << "HybridDataBase engine read allocated " << allocations
                  << " times after warm-up.\n";
        return false;
    }
    return true;
}

} // namespace

void* operator new(std::size_t size) {
    note_allocation();
    if (void* ptr = std::malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    note_allocation();
    if (void* ptr = std::malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

int main() {
    const bool hvdatabase_ok = test_hvdatabase_read_no_allocations();
    const bool compute_ok = test_compute_mesh_read_no_allocations();
    const bool hybrid_ok = test_hybrid_engine_read_no_allocations();

    if (hvdatabase_ok && compute_ok && hybrid_ok) {
        std::cout << "Allocation-free read hotpaths verified after warm-up.\n";
        return 0;
    }
    return 1;
}
