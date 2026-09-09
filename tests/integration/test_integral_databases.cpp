#include <highvoronoi/integrals.hpp>
#include <highvoronoi/core/detail/locks.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace {
using Lock = highvoronoi::detail::EmptyLock;
using Index = std::uint32_t;
using Area = double;
using Integral = float;

std::size_t checks = 0;
std::size_t failures = 0;
void check(bool ok, const char* text) {
    ++checks;
    if (!ok) { ++failures; std::cerr << "[FAIL] " << text << '\n'; }
    else { std::cout << "[OK]   " << text << '\n'; }
}

template <class T>
bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a == b;
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
        volume = 10.0 + static_cast<Area>(node);
        return true;
    }

    Address area_address_capacity() const noexcept override { return 2; }
    bool read_area(Address address, AreaBuffer& areas) const override {
        if (address == 0) { areas = {1.0, 2.0}; return true; }
        if (address == 1) { areas = {}; return true; } // deliberately odd but legal DB record
        areas.clear(); return false;
    }
    std::optional<Address> area_address(Index node) const override {
        if (node < 2) return static_cast<Address>(node);
        return std::nullopt;
    }

    std::size_t integral_components() const noexcept override { return 2; }
    Address integral_address_capacity() const noexcept override { return 4; }
    bool read_integral(Address address, IntegralBuffer& values) const override {
        switch (address) {
            case 0: values = {10.f, 11.f}; return true;
            case 1: values = {20.f, 21.f, 30.f, 31.f}; return true;
            case 2: values = {}; return true; // nonsensical semantically, legal storage
            case 3: values = {50.f}; return true; // nonsensical semantically, legal storage
            default: values.clear(); return false;
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
    {
        highvoronoi::detail::AreaDatabase<Lock, Area> db(5);
        const auto a = db.push(std::vector<Area>{1.25, 2.5, 3.75});
        const auto b = db.push_scalar(9.5);
        const auto empty = db.push(std::vector<Area>{});
        const auto nonsense = db.push(std::vector<Area>{-7.0, 0.0, 1234.5, -9.0});
        std::vector<Area> out;
        check(db.read(a, out) && same(out, std::vector<Area>{1.25,2.5,3.75}),
              "AreaDatabase stores variable-length floating records across blocks");
        Area scalar = 0;
        check(db.read_scalar(b, scalar) && scalar == 9.5,
              "AreaDatabase supports scalar records");
        check(db.read(empty, out) && out.empty(),
              "AreaDatabase stores and reads an empty record");
        check(db.read(nonsense, out) && same(out, std::vector<Area>{-7.0,0.0,1234.5,-9.0}),
              "AreaDatabase is storage only and preserves semantically nonsensical values exactly");
        db.overwrite(a, std::vector<Area>{6.0, 7.0, 8.0});
        check(db.read(a, out) && same(out, std::vector<Area>{6.0,7.0,8.0}),
              "AreaDatabase overwrite replaces payload without changing address");
        bool length_error = false;
        try { db.overwrite(a, std::vector<Area>{1.0, 2.0}); }
        catch (const std::length_error&) { length_error = true; }
        check(length_error,
              "AreaDatabase overwrite rejects a different record length");
    }

    {
        using RWLock = highvoronoi::detail::ReadWriteLock;
        highvoronoi::detail::AreaDatabase<RWLock, Area> db(5);
        const auto address = db.push(std::vector<Area>{4.0, 5.0});
        std::vector<Area> out;
        check(db.read(address, out) && same(out, std::vector<Area>{4.0, 5.0}),
              "AreaDatabase instantiates with the project read/write lock policy");
    }

    {
        highvoronoi::detail::IntegralDatabase<Lock, Integral> db(7);
        const auto scalar = db.push(std::vector<Integral>{2.f});
        const auto vector = db.push(std::vector<Integral>{3.f,4.f,5.f});
        const auto odd = db.push(std::vector<Integral>{});
        std::vector<Integral> out;
        check(db.read(scalar, out) && same(out, std::vector<Integral>{2.f}),
              "IntegralDatabase represents scalar integrals as length-one records");
        check(db.read(vector, out) && same(out, std::vector<Integral>{3.f,4.f,5.f}),
              "IntegralDatabase represents vector integrals without a separate type");
        check(db.read(odd, out) && out.empty(),
              "IntegralDatabase itself imposes no semantic component-count restriction");
        db.overwrite(vector, std::vector<Integral>{9.f, 8.f, 7.f});
        check(db.read(vector, out) && same(out, std::vector<Integral>{9.f,8.f,7.f}),
              "IntegralDatabase overwrite preserves address and updates equal-length payload");
    }

    auto mesh = std::make_shared<FakeMeshEngine>(2);
    auto engine = std::make_shared<FakeIntegralEngine>(mesh);
    using Adapter = highvoronoi::detail::IntegralEngineDatabaseAdapter<FakeIntegralEngine>;
    auto adapter = std::make_shared<Adapter>(engine);

    {
        highvoronoi::detail::HybridAreaDatabase<Lock, Area, Index> db(8);
        const auto stored0 = db.push(std::vector<Area>{7.0});
        std::shared_ptr<const highvoronoi::detail::AreaEngineInterface<Area,Index>> area_adapter = adapter;
        const auto reg = db.register_engine(area_adapter);
        const auto stored1 = db.push(std::vector<Area>{8.0,9.0});
        std::vector<Area> out;
        check(stored0 == 1 && reg.first_address == 2 && reg.address_capacity == 2 && stored1 == 4,
              "HybridAreaDatabase uses one stored/engine/stored logical address space");
        check(db.read(reg.logical_address(0), out) && same(out, std::vector<Area>{1.0,2.0}),
              "HybridAreaDatabase reads area records through the shared engine adapter");
        check(db.read(stored1, out) && same(out, std::vector<Area>{8.0,9.0}),
              "HybridAreaDatabase keeps later stored records readable");
        db.overwrite(stored1, std::vector<Area>{18.0, 19.0});
        check(db.read(stored1, out) && same(out, std::vector<Area>{18.0,19.0}),
              "HybridAreaDatabase forwards overwrite to stored records");
        bool engine_overwrite_rejected = false;
        try { db.overwrite(reg.logical_address(0), std::vector<Area>{1.0, 2.0}); }
        catch (const std::logic_error&) { engine_overwrite_rejected = true; }
        check(engine_overwrite_rejected,
              "HybridAreaDatabase rejects overwrite of immutable engine records");
    }

    {
        highvoronoi::detail::HybridIntegralDatabase<Lock, Integral, Index> db(8);
        std::shared_ptr<const highvoronoi::detail::IntegralEngineInterface<Integral,Index>> integral_adapter = adapter;
        const auto reg = db.register_engine(integral_adapter);
        std::vector<Integral> out;
        check(reg.first_address == 1 && reg.address_capacity == 4,
              "HybridIntegralDatabase reserves its own independent engine address range");
        check(db.read(reg.logical_address(1), out) &&
                  same(out, std::vector<Integral>{20.f,21.f,30.f,31.f}),
              "HybridIntegralDatabase reads vector records through the same adapter object");
    }

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
