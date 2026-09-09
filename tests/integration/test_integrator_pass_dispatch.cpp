#include <cstddef>
#include <iostream>
#include <string_view>
#include <vector>

namespace polygon_dispatch_test {
struct FakeIntegral;
struct FakeView;
FakeView make_integration_view(FakeIntegral& integral);
} // namespace polygon_dispatch_test

#include <highvoronoi/integration/integrator.hpp>

namespace polygon_dispatch_test {

std::size_t checks = 0;
std::size_t failures = 0;

void check(bool ok, std::string_view text) {
    ++checks;
    if (ok) std::cout << "[OK]   " << text << '\n';
    else { ++failures; std::cerr << "[FAIL] " << text << '\n'; }
}

struct FakeUpdate {
    std::size_t id = 0;
    std::vector<std::size_t> neighbours_;
    std::vector<unsigned char> recompute_;
    int value = 0;

    std::size_t cell() const noexcept { return id; }
    const std::vector<std::size_t>& neighbours() const noexcept {
        return neighbours_;
    }
    bool interface_requires_recompute(std::size_t k) const {
        return recompute_.at(k) != 0;
    }
};

struct FakeIntegral {};

struct FakeView {
    using Index = std::size_t;
    using CellUpdate = FakeUpdate;

    std::vector<std::size_t> committed;

    std::size_t update_count() const noexcept { return 3; }
    void prepare_update(Index i, CellUpdate& u) {
        u.id = i;
        u.neighbours_ = i == 0 ? std::vector<std::size_t>{1, 2}
                               : std::vector<std::size_t>{0};
        u.recompute_.assign(u.neighbours_.size(), 1);
    }
    void commit_update(CellUpdate& u) { committed.push_back(u.id); }
    void finish_update() {}
};

FakeView make_integration_view(FakeIntegral&) { return FakeView{}; }

struct UpdateOnlyAlgorithm {
    std::vector<std::size_t> seen;
    void integrate_cell(FakeUpdate& update) {
        seen.push_back(update.id);
        update.value = 10 + static_cast<int>(update.id);
    }
};

struct PassAwareAlgorithm {
    std::vector<std::size_t> seen;
    std::vector<int> previous_values;

    template <class Pass>
    void integrate_cell(Pass& pass, std::size_t position) {
        auto& update = pass.writable_cell(position);
        seen.push_back(update.id);
        if (position != 0) {
            previous_values.push_back(pass.cell(position - 1).value);
        }
        update.value = 100 + static_cast<int>(position);
    }
};

void test_update_contract_unchanged() {
    FakeIntegral integral;
    UpdateOnlyAlgorithm algorithm;
    const auto report = highvoronoi::integrate(integral, algorithm);
    check(algorithm.seen == std::vector<std::size_t>({0,1,2}),
          "legacy integrate_cell(Update&) dispatch remains unchanged");
    check(report.updated_cells == 3,
          "legacy dispatch still reports all first-pass cells");
}

void test_pass_aware_contract() {
    FakeIntegral integral;
    PassAwareAlgorithm algorithm;
    const auto report = highvoronoi::integrate(integral, algorithm);
    check(algorithm.seen == std::vector<std::size_t>({0,1,2}),
          "pass-aware algorithm receives cells in Julia cellwise order");
    check(algorithm.previous_values == std::vector<int>({100,101}),
          "pass-aware algorithm can read already-computed first-pass cells");
    check(report.updated_cells == 3,
          "pass-aware dispatch preserves integration report accounting");
}

} // namespace polygon_dispatch_test

int main() {
    polygon_dispatch_test::test_update_contract_unchanged();
    polygon_dispatch_test::test_pass_aware_contract();
    std::cout << "checks=" << polygon_dispatch_test::checks
              << " failures=" << polygon_dispatch_test::failures << '\n';
    return polygon_dispatch_test::failures == 0 ? 0 : 1;
}
