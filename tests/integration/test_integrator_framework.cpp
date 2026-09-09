
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fake_integration_test {
struct FakeIntegral;
struct FakeView;
FakeView make_integration_view(FakeIntegral& integral);
} // namespace fake_integration_test

#include <highvoronoi/integration/integrator.hpp>

namespace fake_integration_test {

std::size_t checks = 0;
std::size_t failures = 0;
std::vector<std::string> trace;

void check(bool ok, std::string_view message) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cerr << "[FAIL] " << message << '\n';
    } else {
        std::cout << "[OK]   " << message << '\n';
    }
}

struct FakeUpdate {
    using Index = std::size_t;
    Index cell_id = 0;
    std::vector<Index> neigh;
    std::vector<int> values;
    std::vector<unsigned char> recompute;

    Index cell() const noexcept { return cell_id; }
    const std::vector<Index>& neighbours() const noexcept { return neigh; }
    bool interface_requires_recompute(std::size_t k) const {
        return recompute.at(k) != 0;
    }
};

struct FakeIntegral {};

struct FakeView {
    using Index = std::size_t;
    using CellUpdate = FakeUpdate;

    explicit FakeView(FakeIntegral&) {}

    std::size_t update_count() const noexcept { return 3; }

    void prepare_update(Index cell, CellUpdate& update) {
        trace.push_back("prepare:" + std::to_string(cell));
        prepared.push_back(cell);
        update.cell_id = cell;
        update.neigh = {cell + 10, cell + 20};
        update.values = {static_cast<int>(100 + cell)};
        update.recompute = {static_cast<unsigned char>(1),
                            static_cast<unsigned char>(cell == 1 ? 0 : 1)};
    }

    void prepare_cleanup(CellUpdate& update) {
        trace.push_back("prepare_cleanup:" + std::to_string(update.cell_id));
    }

    void commit_update(CellUpdate& update) {
        if (!cleanup_finished) {
            throw std::logic_error("final commit before cleanup");
        }
        trace.push_back("commit:" + std::to_string(update.cell_id));
        committed.push_back(update.cell_id);
        committed_values.push_back(update.values.front());
    }

    void finish_update() {
        trace.push_back("finish");
        finished = true;
    }

    std::vector<Index> prepared;
    std::vector<Index> committed;
    std::vector<int> committed_values;
    bool cleanup_finished = false;
    bool finished = false;
};

FakeView make_integration_view(FakeIntegral& integral) {
    return FakeView(integral);
}

struct FakeAlgorithm {
    std::vector<std::string> events;

    template <class Pass>
    void begin_pass(Pass& pass) {
        events.push_back("begin:" + std::to_string(pass.size()));
        trace.push_back(events.back());
    }

    void integrate_cell(FakeUpdate& update) {
        events.push_back("integrate:" + std::to_string(update.cell_id));
        trace.push_back(events.back());
        update.values.front() += static_cast<int>(update.cell_id);
    }

    template <class Pass>
    void cleanup_cell(Pass& pass, std::size_t position) {
        auto& update = pass.writable_cell(position);
        events.push_back("cleanup:" + std::to_string(update.cell_id));
        trace.push_back(events.back());
        update.values.front() += 1000;
        if (position + 1 == pass.size()) {
            pass.view().cleanup_finished = true;
        }
    }

    template <class Pass>
    void end_pass(Pass& pass) {
        check(pass.view().committed.empty(),
              "end_pass happens before final cleanup commit");
        events.push_back("end");
        trace.push_back("end");
    }
};

} // namespace fake_integration_test

int main() {
    using namespace fake_integration_test;

    std::cout << "\n============================================================\n";
    std::cout << "[TEST] cellwise two-phase integrator framework\n";
    std::cout << "============================================================\n";

    FakeIntegral integral;
    FakeAlgorithm algorithm;
    const auto report = highvoronoi::integrate(integral, algorithm);

    check(report.updated_cells == 3,
          "driver reports exactly the update prefix size");
    check(report.recomputed_interfaces == 5,
          "driver counts recomputed interfaces before the algorithm runs");

    check(trace.size() >= 10,
          "framework trace captured cellwise first-pass and cleanup ordering");

    const std::vector<std::string> expected_trace{
        "begin:3",
        "prepare:0", "integrate:0",
        "prepare:1", "integrate:1",
        "prepare:2", "integrate:2",
        "prepare_cleanup:0", "cleanup:0",
        "prepare_cleanup:1", "cleanup:1",
        "prepare_cleanup:2", "cleanup:2",
        "end",
        "commit:0", "commit:1", "commit:2",
        "finish"};
    check(trace == expected_trace,
          "driver preserves Julia cellwise prepare-compute order before cleanup");

    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}


