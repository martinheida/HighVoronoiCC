#include <iostream>

#include "hvcc_test_support_voronoi_nodes_20260803.hpp"
#include "hvcc_voronoi_nodes_fixtures_20260803.hpp"
#include "hvcc_voronoi_nodes_compile_contracts_20260803.hpp"
#include "hvcc_voronoi_nodes_basic_tests_20260803.hpp"
#include "hvcc_voronoi_nodes_extended_tests_20260803.hpp"

int main() {
    using hvtest::failed_checks;
    using hvtest::performed_checks;
    using hvtest::run_test;
    using namespace voronoi_nodes_test;

    std::cout << "Running VoronoiNodes tests\n"
              << "==========================\n";

    run_test("stored node provider", test_stored_branch);
    run_test("dynamic stored node provider", test_dynamic_stored_branch);
    run_test("computed node provider", test_computed_branch);
    run_test("internally mixed node provider", test_mixed_branch);

    run_test("extended stored base", test_extended_stored_branch);
    run_test("extended computed base", test_extended_computed_base);
    run_test("extended internally mixed base", test_extended_mixed_base);

    run_test("precomputed extended stored base", test_precomputed_stored_base);
    run_test("precomputed extended computed base", test_precomputed_computed_base);
    run_test("precomputed extended internally mixed base", test_precomputed_mixed_base);

    run_test(
        "Boundary common-interface integration",
        test_boundary_check_uses_common_interface);

    run_test(
        "invalid mirror-index handling",
        test_invalid_mirror_index_is_rejected);

    std::cout << "\nSummary\n"
              << "=======\n"
              << "Checks:   " << performed_checks << '\n'
              << "Failures: " << failed_checks << '\n';

    if (failed_checks == 0) {
        std::cout << "All VoronoiNodes tests passed.\n";
        return 0;
    }

    std::cerr << "VoronoiNodes tests failed.\n";
    return 1;
}
