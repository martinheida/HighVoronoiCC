#include "voronoi_incremental_test_tools.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace voronoi_incremental_test;

inline constexpr std::array<Index, 2> NodeCounts3D{
    Index{24}, Index{50}};
inline constexpr std::array<Index, 2> NodeCounts4D{
    Index{18}, Index{36}};

// -----------------------------------------------------------------------------
// 1. One refinement step.
// -----------------------------------------------------------------------------

template <int Dimension>
bool run_single_refine(Index initial_count) {
    using T = Types<Dimension>;
    using Point = typename T::Point;
    using Mesh = typename T::Mesh;
    using Refine = typename T::Refine;

    const Index add_count = initial_count < Index{30} ? Index{4} : Index{7};
    const auto all = make_random_points<Dimension>(
        static_cast<Index>(initial_count + add_count),
        case_seed<Dimension>(0x53494e474c455245ULL, initial_count));

    std::vector<Point> current(
        all.begin(),
        all.begin() + static_cast<std::ptrdiff_t>(initial_count));
    const std::vector<Point> added(
        all.begin() + static_cast<std::ptrdiff_t>(initial_count),
        all.end());

    auto mesh = make_mesh<Dimension>(current);
    compute_full<Dimension>(*mesh);
    if (!verify_complete_state<Dimension>(*mesh, "single refine initial")) {
        return false;
    }

    Refine refine(
        *mesh,
        added,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters<Dimension>(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());
    (void)refine.compute();

    append_points(current, added);

    return refine.report().appended_nodes == added.size() &&
           validate_against_fresh_reference<Dimension>(
               *mesh,
               current,
               case_label<Dimension>("single refine", initial_count));
}

// -----------------------------------------------------------------------------
// 2. Three successive refinement steps with independent operation state.
// -----------------------------------------------------------------------------

template <int Dimension>
bool run_three_refines(Index initial_count) {
    using T = Types<Dimension>;
    using Point = typename T::Point;
    using Refine = typename T::Refine;

    constexpr std::array<std::size_t, 3> BatchSizes{2, 3, 4};
    constexpr std::size_t TotalAdded = 9;

    const auto all = make_random_points<Dimension>(
        static_cast<Index>(initial_count + Index{TotalAdded}),
        case_seed<Dimension>(0x5448524545524546ULL, initial_count));

    std::vector<Point> current(
        all.begin(),
        all.begin() + static_cast<std::ptrdiff_t>(initial_count));

    auto mesh = make_mesh<Dimension>(current);
    compute_full<Dimension>(*mesh);
    if (!verify_complete_state<Dimension>(*mesh, "three refines initial")) {
        return false;
    }


    std::size_t offset = static_cast<std::size_t>(initial_count);
    for (std::size_t stage = 0; stage < BatchSizes.size(); ++stage) {
        const std::vector<Point> added = slice(
            all,
            offset,
            BatchSizes[stage]);
        offset += BatchSizes[stage];

        Refine refine(
            *mesh,
            added,
            highvoronoi::geometry::KDSearch{8, 1},
            ray_parameters<Dimension>(),
            highvoronoi::SingleThread{},
            highvoronoi::SingleThread{},
            database_parameters<Dimension>(),
            edge_parameters<Dimension>());

        auto& affected = refine.compute();
        if (&affected != &refine.affected()) {
            std::cerr << "      RefineVoronoi did not return its own AFFECTED vector\n";
            return false;
        }
        if (affected.size() != static_cast<std::size_t>(mesh->internal_size())) {
            std::cerr << "      RefineVoronoi AFFECTED vector does not cover stable internal slots\n";
            return false;
        }

        append_points(current, added);

        const std::string label =
            case_label<Dimension>("three refines", initial_count) +
            " stage=" + std::to_string(stage + 1);

        if (!validate_against_fresh_reference<Dimension>(
                *mesh,
                current,
                label)) {
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// 3. Removal with immediate repair.
// -----------------------------------------------------------------------------

template <int Dimension>
bool run_remove_with_recompute(Index initial_count) {
    using T = Types<Dimension>;
    using Point = typename T::Point;
    using Remove = typename T::Remove;

    std::vector<Point> current = make_random_points<Dimension>(
        initial_count,
        case_seed<Dimension>(0x52454d4f56455245ULL, initial_count));

    auto mesh = make_mesh<Dimension>(current);
    compute_full<Dimension>(*mesh);
    if (!verify_complete_state<Dimension>(*mesh, "remove initial")) {
        return false;
    }

    const std::vector<Index> deleted = deletion_indices(initial_count, 0);

    Remove remove(
        *mesh,
        deleted,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters<Dimension>(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());
    (void)remove.compute();

    erase_public_points(current, deleted);

    return remove.report().removed_nodes == deleted.size() &&
           remove.report().recomputed &&
           validate_against_fresh_reference<Dimension>(
               *mesh,
               current,
               case_label<Dimension>("remove+repair", initial_count));
}

// -----------------------------------------------------------------------------
// 4. remove -> refine -> remove -> refine with independent operation state.
//    Every intermediate repaired state is compared with a fresh reference.
// -----------------------------------------------------------------------------

template <int Dimension>
bool run_remove_refine_remove_refine(Index initial_count) {
    using T = Types<Dimension>;
    using Point = typename T::Point;
    using Remove = typename T::Remove;
    using Refine = typename T::Refine;

    constexpr Index FirstAdd = Index{3};
    constexpr Index SecondAdd = Index{4};

    const auto all = make_random_points<Dimension>(
        static_cast<Index>(initial_count + FirstAdd + SecondAdd),
        case_seed<Dimension>(0x4d49584544574f52ULL, initial_count));

    std::vector<Point> current(
        all.begin(),
        all.begin() + static_cast<std::ptrdiff_t>(initial_count));
    const std::vector<Point> first_added = slice(
        all,
        static_cast<std::size_t>(initial_count),
        static_cast<std::size_t>(FirstAdd));
    const std::vector<Point> second_added = slice(
        all,
        static_cast<std::size_t>(initial_count + FirstAdd),
        static_cast<std::size_t>(SecondAdd));

    auto mesh = make_mesh<Dimension>(current);
    compute_full<Dimension>(*mesh);
    if (!verify_complete_state<Dimension>(*mesh, "mixed workflow initial")) {
        return false;
    }


    // Remove 1.
    const auto deleted1 = deletion_indices(mesh->size(), 0);
    Remove remove1(
        *mesh,
        deleted1,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters<Dimension>(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());
    (void)remove1.compute();
    erase_public_points(current, deleted1);
    if (!validate_against_fresh_reference<Dimension>(
            *mesh,
            current,
            case_label<Dimension>("mixed remove1", initial_count))) {
        return false;
    }

    // Refine 1.
    Refine refine1(
        *mesh,
        first_added,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters<Dimension>(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());
    (void)refine1.compute();
    append_points(current, first_added);
    if (!validate_against_fresh_reference<Dimension>(
            *mesh,
            current,
            case_label<Dimension>("mixed refine1", initial_count))) {
        return false;
    }

    // Remove 2. The second deletion set is chosen from the new public state and
    // deliberately includes a near-tail node, so an earlier refinement node can
    // be removed as well.
    const auto deleted2 = deletion_indices(mesh->size(), 1);
    Remove remove2(
        *mesh,
        deleted2,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters<Dimension>(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());
    (void)remove2.compute();
    erase_public_points(current, deleted2);
    if (!validate_against_fresh_reference<Dimension>(
            *mesh,
            current,
            case_label<Dimension>("mixed remove2", initial_count))) {
        return false;
    }

    // Refine 2.
    Refine refine2(
        *mesh,
        second_added,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters<Dimension>(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());
    (void)refine2.compute();
    append_points(current, second_added);

    return validate_against_fresh_reference<Dimension>(
        *mesh,
        current,
        case_label<Dimension>("mixed refine2", initial_count));
}

// -----------------------------------------------------------------------------
// 5. Keep one remover alive across an independent refinement.
//    remove() creates deletion damage; RefineVoronoi handles only insertion;
//    the original remover then closes exactly its own AFFECTED cells.
// -----------------------------------------------------------------------------

template <int Dimension>
bool run_deferred_remove_then_refine(Index initial_count) {
    using T = Types<Dimension>;
    using Point = typename T::Point;
    using Remove = typename T::Remove;
    using Refine = typename T::Refine;

    constexpr Index AddCount = Index{5};

    const auto all = make_random_points<Dimension>(
        static_cast<Index>(initial_count + AddCount),
        case_seed<Dimension>(0x4445464552524544ULL, initial_count));

    std::vector<Point> current(
        all.begin(),
        all.begin() + static_cast<std::ptrdiff_t>(initial_count));
    const std::vector<Point> added = slice(
        all,
        static_cast<std::size_t>(initial_count),
        static_cast<std::size_t>(AddCount));

    auto mesh = make_mesh<Dimension>(current);
    compute_full<Dimension>(*mesh);
    if (!verify_complete_state<Dimension>(*mesh, "deferred workflow initial")) {
        return false;
    }

    const auto deleted = deletion_indices(mesh->size(), 1);

    // Keep this remover alive across the complete refinement operation. Its
    // AFFECTED state describes deletion damage only and is intentionally not
    // shared with RefineVoronoi.
    Remove remover(
        *mesh,
        deleted,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters<Dimension>(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());

    auto& remove_affected = remover.remove();
    if (!remover.report().removed || remover.report().recomputed) {
        std::cerr << "      RemoveVoronoi did not stop after the remove phase\n";
        return false;
    }
    if (&remove_affected != &remover.affected()) {
        std::cerr << "      RemoveVoronoi did not return its own AFFECTED state\n";
        return false;
    }
    const std::size_t remove_affected_count_before_refine =
        affected_count(remove_affected);
    if (remove_affected_count_before_refine == 0) {
        std::cerr << "      deferred RemoveVoronoi produced an empty AFFECTED set\n";
        return false;
    }

    erase_public_points(current, deleted);

    // The removal phase must never leave a geometrically invalid surviving
    // vertex. Completeness is deliberately postponed until remover.compute().
    const auto after_remove_consistency = highvoronoi::verify_mesh(
        *mesh,
        VerificationTolerance,
        false);
    if (!after_remove_consistency.valid()) {
        std::cerr << "      deferred remove left an inconsistent stored vertex\n";
        (void)highvoronoi::verify_mesh(
            *mesh,
            VerificationTolerance,
            true);
        return false;
    }

    // Refine independently on the incomplete-but-consistent mesh. It computes
    // NEW cells and removes OLD vertices invalidated by NEW, but it neither
    // reads nor modifies the remover's AFFECTED state.
    Refine refine(
        *mesh,
        added,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters<Dimension>(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters<Dimension>(),
        edge_parameters<Dimension>());

    (void)refine.compute();
    append_points(current, added);

    if (affected_count(remover.affected()) !=
        remove_affected_count_before_refine) {
        std::cerr << "      RefineVoronoi modified the remover's AFFECTED state\n";
        return false;
    }

    const auto after_refine_consistency = highvoronoi::verify_mesh(
        *mesh,
        VerificationTolerance,
        false);
    if (!after_refine_consistency.valid()) {
        std::cerr << "      refine on deferred-remove mesh left an inconsistent vertex\n";
        (void)highvoronoi::verify_mesh(
            *mesh,
            VerificationTolerance,
            true);
        return false;
    }

    // The mesh gained stable slots while the remover was alive. The remover
    // resizes only its own bit vector immediately before the delayed closure.
    const std::size_t size_before_closure = remover.affected().size();
    if (size_before_closure >= static_cast<std::size_t>(mesh->internal_size())) {
        std::cerr << "      remover AFFECTED unexpectedly followed refinement growth\n";
        return false;
    }

    auto& closed_affected = remover.compute();
    if (&closed_affected != &remover.affected() ||
        !remover.report().recomputed) {
        std::cerr << "      delayed RemoveVoronoi closure did not complete\n";
        return false;
    }
    if (closed_affected.size() !=
        static_cast<std::size_t>(mesh->internal_size())) {
        std::cerr << "      remover AFFECTED was not extended to current stable size\n";
        return false;
    }

    return validate_against_fresh_reference<Dimension>(
        *mesh,
        current,
        case_label<Dimension>("remove -> refine -> remove closure", initial_count));
}

template <int Dimension, class Function>
void run_dimension_matrix(
    std::string_view workflow,
    const std::array<Index, 2>& node_counts,
    Function&& function) {

    for (const Index node_count : node_counts) {
        const bool passed = function.template operator()<Dimension>(node_count);
        check(
            passed,
            case_label<Dimension>(workflow, node_count));
    }
}

struct SingleRefineRunner {
    template <int Dimension>
    bool operator()(Index n) const { return run_single_refine<Dimension>(n); }
};

struct ThreeRefinesRunner {
    template <int Dimension>
    bool operator()(Index n) const { return run_three_refines<Dimension>(n); }
};

struct RemoveRunner {
    template <int Dimension>
    bool operator()(Index n) const { return run_remove_with_recompute<Dimension>(n); }
};

struct MixedRunner {
    template <int Dimension>
    bool operator()(Index n) const { return run_remove_refine_remove_refine<Dimension>(n); }
};

struct DeferredRunner {
    template <int Dimension>
    bool operator()(Index n) const { return run_deferred_remove_then_refine<Dimension>(n); }
};

} // namespace

int main() {
    section("1. One refine step vs fresh full computation");
    run_dimension_matrix<3>("single refine", NodeCounts3D, SingleRefineRunner{});
    run_dimension_matrix<4>("single refine", NodeCounts4D, SingleRefineRunner{});

    section("2. Three successive refine steps with independent AFFECTED state");
    run_dimension_matrix<3>("three refines", NodeCounts3D, ThreeRefinesRunner{});
    run_dimension_matrix<4>("three refines", NodeCounts4D, ThreeRefinesRunner{});

    section("3. Remove nodes and recompute affected cells");
    run_dimension_matrix<3>("remove+repair", NodeCounts3D, RemoveRunner{});
    run_dimension_matrix<4>("remove+repair", NodeCounts4D, RemoveRunner{});

    section("4. remove -> refine -> remove -> refine");
    run_dimension_matrix<3>("mixed workflow", NodeCounts3D, MixedRunner{});
    run_dimension_matrix<4>("mixed workflow", NodeCounts4D, MixedRunner{});

    section("5. remove.remove() -> refine.compute() -> remove.compute()");
    run_dimension_matrix<3>("remove/refine/delayed closure", NodeCounts3D, DeferredRunner{});
    run_dimension_matrix<4>("remove/refine/delayed closure", NodeCounts4D, DeferredRunner{});

    section("Result");
    std::cout << "performed checks: " << performed_checks << '\n'
              << "failed checks:    " << failed_checks << '\n';

    return failed_checks == 0 ? 0 : 1;
}
