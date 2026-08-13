#pragma once

/**
 * @file search_tree_factory_crtp.hpp
 * @brief Built-in search keywords and extensible compile-time tree factory.
 */

#include <highvoronoi/geometry/brute_force_search_tree_crtp.hpp>
#include <highvoronoi/geometry/nanoflann_search_tree_crtp.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace highvoronoi::geometry {

/** @brief Select the built-in nanoflann KD-tree backend. */
struct KDSearch {
    std::size_t leaf_max_size = 10;
    unsigned int build_thread_count = 1;
};

/** @brief Select the built-in exact brute-force backend. */
struct BruteForceSearch {};

/**
 * @brief User-extensible mapping from a keyword and mesh to a concrete tree.
 *
 * A specialization must provide `using type = ...` and a static
 * `make(Mesh&, const Keyword&)` function. No central enum or registration list
 * is required.
 */
template <class Keyword, class Mesh, class Enable = void>
struct SearchTreeFactory;

template <class Mesh>
struct SearchTreeFactory<KDSearch, Mesh, void> {
    using type = NanoflannSearchTree<Mesh>;

    [[nodiscard]] static type make(
        Mesh& mesh,
        const KDSearch& keyword) {
        return type(
            mesh,
            keyword.leaf_max_size,
            keyword.build_thread_count);
    }
};

template <class Mesh>
struct SearchTreeFactory<BruteForceSearch, Mesh, void> {
    using type = BruteForceSearchTree<Mesh>;

    [[nodiscard]] static type make(
        Mesh& mesh,
        const BruteForceSearch&) {
        return type(mesh);
    }
};

template <class Keyword, class Mesh, class = void>
struct IsSearchTreeKeyword : std::false_type {};

template <class Keyword, class Mesh>
struct IsSearchTreeKeyword<
    Keyword,
    Mesh,
    std::void_t<
        typename SearchTreeFactory<
            std::decay_t<Keyword>,
            Mesh>::type,
        decltype(SearchTreeFactory<
                 std::decay_t<Keyword>,
                 Mesh>::make(
                     std::declval<Mesh&>(),
                     std::declval<const std::decay_t<Keyword>&>()))>>
    : std::true_type {};

template <class Keyword, class Mesh>
inline constexpr bool IsSearchTreeKeywordV =
    IsSearchTreeKeyword<Keyword, Mesh>::value;

template <class Keyword, class Mesh>
using SearchTreeFor = typename SearchTreeFactory<
    std::decay_t<Keyword>,
    Mesh>::type;

/** @brief Construct the concrete tree selected by a built-in/custom keyword. */
template <class Keyword, class Mesh>
[[nodiscard]] SearchTreeFor<Keyword, Mesh> make_search_tree(
    Mesh& mesh,
    const Keyword& keyword) {
    static_assert(
        IsSearchTreeKeywordV<Keyword, Mesh>,
        "No SearchTreeFactory specialization exists for this keyword and mesh.");

    return SearchTreeFactory<
        std::decay_t<Keyword>,
        Mesh>::make(mesh, keyword);
}

} // namespace highvoronoi::geometry
