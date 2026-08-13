#include "highvoronoi/detail/hvview.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if(!(condition)) {                                                      \
            std::cerr << "FAILED: " #condition                                 \
                      << " at line " << __LINE__ << '\n';                     \
            ++failures;                                                         \
        }                                                                       \
    } while(false)

using Index = std::uint32_t;

void test_switch_view_zero_based()
{
    // Julia SwitchView(3, 5) corresponds to C++ SwitchView(2, 4).
    const highvoronoi::SwitchView<Index> view(2, 4);

    const std::array<Index, 7> forward_expected{3, 4, 0, 1, 2, 5, 6};
    const std::array<Index, 7> backward_expected{2, 3, 4, 0, 1, 5, 6};

    for(Index i = 0; i < forward_expected.size(); ++i) {
        CHECK((view * i) == forward_expected[i]);
        CHECK((view / i) == backward_expected[i]);
        CHECK((view / (view * i)) == i);
        CHECK((view * (view / i)) == i);
    }

    CHECK(view.first() == 2);
    CHECK(view.last() == 4);
}

void test_switch_view_vectors()
{
    const highvoronoi::SwitchView<Index> view(2, 4);
    const std::vector<Index> input{0, 1, 2, 3, 4, 5, 6};
    const std::vector<Index> expected{3, 4, 0, 1, 2, 5, 6};

    const auto allocated_output = view.forward(input);
    CHECK(allocated_output == expected);

    std::vector<Index> supplied_output(input.size());
    view.forward(input, supplied_output);
    CHECK(supplied_output == expected);

    const auto restored = view.backward(supplied_output);
    CHECK(restored == input);

    std::array<Index, 7> array_output{};
    view.forward(input, array_output);

    for(Index i = 0; i < input.size(); ++i)
        CHECK(array_output[i] == expected[i]);
}

void test_combined_view()
{
    const highvoronoi::SwitchView<Index> outer(1, 3);
    const highvoronoi::SwitchView<Index> inner(2, 4);
    const highvoronoi::CombinedView combined(outer, inner);

    for(Index i = 0; i < 8; ++i) {
        CHECK((combined * i) == (outer * (inner * i)));
        CHECK((combined / (combined * i)) == i);
        CHECK((combined * (combined / i)) == i);
    }
}

void test_shuffle_view()
{
    const std::vector<Index> external_indices{2, 0, 3, 1};
    const std::vector<Index> internal_indices{1, 3, 0, 2};
    const highvoronoi::ShuffleView view(external_indices, internal_indices);

    for(Index i = 0; i < 4; ++i) {
        CHECK((view * i) == external_indices[i]);
        CHECK((view / i) == internal_indices[i]);
        CHECK((view / (view * i)) == i);
    }

    CHECK((view * 4) == 4);
    CHECK((view / 4) == 4);
    CHECK((view * 20) == 20);
    CHECK((view / 20) == 20);
}

void test_shuffle_view_with_explicit_length()
{
    const std::array<Index, 4> external_indices{2, 0, 3, 1};
    const std::array<Index, 4> internal_indices{1, 3, 0, 2};
    const highvoronoi::ShuffleView view(
        external_indices,
        internal_indices,
        Index{2}
    );

    CHECK((view * 0) == 2);
    CHECK((view * 1) == 0);
    CHECK((view * 2) == 2);
    CHECK((view / 0) == 1);
    CHECK((view / 1) == 3);
    CHECK((view / 2) == 2);
}

void test_invalid_switch_range()
{
    bool threw = false;

    try {
        const highvoronoi::SwitchView<Index> invalid(5, 2);
        (void)invalid;
    }
    catch(const std::invalid_argument&) {
        threw = true;
    }

    CHECK(threw);
}

} // namespace

int main()
{
    test_switch_view_zero_based();
    test_switch_view_vectors();
    test_combined_view();
    test_shuffle_view();
    test_shuffle_view_with_explicit_length();
    test_invalid_switch_range();

    if(failures != 0) {
        std::cerr << failures << " hvview test(s) failed.\n";
        return 1;
    }

    std::cout << "All hvview tests passed.\n";
    return 0;
}
