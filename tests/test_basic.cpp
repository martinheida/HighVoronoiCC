
#include <cassert>
#include <string_view>

#include <highvoronoi/highvoronoi.hpp>

int main()
{
    // Test für Template-Code:
    static_assert(highvoronoi::square(4) == 16);

    // Tests für tatsächlich kompilierten Code:
    assert(highvoronoi::version() == std::string_view{"0.1.0"});
    assert(highvoronoi::compiled_example(41) == 42);

    return 0;
}
