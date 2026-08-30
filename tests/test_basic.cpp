#include <cassert>
#include <string_view>

#include <highvoronoi/highvoronoi.hpp>

int main()
{
    assert(highvoronoi::version() == std::string_view{"0.1.0"});
    return 0;
}
