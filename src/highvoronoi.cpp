
#include <highvoronoi/highvoronoi.hpp>

namespace highvoronoi {

std::string_view version() noexcept
{
    return "0.1.0";
}

std::size_t compiled_example(std::size_t value) noexcept
{
    return value + 1;
}

} // namespace highvoronoi
