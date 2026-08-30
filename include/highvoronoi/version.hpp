#pragma once

/**
 * @file version.hpp
 * @brief Small compiled-library interface shared by all public entry headers.
 *
 * This header deliberately contains no template-heavy geometry includes.
 * Source files that only need the library version should include this file
 * instead of the complete HighVoronoi umbrella.
 */

#include <string_view>

#if defined(_WIN32) && defined(HIGHVORONOI_SHARED_LIBRARY)
    #if defined(HIGHVORONOI_BUILDING_LIBRARY)
        #define HIGHVORONOI_API __declspec(dllexport)
    #else
        #define HIGHVORONOI_API __declspec(dllimport)
    #endif
#else
    #define HIGHVORONOI_API
#endif

namespace highvoronoi {

/** Return the semantic version of the compiled HighVoronoi library. */
[[nodiscard]]
HIGHVORONOI_API std::string_view version() noexcept;

} // namespace highvoronoi
