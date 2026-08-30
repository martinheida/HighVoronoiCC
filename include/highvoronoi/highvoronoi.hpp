#pragma once

/**
 * @file highvoronoi.hpp
 * @brief Convenience umbrella for the complete public HighVoronoi API.
 *
 * Prefer a narrower thematic entry header when possible:
 *
 * @code{.cpp}
 * #include <highvoronoi/voronoi.hpp>       // ordinary Euclidean Voronoi
 * #include <highvoronoi/high_voronoi.hpp>  // incremental/periodic HighVoronoi
 * #include <highvoronoi/mesh_engines.hpp>  // optional computed mesh engines
 * @endcode
 *
 * This header intentionally imports all currently supported public modules.
 */

#include <highvoronoi/voronoi.hpp>
#include <highvoronoi/high_voronoi.hpp>
#include <highvoronoi/mesh_engines.hpp>
