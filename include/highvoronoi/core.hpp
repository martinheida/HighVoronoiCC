#pragma once

/**
 * @file core.hpp
 * @brief Common public types shared by Voronoi and HighVoronoi construction.
 *
 * This layer contains no complete Voronoi construction algorithm. It is useful
 * for future modules such as spherical Voronoi geometry or integral evaluation
 * that need the common point, boundary, node, parameter and database types
 * without importing every current construction backend.
 */

#include <highvoronoi/version.hpp>
#include <highvoronoi/database.hpp>
#include <highvoronoi/parameters.hpp>

#include <highvoronoi/geometry/point.hpp>
#include <highvoronoi/geometry/boundary.hpp>
#include <highvoronoi/geometry/voronoi_nodes.hpp>
