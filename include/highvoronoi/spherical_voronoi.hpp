#pragma once

/** @file spherical_voronoi.hpp
 *  @brief Public entry point for spherical and antipodal Voronoi meshes.
 *
 *  Like the ordinary `voronoi.hpp` and `high_voronoi.hpp` entry headers,
 *  this header imports the common public core first.  Users can therefore
 *  construct the database/parameter types required by `SphereVoronoiMesh`
 *  without depending on transitive implementation-header includes.
 */

#include <highvoronoi/core.hpp>
#include <highvoronoi/mesh/spherical_voronoi_mesh.hpp>
