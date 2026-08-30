#pragma once

/**
 * @file high_voronoi.hpp
 * @brief Public entry point for incremental and periodic HighVoronoi meshes.
 *
 * This header exposes the persistent HighVoronoi owner, its complete
 * incremental/periodic compute orchestration, the visible-first read view and
 * common validation helpers.
 *
 * The current implementation still shares the ordinary incremental backend
 * infrastructure internally. Therefore some classical implementation headers
 * are transitively visible today even though VoronoiMesh is not part of this
 * entry header's intended public surface.
 */

#include <highvoronoi/core.hpp>

#include <highvoronoi/geometry/high_voronoi_mesh.hpp>
#include <highvoronoi/geometry/compute_high_voronoi.hpp>
#include <highvoronoi/geometry/visible_first_mesh.hpp>

#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/mesh_cell_validation.hpp>
