
#pragma once

/**
 * @file high_voronoi.hpp
 * @brief Public entry point for incremental and periodic HighVoronoi meshes.
 *
 * The Level-1/Level-2 facade is exposed as high_voronoi_mesh(), compute(),
 * refine(), and remove(). The persistent HighVoronoiMesh /
 * ComputeHighVoronoi stack remains the Level-3 full-control API.
 *
 * This header also exposes the visible-first read view and common validation
 * helpers.
 *
 * The current implementation still shares the ordinary incremental backend
 * infrastructure internally. Therefore some classical implementation headers
 * are transitively visible today even though VoronoiMesh is not part of this
 * entry header's intended public surface.
 */

#include <highvoronoi/core.hpp>

#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/algorithm/high_voronoi/compute_high_voronoi.hpp>
#include <highvoronoi/mesh/visible_first_mesh.hpp>

#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/mesh/validation/mesh_cell_validation.hpp>

#include <highvoronoi/highvoronoi_api.hpp>
