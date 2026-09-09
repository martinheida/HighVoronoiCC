
#pragma once

/**
 * @file voronoi.hpp
 * @brief Public entry point for ordinary Euclidean Voronoi construction.
 *
 * The Level-1/Level-2 facade is exposed as voronoi_mesh(), compute(),
 * refine(), remove(), and VoronoiConfig. The existing VoronoiMesh /
 * ComputeVoronoi stack remains the Level-3 full-control API.
 *
 * This entry point intentionally does not include HighVoronoiMesh or the
 * incremental periodic HighVoronoi orchestration.
 */

#include <highvoronoi/core.hpp>

#include <highvoronoi/mesh/voronoi_mesh.hpp>

#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/remove_voronoi.hpp>

#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/mesh/validation/mesh_cell_validation.hpp>

#include <highvoronoi/voronoi_api.hpp>
