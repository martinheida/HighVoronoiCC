#pragma once

/**
 * @file voronoi.hpp
 * @brief Public entry point for ordinary Euclidean Voronoi construction.
 *
 * Include this header when only the classical VoronoiMesh / ComputeVoronoi
 * path is required. It intentionally does not include HighVoronoiMesh or the
 * incremental periodic HighVoronoi orchestration.
 */

#include <highvoronoi/core.hpp>

#include <highvoronoi/geometry/voronoi_mesh.hpp>

#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/refine_voronoi.hpp>
#include <highvoronoi/geometry/remove_voronoi.hpp>

#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/mesh_cell_validation.hpp>
