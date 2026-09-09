#pragma once

/**
 * @file mesh_engines.hpp
 * @brief Optional computed/hybrid mesh engines.
 *
 * Kept separate from the ordinary and HighVoronoi entry headers so users that
 * provide explicitly stored generator nodes do not pay the parse cost of the
 * computed-mesh infrastructure.
 */

#include <highvoronoi/core.hpp>

#include <highvoronoi/mesh/engine/compute_mesh_engine.hpp>
#include <highvoronoi/mesh/engine/compute_mesh.hpp>
#include <highvoronoi/mesh/engine/cuboid_mesh_engine.hpp>
