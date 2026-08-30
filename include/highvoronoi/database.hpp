#pragma once

/**
 * @file database.hpp
 * @brief Public entry point for HighVoronoi vertex databases.
 *
 * The concrete implementations remain organized under detail/ because they are
 * low-level storage components. Users should include this public forwarding
 * header instead of depending on the physical detail/ path.
 */

#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/detail/hybrid_database.hpp>
