# Incremental examples

These examples demonstrate the mutation workflows that are intentionally more
important than the one-shot construction examples.

## Ordinary Voronoi: refine and remove

`voronoi_refine_remove.cpp`

1. builds a bounded 3D `VoronoiMesh`;
2. computes it with `ComputeVoronoi`;
3. appends four generators through `RefineVoronoi`;
4. removes two generators through `RemoveVoronoi`;
5. checks topological completeness after every stage.

Public entry header:

```cpp
#include <highvoronoi/voronoi.hpp>
```

## HighVoronoi: refine and remove

`high_voronoi_refine_remove.cpp`

1. appends an initial visible node block to a non-periodic `HighVoronoiMesh`;
2. integrates it with `ComputeHighVoronoi`;
3. appends another visible block and calls `ComputeHighVoronoi` again;
4. marks two visible nodes deleted with `erase_visible_nodes`;
5. calls `ComputeHighVoronoi` once more to repair the deletion.

The example deliberately uses the public HighVoronoi workflow rather than
constructing `RefineVoronoi` and `RemoveVoronoi` directly: pending visible
insertions/deletions are owned and orchestrated by `ComputeHighVoronoi`.

Public entry header:

```cpp
#include <highvoronoi/high_voronoi.hpp>
```

## Periodic HighVoronoi refinement

`periodic_high_voronoi_refine.cpp`

The visible domain is `[0,1]^3`, periodic in x and y and non-periodic in z.

1. 28 visible generators are inserted and fully periodized;
2. the example shows that invisible internal reference nodes were created;
3. three generators close to periodic faces are appended;
4. `ComputeHighVoronoi` performs incremental refinement and periodic closure;
5. `VisibleFirstMesh` plus `verify_mesh_complete_cells` checks the visible-cell
   topology;
6. every public HighVoronoi vertex is checked to lie in the visible domain.

This example is intentionally smaller than the full periodic reference
regression test: it demonstrates the public workflow rather than reimplementing
an independent 3x3 reference tiling.
