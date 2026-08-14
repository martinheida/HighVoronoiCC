# Parallel construction examples

These examples demonstrate the two independent parallelization axes of
`ComputeVoronoi`.

- `parallel_cast.cpp`: one mesh branch with several `VoronoiWorker` threads.
- `parallel_mesh.cpp`: several communicating mesh branches with one worker each.
- `parallel_combined.cpp`: both parallelization axes enabled simultaneously.

The three files deliberately use the same deterministic 3D point cloud and the
same geometric/search configuration.  The important difference is therefore
visible directly in the two threading arguments passed to `ComputeVoronoi`.

## Shared-state configuration

Parallel construction uses `ReadWriteLock` for the persistent `HVDataBase`.
The examples also use `StaticHash<16>` for the persistent database hash and the
cell-local EdgeHash.  Their independent sub-table locks reduce contention when
operations are distributed among different sub-tables.

The cell-local vertex queue uses `DirectHash`.  `VoronoiVertexQueue` currently
has an outer queue lock around claim/pop/reset, so changing its internal hash to
`StaticHash<N>` does not remove that outer synchronization point.

## Thread-count choice

`MeshThreading` and `CastThreading` are independent.  A larger number is not
automatically faster.  In particular, worker-level parallelism can temporarily
lose utilization when the shared branch queue is empty while another worker is
still computing a vertex that may generate new work.  The examples therefore
show configuration, not a claim about the optimal thread split.
