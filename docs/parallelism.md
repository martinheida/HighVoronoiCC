# Parallel computation

HighVoronoi internally supports two independent construction-parallel axes, but the convenience API intentionally exposes only the currently recommended one at Level 1.

## Level 1: mesh-level parallelism

```cpp
auto mesh = highvoronoi::voronoi_mesh<Dim>(
    points,
    point_count,
    boundary,
    highvoronoi::MultiThread{4});

highvoronoi::compute(mesh);
```

The same form works for `high_voronoi_mesh`.

Level 1 interprets `MultiThread{N}` as **mesh-level parallelism**. The facade also selects the compatible storage automatically:

```text
persistent database lock   ReadWriteLock
persistent database hash   StaticHash<16>
edge hash                  StaticHash<16>
cell-local queue hash      DirectHash
```

This matches the current project guidance: the 5D benchmark shows useful mesh-level scaling, while cast-worker parallelism is supported but not currently a performance win in that workload.

The count is stored by the convenience mesh and reused by `compute`, `refine`, and `remove`. Override it for one call when desired:

```cpp
highvoronoi::compute(mesh, 8);
highvoronoi::refine(mesh, added, added_count, 8);
highvoronoi::remove(mesh, {Mesh::Index{2}}, 8);
```

For a `SingleThread` convenience mesh, an override is ignored; its native database deliberately uses `EmptyLock` and the type does not silently become a parallel mesh later.

## Why the persistent/edge hashes are sharded

The parallel examples have established a useful distinction:

- persistent database and edge state can benefit from independently locked shards;
- the vertex queue already owns an outer queue synchronization point around claim/pop/reset, so sharding its internal duplicate table does not remove that outer bottleneck.

That is why the Level-1 parallel facade uses `StaticHash<16>` for database/edge state but retains `DirectHash` for the queue.

See [Hashing and storage](configuration.md#hashing-and-storage) before changing the shard count merely because a larger number looks more parallel.

## Level 2

`VoronoiConfig<MultiThread,...>` lets you change ray casting, search parameters, hash generator, and persistent/edge container types while preserving the simple mesh-level threading model.

Example:

```cpp
using Config = highvoronoi::VoronoiConfig<
    highvoronoi::MultiThread,
    highvoronoi::CombinedRaycast,
    highvoronoi::FNV64_128HashGenerator,
    highvoronoi::StaticHash<32>,
    highvoronoi::StaticHash<32>>;
```

This is normally the right level for a user who knows that lock contention or hash layout is the issue but does not need to own `HVDataBase`/`ComputeVoronoi` manually.

## Level 3: both threading axes

The native `ComputeVoronoi` interface exposes:

```text
MeshThreading
    one or several communicating reordered mesh branches

CastThreading
    one or several geometry workers inside each branch
```

Mesh-level example:

```cpp
using Compute = highvoronoi::ComputeVoronoi<
    Mesh,
    RayCaster,
    highvoronoi::MultiThread,
    highvoronoi::SingleThread,
    QueueParameters,
    EdgeParameters>;
```

Cast-level example:

```cpp
using Compute = highvoronoi::ComputeVoronoi<
    Mesh,
    RayCaster,
    highvoronoi::SingleThread,
    highvoronoi::MultiThread,
    QueueParameters,
    EdgeParameters>;
```

Both axes may be multithreaded simultaneously. Use this only after profiling; maximizing both counts is not automatically faster.

Any native parallel `ComputeVoronoi` requires a persistent database with `ReadWriteLock`. The convenience API enforces this by construction through the wrapper type.

## KD-tree build threads are separate

`KDSearch{leaf_size, build_threads}` configures search-tree building, not Voronoi mesh/cast worker counts. Treat it as a separate performance parameter.

## Parallel integration

Integration has its own `ParallelIntegrationExecution` policy and is independent of construction threading. The simple integration API is a later facade; the current detailed interface is described in [Integration](integration.md).

## Validation

After changing threading or sharding, compare against a validated serial result and use the completeness validators. A per-vertex consistency check alone cannot reveal missing topology.
