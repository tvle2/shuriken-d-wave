# Shuriken D-Wave Mapping

Standalone project for mapping and smoke-testing the open-boundary Shuriken
(square-kagome) Ising model on D-Wave QPUs.

This project is intentionally separate from `dwave-gibb-sampling`, but reuses
its code organization and D-Wave helper logic.

## Scope: Gate 0A

The first question is purely graph-theoretic/hardware-connectivity:

> Can an open Shuriken interaction graph be represented with one physical
> qubit per logical spin on the live `Advantage2_system4` working graph?

The mapped longitudinal Hamiltonian is

```text
H_Z = J_AA sum_AA Z_i Z_j
    + x_x J_AA sum_ABx Z_i Z_j
    + x_y J_AA sum_ABy Z_i Z_j
```

with no programmed B-B couplers and no longitudinal fields.

For an open `Lx x Ly` patch:

```text
N = 6 Lx Ly - Lx - Ly
E = 12 Lx Ly - 4 Lx - 4 Ly
```

For the main `Z6` target, `Lx = Ly = 6`:

```text
N = 204 logical spins
E = 384 logical couplers
```

## Project structure

```text
shuriken-dwave/
├── README.md
├── requirements.txt
├── embeddings/
├── data/
├── src/
│   ├── helpers.py
│   ├── shuriken.py
│   ├── find_embeddings.py
│   └── sample_dwave.py
└── tests/
    └── test_shuriken.py
```

`helpers.py` is copied from the existing `dwave-gibb-sampling` project so this
new folder is self-contained.  The connection, result-saving, and embedding-file
conventions are therefore preserved without importing the old repository.

## Installation

Create a new environment if desired:

```bash
cd shuriken-dwave
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure your Leap credentials in the same way as for the existing project so
that `Client.from_config()` can find them.

## Step 1: local logic checks

```bash
python -m pytest tests/test_shuriken.py
```

If `pytest` is not installed, either install it or skip this step; the mapping
scripts also perform internal consistency checks.

The tests verify the logical node/edge counts and all four closed-form `k`
tracks for square patches `2x2` through `6x6` on ideal `Z6`.

## Step 2: certify the live Advantage2_system4 graph

```bash
python src/find_embeddings.py
```

The script tests:

```text
6x6, 5x5, 4x4, 3x3, 2x2
```

For each size it:

1. builds the exact logical Shuriken graph;
2. enumerates translated closed-form Zephyr placements;
3. self-checks the analytic Zephyr construction;
4. verifies every mapped qubit on the live working graph;
5. verifies every required live coupler;
6. records extra hardware couplers that must be programmed to zero;
7. selects vertex-disjoint valid copies;
8. optionally tries a native `minorminer.subgraph` fallback if every analytic
   placement is blocked by hardware defects;
9. saves the embedding and a graph-specific JSON certificate.

Expected files include

```text
embeddings/
  Advantage2_system4_<graph-id>_204.txt
  Advantage2_system4_<graph-id>_shuriken_L6x6_certificate.json
  Advantage2_system4_<graph-id>_shuriken_summary.json
```

## Step 3: optional QPU smoke test

After choosing a patch size that has a live certificate, set `LX` and `LY` in
`src/sample_dwave.py`, then run

```bash
python src/sample_dwave.py
```

This submits only a small native/classical smoke test.  It is **not** the final
quantum-mediated-square-ice experiment: there is no B-sublattice chain encoding
or sublattice-selective anneal-offset control yet.

## Reference longitudinal parameters

The smoke test currently uses the proposal's representative rectangular point:

```text
J_AA = 1.0
x_x = 0.68922
x_y = 0.95
```

These numbers do not affect graph embeddability.

## Next gate

After Gate 0A succeeds, the next project is Gate 0B:

```text
A logical spins -> one physical qubit each
B logical spins -> uniform ferromagnetic chains and/or B anneal offsets
```

The purpose is to suppress the effective B-sublattice transverse field while
retaining the Shuriken longitudinal interaction geometry.

## Live-QPU defect repair

A large rigid fixed-`k` Zephyr block is intentionally only the first diagnostic.
On a fabricated working graph, even a few-percent qubit defect rate makes a
204-qubit defect-free rigid block unlikely. `find_embeddings.py` therefore uses
this search order:

1. rigid translated fixed-`k` blocks;
2. **local `k`-track repair** with one physical qubit per logical spin;
3. arbitrary `minorminer.subgraph` native fallback only if structured repair fails.

The local repair is still a native embedding: it introduces no chains. Each
logical site retains its analytic Zephyr location `(u,w,j,z)` but can choose one
of the four `k` tracks. Compatibility is checked directly against the live QPU
working graph and solved as a small-domain constraint-satisfaction problem.

Select a QPU without editing the source, for example:

```bash
DWAVE_QPU=Advantage2_system1 python src/find_embeddings.py
```
