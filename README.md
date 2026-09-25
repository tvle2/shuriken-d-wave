# Shuriken D-Wave

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



