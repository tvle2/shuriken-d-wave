import networkx as nx


DEFAULT_J_AA = 1.0
DEFAULT_X_X = 0.68922
DEFAULT_X_Y = 0.95


def shuriken_size(lx, ly):
    """Return (number_of_spins, number_of_couplers) for an open patch."""
    if lx < 1 or ly < 1:
        raise ValueError("lx and ly must be positive")

    n = 6 * lx * ly - lx - ly
    e = 12 * lx * ly - 4 * lx - 4 * ly
    return n, e


def create_shuriken(lx, ly):
    """
    Build the open-boundary Shuriken (square-kagome) logical graph.

    Cell (i, j) contains four A spins labelled clockwise
        A0 = NW, A1 = NE, A2 = SE, A3 = SW.

    Bx(i,j) lies between cells (i,j) and (i+1,j) and couples to
        A1,A2 of the left cell and A0,A3 of the right cell.

    By(i,j) lies between cells (i,j) and (i,j+1) and couples to
        A0,A1 of the lower cell and A2,A3 of the upper cell.

    Returns
    -------
    G : networkx.Graph
    logical_to_name : dict[int, tuple]
    name_to_logical : dict[tuple, int]
    edge_types : dict[tuple[int, int], str]
        Edge labels are "AA", "ABx", and "ABy".
    """
    if lx < 1 or ly < 1:
        raise ValueError("lx and ly must be positive")

    names = []

    for j in range(ly):
        for i in range(lx):
            for a in range(4):
                names.append(("A", i, j, a))

    # Horizontal mediator spins.
    for j in range(ly):
        for i in range(lx - 1):
            names.append(("Bx", i, j))

    for j in range(ly - 1):
        for i in range(lx):
            names.append(("By", i, j))

    logical_to_name = {logical: name for logical, name in enumerate(names)}
    name_to_logical = {name: logical for logical, name in logical_to_name.items()}

    graph = nx.Graph()
    graph.add_nodes_from(logical_to_name)
    edge_types = {}

    def add_edge(name_u, name_v, edge_type):
        u = name_to_logical[name_u]
        v = name_to_logical[name_v]
        edge = tuple(sorted((u, v)))
        graph.add_edge(*edge)
        edge_types[edge] = edge_type

    # Four AA perimeter bonds on every square.
    for j in range(ly):
        for i in range(lx):
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                add_edge(("A", i, j, a), ("A", i, j, b), "AA")

    # Horizontal Bx sites.
    for j in range(ly):
        for i in range(lx - 1):
            b = ("Bx", i, j)
            add_edge(b, ("A", i, j, 1), "ABx")
            add_edge(b, ("A", i, j, 2), "ABx")
            add_edge(b, ("A", i + 1, j, 0), "ABx")
            add_edge(b, ("A", i + 1, j, 3), "ABx")

    # Vertical By sites.
    for j in range(ly - 1):
        for i in range(lx):
            b = ("By", i, j)
            add_edge(b, ("A", i, j, 0), "ABy")
            add_edge(b, ("A", i, j, 1), "ABy")
            add_edge(b, ("A", i, j + 1, 2), "ABy")
            add_edge(b, ("A", i, j + 1, 3), "ABy")

    expected_n, expected_e = shuriken_size(lx, ly)
    expected_counts = {
        "AA": 4 * lx * ly,
        "ABx": 4 * (lx - 1) * ly,
        "ABy": 4 * lx * (ly - 1),
    }
    actual_counts = {
        edge_type: sum(value == edge_type for value in edge_types.values())
        for edge_type in expected_counts
    }

    assert graph.number_of_nodes() == expected_n
    assert graph.number_of_edges() == expected_e
    assert actual_counts == expected_counts

    return graph, logical_to_name, name_to_logical, edge_types


def shuriken_couplings(
    lx,
    ly,
    j_aa=DEFAULT_J_AA,
    x_x=DEFAULT_X_X,
    x_y=DEFAULT_X_Y,
):
    """
    Return logical Ising couplings for

        H_Z = J_AA sum_AA Z_i Z_j
            + x_x J_AA sum_ABx Z_i Z_j
            + x_y J_AA sum_ABy Z_i Z_j.

    D-Wave convention: positive J is antiferromagnetic.
    No B-B couplings and no longitudinal fields are added.
    """
    _, logical_to_name, name_to_logical, edge_types = create_shuriken(lx, ly)

    couplings = {}
    for edge, edge_type in edge_types.items():
        if edge_type == "AA":
            coupling = j_aa
        elif edge_type == "ABx":
            coupling = x_x * j_aa
        elif edge_type == "ABy":
            coupling = x_y * j_aa
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")

        couplings[edge] = coupling

    return couplings, logical_to_name, name_to_logical, edge_types


def zephyr_coordinate_to_linear(coord, m, t=4):
    """
    Convert a Zephyr coordinate (u,w,k,j,z) to D-Wave's linear index.

    Formula from the official Zephyr topology definition:
        q = (((u*(2m+1)+w)*t+k)*2+j)*m+z
    """
    u, w, k, j, z = coord

    if u not in (0, 1):
        raise ValueError(f"invalid Zephyr orientation u={u}")
    if not 0 <= w < 2 * m + 1:
        raise ValueError(f"invalid Zephyr w={w} for m={m}")
    if not 0 <= k < t:
        raise ValueError(f"invalid Zephyr k={k} for t={t}")
    if j not in (0, 1):
        raise ValueError(f"invalid Zephyr j={j}")
    if not 0 <= z < m:
        raise ValueError(f"invalid Zephyr z={z} for m={m}")

    return (((u * (2 * m + 1) + w) * t + k) * 2 + j) * m + z


def _is_ideal_zephyr_edge(coord_u, coord_v):
    """
    Check adjacency directly from the official Zephyr edge definitions.
    """
    if coord_u == coord_v:
        return False

    u1, w1, k1, j1, z1 = coord_u
    u2, w2, k2, j2, z2 = coord_v

    # External edge: same (u,w,k,j), adjacent z.
    if (u1, w1, k1, j1) == (u2, w2, k2, j2) and abs(z1 - z2) == 1:
        return True

    # Odd edge: (u,w,k,0,z) ~ (u,w,k,1,z-alpha), alpha in {0,1}.
    if u1 == u2 and w1 == w2 and k1 == k2 and {j1, j2} == {0, 1}:
        if j1 == 0:
            z_j0, z_j1 = z1, z2
        else:
            z_j0, z_j1 = z2, z1
        if z_j1 in (z_j0, z_j0 - 1):
            return True

    # Internal edge. Put the vertical coordinate first.
    if u1 == u2:
        return False
    if u1 == 1:
        coord_u, coord_v = coord_v, coord_u

    _, wv, _, jv, zv = coord_u
    _, wh, _, jh, zh = coord_v

    # Official form:
    # (0,2w+1-alpha,k,j,z-j*beta)
    #   ~ (1,2z+1-beta,h,i,w-i*alpha)
    for alpha in (0, 1):
        w_numerator = wv - 1 + alpha
        if w_numerator % 2:
            continue
        w = w_numerator // 2

        for beta in (0, 1):
            z_numerator = wh - 1 + beta
            if z_numerator % 2:
                continue
            z = z_numerator // 2

            if zv == z - jv * beta and zh == w - jh * alpha:
                return True

    return False


def analytic_zephyr_embedding(
    lx,
    ly,
    m,
    k,
    t=4,
    x_offset=0,
    y_offset=0,
):
    """
    Closed-form native Shuriken embedding into ideal Z_{m,t}.
    """
    if x_offset < 0 or y_offset < 0:
        raise ValueError("offsets must be nonnegative")
    if x_offset + lx > m or y_offset + ly > m:
        raise ValueError(
            f"patch {lx}x{ly} with offset ({x_offset},{y_offset}) "
            f"does not fit inside Zephyr m={m}"
        )
    if not 0 <= k < t:
        raise ValueError(f"k={k} is outside 0..{t - 1}")

    logical_graph, logical_to_name, _, _ = create_shuriken(lx, ly)
    mapping = {}
    coordinates = {}

    for logical, name in logical_to_name.items():
        site_type = name[0]

        if site_type == "A":
            _, i, j, a = name
            I = i + x_offset
            J = j + y_offset
            parity = (I + J) % 2

            if parity == 0:
                table = {
                    0: (0, 2 * I + 1, k, 1, J),
                    1: (1, 2 * J + 2, k, 1, I),
                    2: (0, 2 * I + 2, k, 1, J),
                    3: (1, 2 * J + 1, k, 1, I),
                }
            else:
                table = {
                    0: (1, 2 * J + 2, k, 1, I),
                    1: (0, 2 * I + 2, k, 1, J),
                    2: (1, 2 * J + 1, k, 1, I),
                    3: (0, 2 * I + 1, k, 1, J),
                }
            coord = table[a]

        elif site_type == "Bx":
            _, i, j = name
            I = i + x_offset
            J = j + y_offset
            parity = (I + J) % 2
            if parity == 0:
                coord = (1, 2 * J + 2, k, 0, I + 1)
            else:
                coord = (1, 2 * J + 1, k, 0, I + 1)

        elif site_type == "By":
            _, i, j = name
            I = i + x_offset
            J = j + y_offset
            parity = (I + J) % 2
            if parity == 0:
                coord = (0, 2 * I + 1, k, 0, J + 1)
            else:
                coord = (0, 2 * I + 2, k, 0, J + 1)

        else:
            raise ValueError(f"Unknown Shuriken site type: {site_type}")

        coordinates[logical] = coord
        mapping[logical] = zephyr_coordinate_to_linear(coord, m=m, t=t)

    if len(set(mapping.values())) != len(mapping):
        raise ValueError("closed-form Zephyr map is not injective")

    # Check every required logical edge against the topology formula itself.
    invalid_ideal_edges = [
        (u, v, coordinates[u], coordinates[v])
        for u, v in logical_graph.edges()
        if not _is_ideal_zephyr_edge(coordinates[u], coordinates[v])
    ]
    if invalid_ideal_edges:
        raise ValueError(
            "closed-form map failed the ideal-Zephyr edge check; first failure: "
            f"{invalid_ideal_edges[0]}"
        )

    return mapping, coordinates


def verify_native_embedding(logical_graph, hardware_graph, mapping):
    """Return a certificate for a one-physical-qubit-per-logical-spin map."""
    logical_nodes = set(logical_graph.nodes())
    if set(mapping) != logical_nodes:
        return {
            "valid": False,
            "reason": "mapping does not cover exactly the logical node set",
        }

    physical_nodes = list(mapping.values())
    duplicate_count = len(physical_nodes) - len(set(physical_nodes))
    missing_qubits = sorted(
        qubit for qubit in physical_nodes if qubit not in hardware_graph
    )

    required_physical_edges = set()
    missing_couplers = []
    for u, v in logical_graph.edges():
        pu, pv = mapping[u], mapping[v]
        edge = tuple(sorted((pu, pv)))
        required_physical_edges.add(edge)
        if not hardware_graph.has_edge(*edge):
            missing_couplers.append((u, v, pu, pv))

    used_qubits = set(physical_nodes)
    extra_couplers = sorted(
        tuple(sorted(edge))
        for edge in hardware_graph.subgraph(used_qubits).edges()
        if tuple(sorted(edge)) not in required_physical_edges
    )

    valid = (
        duplicate_count == 0
        and not missing_qubits
        and not missing_couplers
    )

    return {
        "valid": valid,
        "num_logical_nodes": logical_graph.number_of_nodes(),
        "num_logical_edges": logical_graph.number_of_edges(),
        "num_physical_qubits": len(set(physical_nodes)),
        "duplicate_physical_qubits": duplicate_count,
        "missing_qubits": missing_qubits,
        "missing_couplers": missing_couplers,
        "num_extra_couplers_to_zero": len(extra_couplers),
        "extra_couplers_to_zero": extra_couplers,
    }


def analytic_zephyr_track_domains(
    lx,
    ly,
    m,
    t=4,
    x_offset=0,
    y_offset=0,
):
    """
    Return the physical-qubit choices obtained by keeping the closed-form
    Shuriken geometry fixed while allowing each logical site to choose its
    Zephyr k-track independently.

    The fixed-k construction is an ideal-topology embedding.  On a fabricated
    working graph, a large fixed-k patch is very likely to hit at least one
    disabled qubit.  Because Zephyr has t parallel k-tracks, many such defects
    can be repaired locally without introducing chains.

    Returns
    -------
    physical_by_logical : dict[int, dict[int, int]]
        physical_by_logical[v][k] gives the linear Zephyr qubit index.
    coordinate_by_logical : dict[int, dict[int, tuple]]
        coordinate_by_logical[v][k] gives the corresponding (u,w,k,j,z).
    """
    # Use k=0 to obtain the topology location for every logical site.  The
    # ideal construction is already self-checked by analytic_zephyr_embedding.
    _, base_coordinates = analytic_zephyr_embedding(
        lx=lx,
        ly=ly,
        m=m,
        t=t,
        k=0,
        x_offset=x_offset,
        y_offset=y_offset,
    )

    physical_by_logical = {}
    coordinate_by_logical = {}

    for logical, coord0 in base_coordinates.items():
        u, w, _, j, z = coord0
        coordinate_by_logical[logical] = {}
        physical_by_logical[logical] = {}
        for k in range(t):
            coord = (u, w, k, j, z)
            coordinate_by_logical[logical][k] = coord
            physical_by_logical[logical][k] = zephyr_coordinate_to_linear(
                coord, m=m, t=t
            )

    # Base topology locations must be distinct.  If two logical sites shared
    # the same (u,w,j,z), they would collide for every common k.
    bases = [
        (coord[0], coord[1], coord[3], coord[4])
        for coord in base_coordinates.values()
    ]
    if len(set(bases)) != len(bases):
        raise ValueError("closed-form Shuriken map has duplicate Zephyr bases")

    return physical_by_logical, coordinate_by_logical


def _ac3_domains(logical_graph, compatibility, domains, initial_arcs=None):
    """AC-3 constraint propagation for the local-k repair CSP."""
    from collections import deque

    if initial_arcs is None:
        queue = deque((u, v) for u, v in logical_graph.edges())
        queue.extend((v, u) for u, v in logical_graph.edges())
    else:
        queue = deque(initial_arcs)

    while queue:
        u, v = queue.popleft()
        allowed = compatibility[(u, v)]
        old_domain = domains[u]
        new_domain = {
            ku
            for ku in old_domain
            if any(kv in allowed.get(ku, set()) for kv in domains[v])
        }

        if new_domain == old_domain:
            continue

        if not new_domain:
            return False

        domains[u] = new_domain
        for w in logical_graph.neighbors(u):
            if w != v:
                queue.append((w, u))

    return True


def repair_zephyr_embedding_with_local_k(
    logical_graph,
    hardware_graph,
    lx,
    ly,
    m,
    t=4,
    x_offset=0,
    y_offset=0,
):
    """
    Repair a translated closed-form Zephyr placement by choosing k locally.

    No chains are introduced: every logical spin is still represented by one
    physical qubit.  Only the k coordinate may vary from site to site.

    The problem is a small-domain CSP.  Each logical site has at most t=4
    candidate physical qubits.  For every logical edge, allowed (k_u,k_v)
    pairs are read directly from the live QPU working graph.  AC-3 plus
    backtracking finds a consistent assignment if one exists.

    Returns
    -------
    None if no local-k repair exists for this translation, otherwise a dict
    with keys mapping, coordinates, k_assignment, and diagnostics.
    """
    physical_by_logical, coordinate_by_logical = analytic_zephyr_track_domains(
        lx=lx,
        ly=ly,
        m=m,
        t=t,
        x_offset=x_offset,
        y_offset=y_offset,
    )

    domains = {}
    for v in logical_graph.nodes():
        active_k = {
            k
            for k, physical in physical_by_logical[v].items()
            if physical in hardware_graph
        }
        if not active_k:
            return None
        domains[v] = active_k

    compatibility = {}
    for u, v in logical_graph.edges():
        uv = {}
        vu = {}
        for ku in domains[u]:
            pu = physical_by_logical[u][ku]
            for kv in domains[v]:
                pv = physical_by_logical[v][kv]
                if hardware_graph.has_edge(pu, pv):
                    uv.setdefault(ku, set()).add(kv)
                    vu.setdefault(kv, set()).add(ku)
        compatibility[(u, v)] = uv
        compatibility[(v, u)] = vu

    work_domains = {v: set(values) for v, values in domains.items()}
    if not _ac3_domains(logical_graph, compatibility, work_domains):
        return None

    # Track a rough search effort for the certificate/log.
    search_nodes = 0

    def recurse(current_domains):
        nonlocal search_nodes
        search_nodes += 1

        unresolved = [v for v, d in current_domains.items() if len(d) > 1]
        if not unresolved:
            return {v: next(iter(d)) for v, d in current_domains.items()}

        # Minimum remaining values, then highest degree.  This tends to expose
        # constrained odd-coupler/equality regions early.
        v = min(
            unresolved,
            key=lambda node: (len(current_domains[node]), -logical_graph.degree[node]),
        )

        # Least-constraining-value ordering.
        def value_score(k):
            score = 0
            for nbr in logical_graph.neighbors(v):
                allowed = compatibility[(v, nbr)].get(k, set())
                score += len(allowed.intersection(current_domains[nbr]))
            return -score

        for k in sorted(current_domains[v], key=value_score):
            trial = {node: set(values) for node, values in current_domains.items()}
            trial[v] = {k}
            arcs = [(nbr, v) for nbr in logical_graph.neighbors(v)]
            if not _ac3_domains(logical_graph, compatibility, trial, initial_arcs=arcs):
                continue
            solution = recurse(trial)
            if solution is not None:
                return solution

        return None

    assignment = recurse(work_domains)
    if assignment is None:
        return None

    mapping = {
        v: physical_by_logical[v][assignment[v]]
        for v in logical_graph.nodes()
    }
    coordinates = {
        v: coordinate_by_logical[v][assignment[v]]
        for v in logical_graph.nodes()
    }

    verification = verify_native_embedding(logical_graph, hardware_graph, mapping)
    if not verification["valid"]:
        raise RuntimeError("local-k CSP returned a mapping that failed verification")

    k_histogram = {k: 0 for k in range(t)}
    for k in assignment.values():
        k_histogram[k] += 1

    return {
        "mapping": mapping,
        "coordinates": coordinates,
        "k_assignment": assignment,
        "k_histogram": k_histogram,
        "search_nodes": search_nodes,
        "verification": verification,
    }
