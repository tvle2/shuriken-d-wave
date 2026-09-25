import json
import os
import random
from pathlib import Path

import minorminer.subgraph

import helpers as dh
from shuriken import (
    create_shuriken,
    repair_zephyr_embedding_with_local_k,
    verify_native_embedding,
)


SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
EMBEDDINGS_DIRECTORY = PROJECT_ROOT / "embeddings"
EMBEDDINGS_DIRECTORY.mkdir(parents=True, exist_ok=True)


QA_DEVICE = os.environ.get("DWAVE_QPU", "Advantage2_system1")

LX = int(os.environ.get("SHURIKEN_LX", "6"))
LY = int(os.environ.get("SHURIKEN_LY", str(LX)))


NUM_RESTARTS = int(os.environ.get("EMBED_RESTARTS", "20"))

USE_MINORMINER_FALLBACK = os.environ.get("USE_MINORMINER_FALLBACK", "1") != "0"
MINORMINER_TIMEOUT_SECONDS = int(os.environ.get("MINORMINER_TIMEOUT", "300"))
MINORMINER_TRIES_PER_FAILURE = int(os.environ.get("MINORMINER_TRIES", "2"))


MAX_COPIES = int(os.environ.get("MAX_COPIES", "1000"))
RANDOM_SEED = int(os.environ.get("EMBED_SEED", "7"))


def solver_identity_text(solver):
    identity = getattr(solver, "identity", None)
    if identity is not None:
        return str(identity)
    return f"{QA_DEVICE};graph_id={solver.graph_id}"


def solver_zephyr_shape(solver):
    topology = solver.properties.get("topology", {})
    topology_type = topology.get("type")
    shape = topology.get("shape")
    if topology_type != "zephyr":
        raise ValueError(
            f"{QA_DEVICE} reports topology={topology_type!r}; "
            "this structure-aware search currently expects Zephyr"
        )
    if not shape:
        raise ValueError("solver did not report a Zephyr topology shape")
    m = int(shape[0])
    t = int(shape[1]) if len(shape) > 1 else 4
    return m, t


def shuffled_translations(lx, ly, m, rng):
    translations = [
        (x_offset, y_offset)
        for x_offset in range(m - lx + 1)
        for y_offset in range(m - ly + 1)
    ]
    rng.shuffle(translations)
    return translations


def find_one_structured_embedding(logical_graph, remaining_graph, lx, ly, m, t, rng):
    """Find one native copy using the analytic Zephyr layout plus local k repair."""
    for x_offset, y_offset in shuffled_translations(lx, ly, m, rng):
        repaired = repair_zephyr_embedding_with_local_k(
            logical_graph=logical_graph,
            hardware_graph=remaining_graph,
            lx=lx,
            ly=ly,
            m=m,
            t=t,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        if repaired is None:
            continue

        return {
            "mapping": repaired["mapping"],
            "method": "zephyr_local_k_repair",
            "x_offset": x_offset,
            "y_offset": y_offset,
            "k_histogram": repaired["k_histogram"],
            "search_nodes": repaired["search_nodes"],
            "verification": repaired["verification"],
        }
    return None


def find_one_minorminer_embedding(logical_graph, remaining_graph):
    for _ in range(MINORMINER_TRIES_PER_FAILURE):
        mapping = minorminer.subgraph.find_subgraph(
            logical_graph,
            remaining_graph,
            timeout=MINORMINER_TIMEOUT_SECONDS,
        )
        if not mapping:
            continue

        verification = verify_native_embedding(logical_graph, remaining_graph, mapping)
        if not verification["valid"]:
            raise RuntimeError(
                "minorminer.subgraph returned a mapping that failed verification"
            )

        return {
            "mapping": mapping,
            "method": "minorminer.subgraph",
            "x_offset": None,
            "y_offset": None,
            "k_histogram": None,
            "search_nodes": None,
            "verification": verification,
        }
    return None


def find_parallel_embeddings(logical_graph, hardware_graph, lx, ly, m, t):
    n = logical_graph.number_of_nodes()
    vertex_upper_bound = hardware_graph.number_of_nodes() // n
    target_copies = min(vertex_upper_bound, MAX_COPIES)

    best = []
    restart_summaries = []

    for restart in range(NUM_RESTARTS):
        rng = random.Random(RANDOM_SEED + restart)
        remaining = hardware_graph.copy()
        current = []

        print(
            f"\nParallel restart {restart + 1}/{NUM_RESTARTS}: "
            f"starting with {remaining.number_of_nodes()} qubits"
        )

        while len(current) < target_copies and remaining.number_of_nodes() >= n:
            found = find_one_structured_embedding(
                logical_graph, remaining, lx, ly, m, t, rng
            )

            if found is None and USE_MINORMINER_FALLBACK:
                print("  structured search exhausted; trying minorminer fallback")
                found = find_one_minorminer_embedding(logical_graph, remaining)

            if found is None:
                break

            mapping = found["mapping"]
            physical_qubits = list(mapping.values())

            # Explicit safety checks before we mutate the residual graph.
            if len(physical_qubits) != n or len(set(physical_qubits)) != n:
                raise RuntimeError("embedding is not one-to-one")
            verification = verify_native_embedding(logical_graph, hardware_graph, mapping)
            if not verification["valid"]:
                raise RuntimeError("embedding failed verification on full working graph")

            current.append(found)
            remaining.remove_nodes_from(physical_qubits)

            print(
                f"  copy {len(current):2d}: method={found['method']}, "
                f"remaining={remaining.number_of_nodes()} qubits"
            )

        restart_summaries.append(
            {
                "restart": restart,
                "num_embeddings": len(current),
                "remaining_qubits": remaining.number_of_nodes(),
                "methods": [item["method"] for item in current],
            }
        )

        print(f"  -> restart found {len(current)} vertex-disjoint embeddings")

        if len(current) > len(best):
            best = current

        if len(best) == target_copies:
            print("Reached the vertex-count upper bound; stopping restarts early.")
            break

    return best, restart_summaries, vertex_upper_bound


def save_embeddings(device, graph_id, logical_graph, embeddings):
    n = logical_graph.number_of_nodes()
    path = EMBEDDINGS_DIRECTORY / f"{device}_{graph_id}_{n}.txt"
    with open(path, "w") as file:
        file.write(str([item["mapping"] for item in embeddings]))
    return path


def save_summary(
    device,
    solver,
    logical_graph,
    lx,
    ly,
    embeddings,
    restart_summaries,
    vertex_upper_bound,
):
    graph_id = str(solver.graph_id)
    path = EMBEDDINGS_DIRECTORY / (
        f"{device}_{graph_id}_shuriken_L{lx}x{ly}_parallel_summary.json"
    )

    payload = {
        "device": device,
        "solver_identity": solver_identity_text(solver),
        "graph_id": graph_id,
        "Lx": lx,
        "Ly": ly,
        "N": logical_graph.number_of_nodes(),
        "E": logical_graph.number_of_edges(),
        "num_parallel_embeddings": len(embeddings),
        "vertex_count_upper_bound": vertex_upper_bound,
        "embeddings": [
            {
                "index": idx,
                "method": item["method"],
                "x_offset": item["x_offset"],
                "y_offset": item["y_offset"],
                "k_histogram": item["k_histogram"],
                "physical_qubits": sorted(item["mapping"].values()),
            }
            for idx, item in enumerate(embeddings)
        ],
        "restart_summaries": restart_summaries,
    }

    with open(path, "w") as file:
        json.dump(payload, file, indent=2)
    return path


def main():
    logical_graph, _, _, _ = create_shuriken(LX, LY)
    n = logical_graph.number_of_nodes()
    e = logical_graph.number_of_edges()

    hardware_graph, solver = dh.start_dwave_connection(QA_DEVICE)
    if solver is None:
        raise RuntimeError(f"Could not connect to {QA_DEVICE}")

    m, t = solver_zephyr_shape(solver)
    if LX > m or LY > m:
        raise ValueError(
            f"Requested {LX}x{LY} patch does not fit the analytic Z{m} construction"
        )

    print(f"solver requested: {QA_DEVICE}")
    print(f"solver identity:  {solver_identity_text(solver)}")
    print(f"graph_id:         {solver.graph_id}")
    print(f"topology:         Z{m}, t={t}")
    print(
        f"working graph:    {hardware_graph.number_of_nodes()} qubits, "
        f"{hardware_graph.number_of_edges()} couplers"
    )
    print(f"logical Shuriken: {LX}x{LY}, N={n}, E={e}")
    print(
        "vertex-only upper bound on simultaneous copies: "
        f"{hardware_graph.number_of_nodes() // n}"
    )

    best, restart_summaries, vertex_upper_bound = find_parallel_embeddings(
        logical_graph=logical_graph,
        hardware_graph=hardware_graph,
        lx=LX,
        ly=LY,
        m=m,
        t=t,
    )

    if not best:
        print("\nNo native one-qubit-per-spin embedding was found.")
        print("This is a search result, not a proof of non-embeddability.")
        return

    embedding_path = save_embeddings(
        QA_DEVICE, str(solver.graph_id), logical_graph, best
    )
    summary_path = save_summary(
        QA_DEVICE,
        solver,
        logical_graph,
        LX,
        LY,
        best,
        restart_summaries,
        vertex_upper_bound,
    )

    print("\n" + "=" * 76)
    print(f"BEST: {len(best)} vertex-disjoint native embeddings")
    print(f"logical size: {LX}x{LY}, N={n}, E={e}")
    print(f"vertex-count upper bound: {vertex_upper_bound}")
    print(f"embedding file: {embedding_path}")
    print(f"summary file:   {summary_path}")
    print("These mappings are in the same list-of-dicts format used by the Gibbs project.")


if __name__ == "__main__":
    main()
