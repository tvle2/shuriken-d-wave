

import json
import os
from pathlib import Path

import helpers as dh
from shuriken import shuriken_couplings, shuriken_size


SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
EMBEDDINGS_DIRECTORY = PROJECT_ROOT / "embeddings"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "shuriken_native_smoke"

QPU_DEVICES = ["Advantage2_system4"]

LX = 6
LY = 6
N, E = shuriken_size(LX, LY)


J_AA = 1.0
X_X = 0.68922
X_Y = 0.95
ENERGY_SCALE = 0.5

NUM_READS = 100
NUM_JOBS = 1
ANNEALING_TIME_US = 20.0


def map_all_embeddings(mappings, logical_couplings, energy_scale):
    """Reuse the map-all-disjoint-embeddings pattern from sample_dwave.py."""
    h_physical = {}
    J_physical = {}

    for mapping in mappings:
        h_physical.update({mapping[i]: 0.0 for i in range(N)})

        for (u, v), coupling in logical_couplings.items():
            edge = tuple(sorted((mapping[u], mapping[v])))
            if edge in J_physical:
                raise ValueError(
                    f"Physical coupler {edge} is used by more than one embedding"
                )
            J_physical[edge] = coupling * energy_scale

    return h_physical, J_physical


def validate_programmed_ranges(solver, J_physical):
    """Basic individual-coupler range check before submission."""
    j_range = solver.properties.get("j_range")
    if not j_range:
        return

    j_min, j_max = j_range
    bad = [
        (edge, value)
        for edge, value in J_physical.items()
        if value < j_min or value > j_max
    ]
    if bad:
        raise ValueError(
            f"Programmed coupling {bad[0]} is outside solver j_range={j_range}"
        )


def main():
    logical_couplings, logical_to_name, _, edge_types = shuriken_couplings(
        LX,
        LY,
        j_aa=J_AA,
        x_x=X_X,
        x_y=X_Y,
    )

    if len(logical_to_name) != N or len(logical_couplings) != E:
        raise RuntimeError("Shuriken logical-size consistency check failed")

    for device in QPU_DEVICES:
        hardware_graph, solver = dh.start_dwave_connection(device)
        if solver is None:
            raise RuntimeError(f"Could not connect to {device}")

        hardware_graph.add_nodes_from(solver.nodes)

        embedding_file = EMBEDDINGS_DIRECTORY / (
            f"{device}_{solver.graph_id}_{N}.txt"
        )
        if not embedding_file.is_file():
            raise FileNotFoundError(
                f"No live-graph embedding for {LX}x{LY}. "
                f"Run find_embeddings.py first: {embedding_file}"
            )

        mappings, saved_graph_id = dh.load_embeddings(embedding_file, N)
        if str(solver.graph_id) != saved_graph_id:
            raise ValueError(
                f"Stale embedding for {device}: saved graph_id={saved_graph_id}, "
                f"current graph_id={solver.graph_id}"
            )

        h, J = map_all_embeddings(
            mappings,
            logical_couplings,
            ENERGY_SCALE,
        )

        unavailable_qubits = [qubit for qubit in h if qubit not in hardware_graph]
        if unavailable_qubits:
            raise ValueError(
                f"Embedding uses inactive qubit {unavailable_qubits[0]}"
            )

        unavailable_couplers = [
            edge for edge in J if not hardware_graph.has_edge(*edge)
        ]
        if unavailable_couplers:
            raise ValueError(
                f"Embedding uses inactive coupler {unavailable_couplers[0]}"
            )

        validate_programmed_ranges(solver, J)

        output_directory = OUTPUT_DIRECTORY / device / str(solver.graph_id)
        os.makedirs(output_directory, exist_ok=True)

        active_qubits_by_embedding = [
            [mapping[logical] for logical in range(N)]
            for mapping in mappings
        ]

        metadata = {
            "device": device,
            "graph_id": str(solver.graph_id),
            "topology": solver.properties.get("topology"),
            "lx": LX,
            "ly": LY,
            "N": N,
            "E": E,
            "embedding_file": embedding_file.name,
            "num_embeddings": len(mappings),
            "logical_qubit_order": list(range(N)),
            "active_qubits_by_embedding": active_qubits_by_embedding,
            "J_AA": J_AA,
            "x_x": X_X,
            "x_y": X_Y,
            "energy_scale": ENERGY_SCALE,
            "annealing_time_us": ANNEALING_TIME_US,
            "edge_type_counts": {
                edge_type: sum(value == edge_type for value in edge_types.values())
                for edge_type in ("AA", "ABx", "ABy")
            },
            "logical_site_names": {
                str(logical): list(name)
                for logical, name in logical_to_name.items()
            },
        }

        with open(
            output_directory / f"{device}_shuriken_embedding_metadata.json",
            "w",
        ) as file:
            json.dump(metadata, file, indent=2)

        params = {
            "num_reads": NUM_READS,
            "auto_scale": False,
            "annealing_time": ANNEALING_TIME_US,
        }

        base_path = output_directory / (
            f"{device}_Shuriken_L{LX}x{LY}_"
            f"xx{X_X}_xy{X_Y}_es{ENERGY_SCALE}_at{ANNEALING_TIME_US}"
        )

        print(
            f"Running native Shuriken smoke test on {device}: "
            f"L={LX}x{LY}, N={N}, E={E}, "
            f"embeddings={len(mappings)}"
        )

        dh.run_dwave(
            params,
            str(base_path),
            h,
            J,
            solver,
            num_jobs=NUM_JOBS,
        )


if __name__ == "__main__":
    main()
