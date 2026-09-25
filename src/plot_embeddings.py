
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import dwave_networkx as dnx

import helpers as dh
from shuriken import create_shuriken


SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
EMBEDDINGS_DIRECTORY = PROJECT_ROOT / "embeddings"
PLOTS_DIRECTORY = PROJECT_ROOT / "plots"
PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

QA_DEVICE = os.environ.get("DWAVE_QPU", "Advantage2_system1")
LX = int(os.environ.get("SHURIKEN_LX", "6"))
LY = int(os.environ.get("SHURIKEN_LY", str(LX)))

# Optional controls.
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"
PLOT_INDIVIDUAL = os.environ.get("PLOT_INDIVIDUAL", "0") == "1"
MAX_INDIVIDUAL = int(os.environ.get("MAX_INDIVIDUAL", "4"))
DPI = int(os.environ.get("PLOT_DPI", "250"))


def zephyr_shape(solver):
    topology = solver.properties.get("topology", {})
    if topology.get("type") != "zephyr":
        raise ValueError(
            f"{QA_DEVICE} is not Zephyr: topology={topology.get('type')!r}"
        )
    shape = topology.get("shape")
    if not shape:
        raise ValueError("solver did not report topology.shape")
    m = int(shape[0])
    t = int(shape[1]) if len(shape) > 1 else 4
    return m, t


def make_zephyr_graphs(solver):
    """Return ideal and live Zephyr graphs using linear qubit labels."""
    m, t = zephyr_shape(solver)

    ideal_graph = dnx.zephyr_graph(
        m,
        t=t,
        coordinates=False,
        data=True,
    )

    live_graph = dnx.zephyr_graph(
        m,
        t=t,
        node_list=list(solver.nodes),
        edge_list=list(solver.undirected_edges),
        coordinates=False,
        data=True,
        check_node_list=True,
        check_edge_list=True,
    )

    return ideal_graph, live_graph, m, t


def draw_background(ax, ideal_graph, live_graph, pos, show_faults=True):
    """Draw ideal Zephyr faintly and the current working graph on top."""
    nx.draw_networkx_edges(
        ideal_graph,
        pos,
        ax=ax,
        edge_color="0.90",
        width=0.18,
        alpha=0.22,
    )
    nx.draw_networkx_nodes(
        ideal_graph,
        pos,
        ax=ax,
        node_color="0.88",
        node_size=1.0,
        alpha=0.30,
        linewidths=0,
    )

    nx.draw_networkx_edges(
        live_graph,
        pos,
        ax=ax,
        edge_color="0.58",
        width=0.30,
        alpha=0.38,
    )
    nx.draw_networkx_nodes(
        live_graph,
        pos,
        ax=ax,
        node_color="0.35",
        node_size=2.1,
        alpha=0.65,
        linewidths=0,
    )

    if show_faults:
        missing_nodes = sorted(set(ideal_graph.nodes) - set(live_graph.nodes))
        if missing_nodes:
            nx.draw_networkx_nodes(
                ideal_graph,
                pos,
                nodelist=missing_nodes,
                ax=ax,
                node_color="red",
                node_size=5.0,
                alpha=0.80,
                linewidths=0,
            )

    ax.set_aspect("equal")
    ax.set_axis_off()


def save_figure(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"saved: {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_live_topology(ideal_graph, live_graph, pos, solver, m):
    fig, ax = plt.subplots(figsize=(14, 14))
    draw_background(ax, ideal_graph, live_graph, pos, show_faults=True)

    missing_nodes = len(set(ideal_graph.nodes) - set(live_graph.nodes))
    missing_edges = len(set(map(frozenset, ideal_graph.edges)) - set(map(frozenset, live_graph.edges)))

    # ax.set_title(
    #     f"{QA_DEVICE} live Zephyr Z{m} working graph\n"
    #     f"graph_id={solver.graph_id} | "
    #     f"live qubits={live_graph.number_of_nodes()} | "
    #     f"live couplers={live_graph.number_of_edges()} | "
    #     f"missing qubits={missing_nodes} | missing ideal couplers={missing_edges}"
    # )

    path = PLOTS_DIRECTORY / (
        f"{QA_DEVICE}_{solver.graph_id}_Z{m}_working_graph.png"
    )
    save_figure(fig, path)


def physical_interaction_edges(logical_graph, mapping):
    return [(mapping[u], mapping[v]) for u, v in logical_graph.edges]


def plot_parallel_embeddings(
    ideal_graph,
    live_graph,
    pos,
    logical_graph,
    mappings,
    solver,
    m,
):
    fig, ax = plt.subplots(figsize=(16, 16))
    draw_background(ax, ideal_graph, live_graph, pos, show_faults=False)

    cmap = plt.get_cmap("turbo", max(1, len(mappings)))

    for index, mapping in enumerate(mappings):
        color = cmap(index)
        physical_nodes = list(mapping.values())
        physical_edges = physical_interaction_edges(logical_graph, mapping)

        nx.draw_networkx_edges(
            live_graph,
            pos,
            edgelist=physical_edges,
            ax=ax,
            edge_color=[color],
            width=1.20,
            alpha=0.90,
        )
        nx.draw_networkx_nodes(
            live_graph,
            pos,
            nodelist=physical_nodes,
            ax=ax,
            node_color=[color],
            node_size=9.0,
            alpha=0.95,
            linewidths=0,
        )

    # ax.set_title(
    #     f"Parallel native Shuriken embeddings on {QA_DEVICE} (Z{m})\n"
    #     f"L={LX}x{LY}, N={logical_graph.number_of_nodes()}, "
    #     f"copies={len(mappings)}, graph_id={solver.graph_id}"
    # )

    # A legend is useful for a modest number of copies but unreadable for dozens.
    if len(mappings) <= 12:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="-",
                markersize=6,
                linewidth=2,
                color=cmap(i),
                label=f"copy {i}",
            )
            for i in range(len(mappings))
        ]
        # ax.legend(handles=handles, loc="upper right", frameon=True)

    path = PLOTS_DIRECTORY / (
        f"{QA_DEVICE}_{solver.graph_id}_shuriken_L{LX}x{LY}_parallel.png"
    )
    save_figure(fig, path)


def plot_single_embedding(
    ideal_graph,
    live_graph,
    pos,
    logical_graph,
    logical_to_name,
    mapping,
    copy_index,
    solver,
    m,
):
    """Plot one embedding and distinguish A, Bx and By physical qubits."""
    fig, ax = plt.subplots(figsize=(16, 16))
    draw_background(ax, ideal_graph, live_graph, pos, show_faults=False)

    physical_edges = physical_interaction_edges(logical_graph, mapping)
    nx.draw_networkx_edges(
        live_graph,
        pos,
        edgelist=physical_edges,
        ax=ax,
        edge_color="black",
        width=1.35,
        alpha=0.90,
    )

    groups = {
        "A": [],
        "Bx": [],
        "By": [],
    }
    for logical, name in logical_to_name.items():
        groups[name[0]].append(mapping[logical])

    styles = {
        "A": {"color": "tab:blue", "marker": "o", "size": 16},
        "Bx": {"color": "tab:orange", "marker": "s", "size": 24},
        "By": {"color": "tab:green", "marker": "^", "size": 28},
    }

    for group_name, nodes in groups.items():
        style = styles[group_name]
        nx.draw_networkx_nodes(
            live_graph,
            pos,
            nodelist=nodes,
            ax=ax,
            node_color=style["color"],
            node_shape=style["marker"],
            node_size=style["size"],
            alpha=1.0,
            linewidths=0,
        )

    handles = [
        Line2D(
            [0], [0],
            marker=styles[name]["marker"],
            linestyle="None",
            markerfacecolor=styles[name]["color"],
            markeredgecolor="none",
            markersize=7,
            label=name,
        )
        for name in ("A", "Bx", "By")
    ]
    # ax.legend(handles=handles, loc="best", frameon=True)

    # ax.set_title(
    #     f"Shuriken L={LX}x{LY}, native embedding copy {copy_index}\n"
    #     f"{QA_DEVICE} Z{m}, graph_id={solver.graph_id}"
    # )

    path = PLOTS_DIRECTORY / (
        f"{QA_DEVICE}_{solver.graph_id}_shuriken_L{LX}x{LY}_copy{copy_index}.png"
    )
    save_figure(fig, path)


def main():
    logical_graph, logical_to_name, _, _ = create_shuriken(LX, LY)
    n = logical_graph.number_of_nodes()

    hardware_graph, solver = dh.start_dwave_connection(QA_DEVICE)
    if solver is None:
        raise RuntimeError(f"Could not connect to {QA_DEVICE}")

    ideal_graph, live_graph, m, _ = make_zephyr_graphs(solver)
    pos = dnx.zephyr_layout(ideal_graph, scale=1.0)

    embedding_path = EMBEDDINGS_DIRECTORY / (
        f"{QA_DEVICE}_{solver.graph_id}_{n}.txt"
    )
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embedding_path}\n"
            "Run src/find_embeddings.py first."
        )

    mappings, saved_graph_id = dh.load_embeddings(embedding_path, n)
    if str(saved_graph_id) != str(solver.graph_id):
        raise RuntimeError(
            f"Embedding graph_id={saved_graph_id} does not match "
            f"current solver graph_id={solver.graph_id}"
        )

    print(f"loaded {len(mappings)} parallel embeddings from {embedding_path}")

    plot_live_topology(ideal_graph, live_graph, pos, solver, m)
    plot_parallel_embeddings(
        ideal_graph,
        live_graph,
        pos,
        logical_graph,
        mappings,
        solver,
        m,
    )

    if PLOT_INDIVIDUAL:
        for index, mapping in enumerate(mappings[:MAX_INDIVIDUAL]):
            plot_single_embedding(
                ideal_graph,
                live_graph,
                pos,
                logical_graph,
                logical_to_name,
                mapping,
                index,
                solver,
                m,
            )


if __name__ == "__main__":
    main()
