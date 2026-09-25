import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from shuriken import create_shuriken


def logical_positions(logical_to_name, spacing=3.0, half_square=0.65):
    pos = {}

    for logical, name in logical_to_name.items():
        site_type = name[0]

        if site_type == "A":
            _, i, j, a = name
            cx = i * spacing
            cy = j * spacing

            offsets = {
                0: (-half_square, +half_square),  # A0 = NW
                1: (+half_square, +half_square),  # A1 = NE
                2: (+half_square, -half_square),  # A2 = SE
                3: (-half_square, -half_square),  # A3 = SW
            }

            dx, dy = offsets[a]
            pos[logical] = (cx + dx, cy + dy)

        elif site_type == "Bx":
            _, i, j = name
            pos[logical] = ((i + 0.5) * spacing, j * spacing)

        elif site_type == "By":
            _, i, j = name
            pos[logical] = (i * spacing, (j + 0.5) * spacing)

    return pos


def plot_shuriken(lx=6, ly=6, show_labels=False, save=True):
    graph, logical_to_name, _, edge_types = create_shuriken(lx, ly)
    pos = logical_positions(logical_to_name)

    A_nodes, Bx_nodes, By_nodes = [], [], []
    for logical, name in logical_to_name.items():
        if name[0] == "A":
            A_nodes.append(logical)
        elif name[0] == "Bx":
            Bx_nodes.append(logical)
        elif name[0] == "By":
            By_nodes.append(logical)

    AA_edges, ABx_edges, ABy_edges = [], [], []
    for u, v in graph.edges():
        edge = tuple(sorted((u, v)))
        edge_type = edge_types[edge]

        if edge_type == "AA":
            AA_edges.append((u, v))
        elif edge_type == "ABx":
            ABx_edges.append((u, v))
        elif edge_type == "ABy":
            ABy_edges.append((u, v))

    fig, ax = plt.subplots(figsize=(12, 12))

    # Edges
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=AA_edges,
        width=2.5,
        edge_color="black",
        ax=ax
    )

    nx.draw_networkx_edges(
        graph, pos,
        edgelist=ABx_edges,
        width=1.6,
        style="dashed",
        edge_color="dimgray",
        ax=ax
    )

    nx.draw_networkx_edges(
        graph, pos,
        edgelist=ABy_edges,
        width=1.6,
        style="dotted",
        edge_color="dimgray",
        ax=ax
    )

    # Nodes
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=A_nodes,
        node_size=220,
        node_shape="o",
        node_color="#1f77b4",   # blue
        edgecolors="black",
        linewidths=0.8,
        ax=ax
    )

    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=Bx_nodes,
        node_size=260,
        node_shape="s",
        node_color="#ff7f0e",   # orange
        edgecolors="black",
        linewidths=0.8,
        ax=ax
    )

    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=By_nodes,
        node_size=280,
        node_shape="^",
        node_color="#2ca02c",   # green
        edgecolors="black",
        linewidths=0.8,
        ax=ax
    )

    # Labels
    if show_labels:
        labels = {}
        for logical, name in logical_to_name.items():
            if name[0] == "A":
                _, i, j, a = name
                labels[logical] = f"A{a}\n({i},{j})"
            elif name[0] == "Bx":
                _, i, j = name
                labels[logical] = f"Bx\n({i},{j})"
            elif name[0] == "By":
                _, i, j = name
                labels[logical] = f"By\n({i},{j})"

        nx.draw_networkx_labels(
            graph, pos,
            labels=labels,
            font_size=7,
            ax=ax
        )

    # Remove title and legend
    # ax.set_title(...)
    # ax.legend()

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if save:
        output = Path(f"shuriken_L{lx}x{ly}_clean.png")
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved: {output.resolve()}")

    plt.show()


if __name__ == "__main__":
    plot_shuriken(lx=6, ly=6, show_labels=False, save=True)