import heapq
import math

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st


# Graphs and coordinates from Tasks 2 and 3 of the lab notebook.
AIRPORT_COORDS = {
    "Baggage_Area": (0, 0),
    "Security": (2, 1),
    "Checkpoint": (1, 4),
    "Food_Court": (4, 2),
    "Terminal_Hall": (5, 5),
    "Departure_Gate": (8, 6),
}
AIRPORT_GRAPH = {
    "Baggage_Area": {"Security": 2.2, "Checkpoint": 4.1},
    "Security": {"Food_Court": 2.2},
    "Checkpoint": {"Terminal_Hall": 5.0},
    "Food_Court": {"Terminal_Hall": 3.2, "Departure_Gate": 6.0},
    "Terminal_Hall": {"Departure_Gate": 3.2},
    "Departure_Gate": {},
}

HOSPITAL_COORDS = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6),
}
HOSPITAL_GRAPH = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {},
}


def euclidean_heuristic(node, goal, coordinates):
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return math.hypot(x2 - x1, y2 - y1)


def reconstruct_path(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return list(reversed(path))


def greedy_best_first_search(start, goal, graph, coordinates):
    frontier = [(euclidean_heuristic(start, goal, coordinates), start)]
    parent = {start: None}
    visited = set()
    expansion_order = []

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(parent, current)
            total_cost = sum(graph[u][v] for u, v in zip(path, path[1:]))
            return path, total_cost, expansion_order

        for neighbor, _cost in graph[current].items():
            if neighbor not in visited and neighbor not in parent:
                parent[neighbor] = current
                heapq.heappush(
                    frontier,
                    (euclidean_heuristic(neighbor, goal, coordinates), neighbor),
                )

    return None, math.inf, expansion_order


def a_star_search(start, goal, graph, coordinates):
    frontier = [(euclidean_heuristic(start, goal, coordinates), 0.0, start)]
    parent = {start: None}
    g_cost = {start: 0.0}
    closed = set()
    expansion_order = []

    while frontier:
        _f_current, g_current, current = heapq.heappop(frontier)

        if current in closed:
            continue
        closed.add(current)
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(parent, current)
            return path, g_cost[current], expansion_order

        for neighbor, edge_cost in graph[current].items():
            tentative_g = g_cost[current] + edge_cost
            if tentative_g < g_cost.get(neighbor, math.inf):
                g_cost[neighbor] = tentative_g
                parent[neighbor] = current
                f_score = tentative_g + euclidean_heuristic(
                    neighbor, goal, coordinates
                )
                heapq.heappush(frontier, (f_score, tentative_g, neighbor))

    return None, math.inf, expansion_order


def build_networkx_graph(graph, coordinates):
    network = nx.DiGraph()
    for node, position in coordinates.items():
        network.add_node(node, pos=position)
    for node, neighbors in graph.items():
        for neighbor, cost in neighbors.items():
            network.add_edge(node, neighbor, weight=cost)
    return network


def draw_graph(graph, coordinates, path, start, goal):
    network = build_networkx_graph(graph, coordinates)
    positions = coordinates
    fig, ax = plt.subplots(figsize=(11, 6))

    node_colors = []
    for node in network.nodes:
        if node == start and node == goal:
            node_colors.append("#8e44ad")
        elif node == start:
            node_colors.append("#2ecc71")
        elif node == goal:
            node_colors.append("#f39c12")
        else:
            node_colors.append("#b8d8f0")

    nx.draw_networkx_nodes(
        network, positions, node_size=1900, node_color=node_colors, ax=ax
    )
    nx.draw_networkx_labels(network, positions, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        network,
        positions,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.04",
        edge_color="#777777",
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        network,
        positions,
        edge_labels=nx.get_edge_attributes(network, "weight"),
        font_size=8,
        ax=ax,
    )

    if path and len(path) > 1:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            network,
            positions,
            edgelist=path_edges,
            edge_color="red",
            width=3.5,
            arrows=True,
            arrowsize=22,
            connectionstyle="arc3,rad=0.04",
            ax=ax,
        )

    ax.set_title("Search Graph — Solution Path Highlighted")
    ax.axis("off")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Informed Search Visualizer", page_icon="🧭", layout="wide")
st.title("🧭 Interactive Informed Search Visualization")
st.caption("AI Lab — Greedy Best-First Search (GBFS) and A* Search")

with st.sidebar:
    st.header("Search Settings")
    graph_choice = st.selectbox("Choose graph", ["Airport Baggage Handling", "Hospital Emergency Supply"])
    if graph_choice == "Airport Baggage Handling":
        graph = AIRPORT_GRAPH
        coordinates = AIRPORT_COORDS
        default_start = "Baggage_Area"
        default_goal = "Departure_Gate"
    else:
        graph = HOSPITAL_GRAPH
        coordinates = HOSPITAL_COORDS
        default_start = "Pharmacy"
        default_goal = "Emergency_Ward"

    nodes = list(coordinates.keys())
    start_node = st.selectbox(
        "Initial node", nodes, index=nodes.index(default_start)
    )
    goal_node = st.selectbox(
        "Goal node", nodes, index=nodes.index(default_goal)
    )
    algorithm = st.selectbox(
        "Search algorithm", ["Greedy Best-First Search (GBFS)", "A* Search"]
    )
    run_search = st.button("Run Search", type="primary", use_container_width=True)

if run_search:
    if algorithm == "Greedy Best-First Search (GBFS)":
        path, total_cost, expanded = greedy_best_first_search(
            start_node, goal_node, graph, coordinates
        )
    else:
        path, total_cost, expanded = a_star_search(
            start_node, goal_node, graph, coordinates
        )

    st.session_state["search_result"] = {
        "graph_choice": graph_choice,
        "start": start_node,
        "goal": goal_node,
        "algorithm": algorithm,
        "path": path,
        "cost": total_cost,
        "expanded": expanded,
    }

result = st.session_state.get("search_result")
if result:
    # Clear stale results if the graph selection changes.
    if result["graph_choice"] != graph_choice:
        st.info("Choose the graph and click **Run Search** to display its result.")
    else:
        st.subheader("Search Results")
        if result["path"]:
            col1, col2, col3 = st.columns(3)
            col1.metric("Algorithm", result["algorithm"])
            col2.metric("Total path cost", f'{result["cost"]:.2f}')
            col3.metric("Nodes expanded", len(result["expanded"]))

            st.markdown("**Solution path**")
            st.write(" → ".join(result["path"]))
            st.markdown("**Node expansion order**")
            st.write(" → ".join(result["expanded"]))

            fig = draw_graph(
                graph, coordinates, result["path"], result["start"], result["goal"]
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.error("No path exists between the selected initial and goal nodes.")
            st.write("**Nodes expanded:** " + " → ".join(result["expanded"]))
else:
    st.info("Select the graph, initial node, goal node, and algorithm, then click **Run Search**.")
    fig = draw_graph(graph, coordinates, None, default_start, default_goal)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.caption("Green node: start · Orange node: goal · Red edges: returned solution path")
