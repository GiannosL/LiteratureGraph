import pandas as pd
import networkx as nx
from pathlib import Path
import plotly.graph_objects as go
from src import NODE_TYPE_COLORS, NODE_SIZE_MAP


class ArticleGraph:
    def __init__(self, data: dict, node_types: list[str], edge_types: list[tuple]):
        self._data = data
        self._node_types = node_types
        self._edge_types = edge_types

        # generate graph
        self._graph = _generate_graph(data, node_types, edge_types)

        # add node degrees to graph
        node_degree = dict(self._graph.degree())
        nx.set_node_attributes(self._graph, node_degree, "degree")

        # assign node sizes for graph figure
        node_bins = _bin_node_sizes(graph=self._graph, n_bins=7)

        for my_node, my_attrs in self._graph.nodes(data=True):
            # add size for graph figure
            my_attrs["size"] = NODE_SIZE_MAP.get(node_bins[my_node], 1_000)
            # check if the node is an article
            my_attrs["name"] = data["id_article_map"].get(my_node, my_node)

    def plot(self, file_name: Path = None):
        fig = _plot_interactive_graph(graph=self._graph)

        if file_name == None:
            fig.show()
        else:
            fig.write_html(file_name)


def _generate_graph(data: dict, node_types: list[str], edge_types: list[tuple]) -> nx.Graph:
    #
    my_graph = nx.Graph()
    for node_type in node_types:
        my_attrs = {"type": node_type, "color": NODE_TYPE_COLORS[node_type]}
        my_graph.add_nodes_from((my_node, my_attrs) for my_node in data[node_type])

    for edge_type in edge_types:
        my_graph.add_edges_from(data[edge_type])

    return my_graph


def _plot_interactive_graph(graph: nx.Graph):
    # node positions
    pos = nx.spring_layout(graph, seed=71)
    #
    node_x = [pos[my_node][0] for my_node in graph.nodes()]
    node_y = [pos[my_node][1] for my_node in graph.nodes()]

    #
    marker_params = {
        "size": [graph.nodes[n]["size"] for n in graph.nodes()],
        "color": [graph.nodes[n]["color"] for n in graph.nodes()],
        "line": {"width": 1, "color": "black"},
    }
    node_text = [
        (
            f"Name: {graph.nodes[n]['name']}"
            f"<br>Type: {graph.nodes[n]['type']}"
            f"<br>Degree: {graph.nodes[n]['degree']}"
            #f"<br>Size: {graph.nodes[n]['size']}"
         ) for n in  graph.nodes()
    ]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=marker_params,
        text=node_text,
        hoverinfo="text",
    )

    #
    edge_x, edge_y = [], []
    for edge in graph.edges():
        source, target = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    #
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1, "color": "lightgrey"},
        hoverinfo="none",
    )

    #
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )

    return fig


def _bin_node_sizes(graph: nx.Graph, n_bins: int) -> dict:
    degrees = pd.Series(dict(graph.degree()))
    #
    #bins = pd.qcut(degrees.rank(method="first"), q=n_bins, labels=False)
    bins = pd.cut(degrees, bins=n_bins, labels=False)
    bins = bins.to_dict()

    return bins
