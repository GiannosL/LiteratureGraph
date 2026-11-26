import yaml
import pickle
from typing import Any
from pathlib import Path

from src.data import Data
from src.graph import ArticleGraph
from src.processing import process_articles
from src.file_collector import get_articles


class FileParser:
    def __init__(self):
        pass

    @staticmethod
    def parse_config(file_name: str) -> dict[str, Any]:
        """ parse configuration file """
        with open(file_name, "r") as f:
            contents = yaml.safe_load(f)

        # convert to dictionary
        contents = dict(contents)

        # edit variables
        contents["output_directory"]  = Path(contents["output_directory"]).expanduser()

        # ensure that output directory exists
        contents["output_directory"].mkdir(exist_ok=True, parents=False)

        return contents

    @staticmethod
    def get_data(
            base_url: str,
            tool: str,
            email: str,
            query: str,
            batch_size: int,
            data_file: Path,
    ) -> list[dict[str, object]]:
        # get articles from the web
        articles = get_articles(
            base_url=base_url,
            tool_name=tool,
            email=email,
            query=query,
            batch_size=batch_size,
        )
        # save data dictionary
        with open(data_file, "wb") as f:
            pickle.dump(articles, f)

        return articles


def flow(config: dict[str, Any])  -> Data:
    """
    Control program's flow.
    """
    # data object to store all information
    data = Data()

    # data and output directory
    out_dir = config["output_directory"]
    data_dir = out_dir / "data/"
    data_dir.mkdir(exist_ok=True, parents=False)
    # downloaded articles
    article_file = data_dir / "articles.pkl"
    # all data generated during the run from
    data_file = out_dir / "data.pkl"

    # re-run the whole script
    if config["update_all"] or (not article_file.is_file()):
        # get article information
        articles = FileParser().get_data(
            base_url=config["base_url"],
            tool=config["tool_name"],
            email=config["email"],
            query=config["query"],
            batch_size=config["batch_size"],
            data_file=article_file,
        )

        # subset for testing subsets
        if config["subset"]:
            articles = articles[:500]

        # node list and edge lists
        data.node_edge_dictionary, data.node_types, data.edge_types = process_articles(articles)
        print("\n[ ] Data processed.")

        # generate graphs for different node-type combinations
        ## combinations of nodes for different graphs
        node_combinations = {
            "only_author": ["author"],
            "only_keyword": ["keyword"],
            "article_author": ["article", "author"],
            "author_keyword": ["author", "keyword"],
            "all": ["author", "keyword", "article"],  # necessary for later parts
        }
        data = _get_graphs(node_comb=node_combinations, data=data)
        print("[ ] Graphs generated.")

        # node representation learning
        print("\n=== MetaPath2Vec ===")
        data.get_node_embeddings(
            metapath=[
                ("article", "is_written_by", "author"),
                ("author", "writes_with", "author"),
                ("author", "works_on", "keyword"),
                ("keyword", "appears_with", "keyword"),
                ("keyword", "rev_works_on", "author"),
                ("author", "rev_is_written_by", "article")
            ],
            m2v_settings=config["metapath_settings"],
        )
        print("\n[ ] MetaPath2Vec finished.")

        # node similarities between different node types
        data.get_similarities(
            node_combos=[("article", "author"), ("author", "keyword"), ("keyword", "keyword")],
            out_dir=config["output_directory"],
        )

        # cluster nodes based on embedding
        data.cluster_nodes_by_type(out_dir=config["output_directory"])

        # save data object
        with open(data_file, "wb") as f:
            pickle.dump(data, f)
    else:
        with open(data_file, "rb") as f:
            data = pickle.load(f)

    print("[ ] Data collected.")

    return data


def _get_graphs(node_comb: dict, data: Data) -> Data:

    graph_dictionary = {}
    for node_combo, my_node_combo in node_comb.items():
        print(f"     [ ] Generating: {node_combo} graph.")
        # select edge types with relevant node types
        my_edge_types = [e for e in data.edge_types if (e[0] in my_node_combo) and (e[2] in my_node_combo)]
        #
        article_graph = ArticleGraph(
            data=data.node_edge_dictionary,
            node_types=my_node_combo,
            edge_types=my_edge_types,
        )

        # save graph
        graph_dictionary[node_combo] = article_graph

    # save in data object
    data.graph_dictionary = graph_dictionary

    return data
