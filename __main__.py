from src.data import Data
from src import FileParser
from src.plotter import Plotter
from src.graph import ArticleGraph
from src.processing import process_articles

CONFIG_FILE = "configuration.yaml"


def main():
    # data object to store all information
    data = Data()

    # parse run configuration
    parser = FileParser()
    config = parser.parse_config(file_name=CONFIG_FILE)

    # get article information
    articles = parser.get_data(
        base_url=config["base_url"],
        tool=config["tool_name"],
        email=config["email"],
        query=config["query"],
        batch_size=config["batch_size"],
        update_flag=config["update_articles"],
        out_dir=config["output_directory"],
    )
    print("\n[ ] Data collected.")

    # node list and edge lists
    #data, all_node_types, all_edge_types = process_articles(articles[:300])
    data.node_edge_dictionary, data.node_types, data.edge_types = process_articles(articles[:300])
    print("[ ] Data processed.")

    # generate graphs for different node-type combinations
    ## combinations of nodes for different graphs
    node_combinations = {
        "only_author": ["author"],
        "only_keyword": ["keyword"],
        "article_author": ["article", "author"],
        "author_keyword": ["author", "keyword"],
        "all": ["author", "keyword", "article"], # necessary for later parts
    }
    data = get_graphs(node_comb=node_combinations, data=data)

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

    #
    plot = Plotter()
    plot.metapath2vec_loss_history(
        data=data,
        output_file=config["output_directory"]/"figure_metapath2vec_loss_history.png",
    )
    #
    plot.embedding_space(
        data=data,
        output_file=config["output_directory"]/"figure_embedding_space_projection_all.html")
    #
    for node_type in data.node_types:
        plot.embedding_space(
            data=data,
            node_type=node_type,
            output_file=config["output_directory"]/f"figure_embedding_space_projection_{node_type}s.html")
    
    exit()
    # plot Graph structures
    for my_graph_type, graph_struct in graph_dictionary.items():
        output_file_name = config["output_directory"] / f"figure_{my_graph_type}.html"
        graph_struct.plot(file_name=output_file_name)

    print("\n[!] Done")


def get_graphs(node_comb: dict, data: Data) -> Data:

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


if __name__ == "__main__":
    main()
