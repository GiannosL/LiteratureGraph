from src import FileParser
from src.graph import ArticleGraph
from src.processing import process_articles
from src.node_representation import get_node_embeddings

CONFIG_FILE = "configuration.yaml"

def main():
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

    # node list and edge lists
    data, all_node_types, all_edge_types = process_articles(articles[:500])

    #
    node_combinations = {
        "only_author": ["author"],
        "only_keyword": ["keyword"],
        "article_author": ["article", "author"],
        "author_keyword": ["author", "keyword"],
    }
    graph_dictionary = {}
    for node_combo, my_node_combo in node_combinations.items():
        # select edge types with relevant node types
        my_edge_types = [e for e in all_edge_types if (e[0] in my_node_combo) and (e[2] in my_node_combo)]
        #
        article_graph = ArticleGraph(
            data=data,
            node_types=my_node_combo,
            edge_types=my_edge_types,
        )
        output_file_name = config["output_directory"] / f"figure_{node_combo}.html"
        article_graph.plot(file_name=output_file_name)

        # save graph
        graph_dictionary[node_combo] = article_graph

    # node representation learning
    node_representations = get_node_embeddings(
        graph=graph_dictionary["author_keyword"],
        metapath=[
            ("author", "writes_with", "author"),
            ("author", "works_on", "keyword"),
            ("keyword", "appears_with", "keyword"),
            ("keyword", "rev_works_on", "author"),
        ],
        metapath_sett={
            "embedding_dimensions": 256,
            "walk_length": 20,
            "context_size": 5,
            "walks_per_node":5,
            "num_negative_samples": 3,
            "learning_rate": 0.001,
            "epochs": 10
        }
    )

    print("\n[!] Done")


if __name__ == "__main__":
    main()
