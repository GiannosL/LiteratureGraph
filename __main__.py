from src import FileParser
from src.graph import ArticleGraph
from src.processing import process_articles

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
        "figure_only_author.html": ["author"],
        "figure_only_keyword.html": ["keyword"],
        "figure_article_author.html": ["article", "author"],
        "figure_author_keyword.html": ["author", "keyword"],
    }
    for output_file_name, my_node_combo in node_combinations.item():
        # select edge types with relevant node types
        my_edge_types = [e for e in all_edge_types if (e[0] in my_node_combo) and (e[2] in my_node_combo)]
        #
        article_graph = ArticleGraph(
            data=data,
            node_types=my_node_combo,
            edge_types=my_edge_types,
        )
        output_file_name = config["output_directory"] / output_file_name
        article_graph.plot()


if __name__ == "__main__":
    main()
