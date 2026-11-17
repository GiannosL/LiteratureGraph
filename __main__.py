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
    data, node_types, edge_types = process_articles(articles[:100])

    #
    article_graph = ArticleGraph(
        data=data,
        node_types=node_types,
        edge_types=edge_types,
    )
    article_graph.plot()


if __name__ == "__main__":
    main()
