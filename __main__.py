from src.plotter import Plotter
from src import FileParser, flow
from src.recommendation import Recommender

CONFIG_FILE = "configuration.yaml"


def main():
    # parse run configuration
    config = FileParser().parse_config(file_name=CONFIG_FILE)

    # program flow for graph, node embedding, similarity, and clustering
    data = flow(config)

    # recommendations
    my_recommender = Recommender(data)
    my_recommender.cluster_descriptions(author_top_k=3, cluster_top_k=10)

    #
    plot = Plotter()
    plot.metapath2vec_loss_history(
        data=data,
        output_file=config["output_directory"]/"figure_metapath2vec_loss_history.html",
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


if __name__ == "__main__":
    main()
