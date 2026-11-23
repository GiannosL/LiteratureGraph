from src import FileParser
from src.plotter import Plotter
from src.graph import ArticleGraph
from src.processing import process_articles
from src.node_representation import get_node_embeddings
from src.similarities import rank_similar, cluster_nodes

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
    print("\n[ ] Data collected.")

    # node list and edge lists
    data, all_node_types, all_edge_types = process_articles(articles[:100])
    print("[ ] Data processed.")

    # generate graphs for different node-type combinations
    node_combinations = {
        "only_author": ["author"],
        "only_keyword": ["keyword"],
        "article_author": ["article", "author"],
        "author_keyword": ["author", "keyword"],
        "all": ["author", "keyword", "article"],
    }
    graph_dictionary = {}
    for node_combo, my_node_combo in node_combinations.items():
        print(f"     [ ] Generating: {node_combo} graph.")
        # select edge types with relevant node types
        my_edge_types = [e for e in all_edge_types if (e[0] in my_node_combo) and (e[2] in my_node_combo)]
        #
        article_graph = ArticleGraph(
            data=data,
            node_types=my_node_combo,
            edge_types=my_edge_types,
        )

        # save graph
        graph_dictionary[node_combo] = article_graph
    print("[ ] Graphs generated.")

    # node representation learning
    print("\n=== MetaPath2Vec: authors-keywords ===")
    node_representations = get_node_embeddings(
        graph=graph_dictionary["all"],
        metapath=[
            ("article", "is_written_by", "author"),
            ("author", "writes_with", "author"),
            ("author", "works_on", "keyword"),
            ("keyword", "appears_with", "keyword"),
            ("keyword", "rev_works_on", "author"),
            ("author", "rev_is_written_by", "article")
        ],
        metapath_sett=config["metapath_settings"],
    )
    print("\n[ ] MetaPath2Vec finished.")

    # Similarity is not used at the moment TODO: Utilize similarities
    keyword_keyword_m2v_similarity = rank_similar(
        embeddings_a=node_representations["embeddings"]["keyword"],
        embeddings_b=node_representations["embeddings"]["keyword"],
        names_a=list(graph_dictionary["all"].node_encodings["keyword"].keys()),
        names_b=list(graph_dictionary["all"].node_encodings["keyword"].keys()),
    )
    author_keyword_m2v_similarity = rank_similar(
        embeddings_a=node_representations["embeddings"]["author"],
        embeddings_b=node_representations["embeddings"]["keyword"],
        names_a=list(graph_dictionary["all"].node_encodings["author"].keys()),
        names_b=list(graph_dictionary["all"].node_encodings["keyword"].keys()),
    )
    article_author_m2v_similarity = rank_similar(
        embeddings_a=node_representations["embeddings"]["author"],
        embeddings_b=node_representations["embeddings"]["article"],
        names_a=list(graph_dictionary["all"].node_encodings["author"].keys()),
        names_b=list(graph_dictionary["all"].node_encodings["article"].keys()),
    )

    # cluster nodes based on embedding
    clusters = {}
    for my_type in graph_dictionary["all"].node_encodings.keys():
        clusters[my_type] = cluster_nodes(
            embeddings=node_representations["embeddings"][my_type],
            names_a=list(graph_dictionary["all"].node_encodings[my_type].keys()),
            norm=True
        )

    #
    plot = Plotter()
    plot.metapath2vec_loss_history(
        loss_history=node_representations["loss_history"],
        output_file=config["output_directory"]/"figure_metapath2vec_loss_history.png",
    )
    plot.embedding_space(
        graph=graph_dictionary["all"],
        embs=node_representations["embeddings"],
        output_file=config["output_directory"]/"figure_embedding_space_projection_all.html")
    plot.embedding_space(
        node_type="keyword",
        graph=graph_dictionary["all"],
        embs=node_representations["embeddings"],
        output_file=config["output_directory"]/"figure_embedding_space_projection_keywords.html")
    plot.embedding_space(
        node_type="author",
        graph=graph_dictionary["all"],
        embs=node_representations["embeddings"],
        output_file=config["output_directory"]/"figure_embedding_space_projection_authors.html")
    plot.embedding_space(
        node_type="article",
        graph=graph_dictionary["all"],
        embs=node_representations["embeddings"],
        output_file=config["output_directory"]/"figure_embedding_space_projection_articles.html")
    # plot Graph structures
    for my_graph_type, graph_struct in graph_dictionary.items():
        output_file_name = config["output_directory"] / f"figure_{my_graph_type}.html"
        graph_struct.plot(file_name=output_file_name)

    print("\n[!] Done")


if __name__ == "__main__":
    main()
