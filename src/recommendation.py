from src.data import Data


class Recommender:
    def __init__(self, data: Data):
        self.author_list = data.node_edge_dictionary["author"]
        self.article_list = data.node_edge_dictionary["article"]
        self.keyword_list = data.node_edge_dictionary["keyword"]
        #
        self.edge_lists = {}
        for my_edge_type, my_edge_list in data.node_edge_dictionary.items():
            if type(my_edge_type) == tuple:
                self.edge_lists[my_edge_type] = my_edge_list
        #
        self.clusters = data.clusters
        self.similarities = data.similarities

    def cluster_descriptions(self, author_top_k: int = 3, cluster_top_k: int = 10) -> list[dict]:
        """
        Author clusters can be described using the top 3 most relevant keywords per author.

        :return:
        """
        # keep information of interest only
        ranks_of_interest = set([i+1 for i in range(author_top_k)])
        # author - keywords
        similarity_df = self.similarities[("author", "keyword")]
        similarity_df = similarity_df[similarity_df["rank"].isin(ranks_of_interest)]
        #
        similarity_df = similarity_df.rename(
            columns={"entity_a": "author", "entity_b": "keyword"}
        )
        similarity_df = similarity_df[["author", "keyword"]]

        # author clusters
        cluster_df = self.clusters["author"]
        cluster_df = cluster_df.rename(columns={"node": "author"})

        cluster_descriptions = []
        for my_cluster, my_df in cluster_df.groupby("cluster"):
            # add keywords
            merged_df = my_df.merge(similarity_df, on="author", how="left")
            # count keywords
            cluster_keyword_count = merged_df["keyword"].value_counts().reset_index(drop=False)

            # topX counts in the dataframe
            top_counts = sorted(set(cluster_keyword_count["count"]), reverse=True)
            top_counts = top_counts[:cluster_top_k]
            cluster_keyword_count = cluster_keyword_count[cluster_keyword_count["count"].isin(top_counts)]
            cluster_keywords = cluster_keyword_count["keyword"].tolist()
            # save outcome
            cluster_descriptions.append(
                {
                    "id": my_cluster, # ID of cluster
                    "authors": cluster_keywords["author"].tolist(), # authors in cluster
                    "descriptive_keywords": cluster_keywords["keyword"].tolist(), # keywords describing the authors
                }
            )

        return cluster_descriptions

    def assign_keywords(self):
        """
        Assign keywords to authors to explain their interests and to articles.
        Immediate keywords can be used from the graph and then supplemented with similarities.
        """
        pass
