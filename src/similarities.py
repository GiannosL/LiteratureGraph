import torch
import hdbscan
import numpy as np
import pandas as pd
import torch.nn.functional as F


def rank_similar(
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        names_a: list[str],
        names_b: list[str]
) -> pd.DataFrame:
    # normalize embeddings
    embs_a = F.normalize(embeddings_a, p=2, dim=1)
    embs_b = F.normalize(embeddings_b, p=2, dim=1)

    # cross-cosine similarity matrix (n_a, n_b)
    similarity_matrix = embeddings_a @ embeddings_b.T

    # convert to dataframe
    df = pd.DataFrame(similarity_matrix.detach().cpu().numpy(), index=names_a, columns=names_b)
    # convert to long format
    df = df.stack().reset_index()
    df.columns = ["entity_a", "entity_b", "similarity_score"]

    # rank entities in B based on their score for A
    df["rank"] = df.groupby("entity_a")["similarity_score"].rank(ascending=False, method="dense").astype(int)
    df = df.sort_values(["entity_a", "rank"]).reset_index(drop=True)

    return df


def cluster_nodes(embeddings: torch.Tensor, names_a: list[str], norm: bool = True) -> pd.DataFrame:
    """
    This function takes as input embeddings for a SPECIFIC NODE TYPE and clusters it using HDBSCAN.
    """
    # convert to numpy
    x = embeddings.detach().cpu().numpy()

    # optionally normalize embeddings
    if norm:
        x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)

    # initialize HDBSCAN
    my_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    # fit model
    clustered_labels = my_clusterer.fit_predict(x)
    # extract membership probabilities
    clustered_label_probabilities = my_clusterer.probabilities_

    # convert to dataframe
    data = {"node": names_a, "cluster": clustered_labels, "probability": clustered_label_probabilities}
    results = pd.DataFrame(data)
    #
    results = results.sort_values(by=["cluster", "probability", "node"], ascending=[True, False, True])
    results = results.reset_index(drop=True)

    return results
