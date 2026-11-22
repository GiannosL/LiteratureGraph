import torch
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
