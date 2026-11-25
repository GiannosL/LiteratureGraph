import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from src.data import Data
from src.graph import ArticleGraph


class Plotter:
    def __init__(self):
        self._fs = (20, 14) # figure size
        self._t = 22 # title size
        self._l = 18 # label size
        self._tick = 14 # ax-tick size
        self._dpi = 500 # dpi

    def metapath2vec_loss_history(self, data: Data, output_file: Path):
        # number of epochs
        loss_history = data.node_representations["loss_history"]
        epochs = [i+1 for i in range(len(loss_history))]

        # plotting
        plt.figure(figsize=self._fs)

        plt.plot(epochs, loss_history, color="skyblue")

        plt.title("MetaPath2Vec loss history", weight="bold", fontsize=self._t)
        plt.xlabel("epoch", fontsize=self._l)
        plt.ylabel("loss history", fontsize=self._l)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=self._tick) # tick size

        plt.savefig(output_file, dpi=self._dpi)
        plt.close()

    @staticmethod
    def embedding_space(data: Data, output_file: Path, node_type: str = ""):
        # data
        graph = data.graph_dictionary["all"]
        embs = data.node_representations["embeddings"]

        # if specified to subset on a specific node type remove the others
        if node_type:
            embs = {key: value for key, value in embs.items() if key == node_type}

        # project into two dimensions
        emb_proj = _generate_tsne_embeddings(embs=embs)
        # map the node names
        node_map, node_map_rev = {}, graph.node_encodings
        for node_type, node_dict in node_map_rev.items():
            node_map[node_type] = {v:k for k,v in node_dict.items()}

        node_list = []
        for i, row in emb_proj.iterrows():
            type_ = row["node_type"]
            node_list.append(node_map[type_][row["node_id"]])
        emb_proj["node_name"] = node_list

        # plotting
        fig = go.Figure()

        for node_type, df in emb_proj.groupby("node_type"):
            #
            fig.add_trace(
                go.Scatter(
                    x=df["dim_1"],
                    y=df["dim_2"],
                    mode="markers",
                    name=node_type,
                    marker={"size": 6, "opacity": 0.7},
                    customdata=df[["node_name", "node_type"]],
                    hovertemplate="%{customdata[0]}<br>type: %{customdata[1]}<br><extra></extra>",
                    #hoverinfo="skip",
                )
            )

        fig.update_layout(
            title="MetaPath2Vec Embeddings - t-SNE",
            xaxis_title="dimension 1",
            yaxis_title="dimension 2",
            template="plotly_white",
            legend_title="Node type",
            width=2500,
            height=1200,
        )

        fig.write_html(output_file)


def _generate_tsne_embeddings(
        embs: dict[str, torch.Tensor],
        perplexity: float = 30.0,
        n_iter: int = 2_000,
        seed: int = 71):
    # Collect embeddings and node metadata
    all_embeddings, node_metadata = [], []

    #
    for node_type, emb_tensor in embs.items():
        #
        array = emb_tensor.detach().cpu().numpy()
        all_embeddings.append(array)
        #
        node_metadata.extend([{"node_type": node_type, "node_id": i} for i in range(len(array))])

    # combine embeddings
    x = np.concatenate(all_embeddings, axis=0)

    # apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        max_iter=n_iter,
        random_state=seed,
        init="pca"
    )
    proj = tsne.fit_transform(x)

    # make the projection into a dataframe
    df = pd.DataFrame(node_metadata)
    df["dim_1"] = proj[:, 0]
    df["dim_2"] = proj[:, 1]

    return df