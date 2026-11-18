import torch
from torch_geometric.nn import MetaPath2Vec
from src.graph import ArticleGraph


def get_node_embeddings(graph: ArticleGraph, metapath: list[tuple[str, str, str]], metapath_sett: dict):
    # load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    g = graph.torch_graph

    # node types in the graph
    metapath_node_types = []
    for edge_type in metapath:
        metapath_node_types.append(edge_type[0])
        metapath_node_types.append(edge_type[2])
    metapath_node_types = sorted(set(metapath_node_types))

    # number of nodes per node-type
    node_counts = {}
    for node_type in g.node_types:
        node_counts[node_type] = g[node_type].num_nodes

    # initialize MetaPath2Vec model
    model = MetaPath2Vec(
        edge_index_dict=g.edge_index_dict,
        embedding_dim=metapath_sett["embedding_dimensions"],
        metapath=metapath,
        walk_length=metapath_sett["walk_length"],
        context_size=metapath_sett["context_size"],
        walks_per_node=metapath_sett["walks_per_node"],
        num_negative_samples=metapath_sett["num_negative_samples"],
        num_nodes_dict=node_counts,
        sparse=True
    )
    model.to(device)

    # data loader for random walks TODO: Fix hard-coded parameters
    loader = model.loader(batch_size=1_000, shuffle=True, num_workers=0)
    # optimizer
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=metapath_sett["learning_rate"])

    # training loop
    model.train()
    for epoch in range(metapath_sett["epochs"]):
        total_loss = 0

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # average loss across batches
        avg_loss = total_loss / len(loader)
        print(f"Epochs {epoch+1}/{metapath_sett['epochs']} | Loss: {avg_loss:.4f}")

    # extract embeddings
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for node_type in metapath_node_types:
            embeddings[node_type] = model(node_type).cpu()

    return embeddings
