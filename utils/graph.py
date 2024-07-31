from typing import (
    Union,
    Any,
    List, 
    Tuple,
    Dict, 
    Optional
)

import numpy as np
import torch
from scipy.sparse import csr_array, csr_matrix
from torch import Tensor, LongTensor
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, coalesce


def edge_index_to_csr_adj(
        edge_index: Tensor,
        num_nodes: int = None,
        edge_attr: Tensor = None, ) -> csr_array:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_attr is None:
        values = torch.ones(len(edge_index[0]))
    else:
        assert len(edge_attr.size()) == 1
        values = edge_attr

    adj = csr_array((values, (edge_index[0], edge_index[1]),),
                    shape=(num_nodes, num_nodes), )
    return adj

def edge_index_to_sparse_csr(edge_index, edge_attr=None, num_nodes=None, bidirectional=False):
    N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
    if edge_attr is None:
        edge_attr = torch.arange(edge_index.size(1))
    else:
        assert len(edge_attr.size()) == 1
    if bidirectional:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, -1 - edge_attr], dim=0)
    whole_adj = SparseTensor.from_edge_index(edge_index, edge_attr, (N, N), is_sorted=False)

    rowptr, col, value = whole_adj.csr()  # convert to csr form
    whole_adj = SparseTensor(rowptr=rowptr, col=col, value=value, sparse_sizes=(N, N), is_sorted=True, trust_data=True)
    return whole_adj

def safe_to_undirected(
        edge_index: LongTensor,
        edge_attr: Tensor = None):
    if is_undirected(edge_index, edge_attr):
        return edge_index, edge_attr
    else:
        return to_undirected(edge_index, edge_attr)

# This function converts an edge list into csr matrix
def edge_list_to_csr(edge_list, num_nodes):
    row, col = edge_list
    data = np.ones(row.size(0), dtype=np.float32)
    indptr = np.zeros(num_nodes + 1, dtype=int)

    # Compute the number of non-zero entries per row
    np.add.at(indptr, row.cpu().numpy() + 1, 1)
    np.cumsum(indptr, out=indptr)

    # Normalize data
    col_sums = np.zeros(num_nodes)
    np.add.at(col_sums, col.cpu().numpy(), data)
    data /= col_sums[col.cpu().numpy()]

    csr = csr_matrix((data, col.cpu().numpy(), indptr), shape=(num_nodes, num_nodes))
    return csr

# This function converts a csr matrix to sparse matrix if needed
def csr_to_torch_sparse(csr):
    coo = csr.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    
# Helper function to compute ppr score
def personalized_pagerank(edge_index, num_nodes, personalization_vector, max_iter, tol=1e-6, delta=None, alpha=0.85):
    csr_matrix = edge_list_to_csr(edge_index, num_nodes)
    csr_matrix = csr_to_torch_sparse(csr_matrix).to(dtype=torch.float32)
    
    # Create the personalization vetor and normalize it
    personalization = torch.tensor([personalization_vector.get(node_id, 1.0) for node_id in range(num_nodes)], dtype=torch.float32)
    personalization /= personalization.sum()

    # Initialize the rank vetor (equally distribute initial probability for each node)
    rank = torch.ones(num_nodes, dtype=torch.float32) / num_nodes
    teleport = (1 - alpha) * personalization # Teleportation probability (瞬移)
    
    count = 0
    # Iterate based on the amount to update the rank vector (Have different probability now)
    for _ in range(max_iter):
        new_rank = alpha * torch.sparse.mm(csr_matrix, rank.unsqueeze(1)).squeeze(1) + teleport
        if torch.norm(new_rank - rank, p=1) < tol:
            break
        if delta is not None and torch.max(torch.abs(new_rank - rank)) < delta:
            print(f"Stopping due to delta criterion at iteration")
            break
        rank = new_rank
        count = count + 1
        
    return {i: rank[i] for i in range(num_nodes)}

def sample_k_hop_subgraph_sparse(
    node_idx: Union[int, list[int], Tensor],
    hop: int,
    edge_index: SparseTensor,
    max_nodes_per_hop=-1,
    use_ppr=True,
    ppr_scores: Optional[dict] = None
):
    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx])
    elif isinstance(node_idx, list):
        node_idx = torch.tensor(node_idx)

    assert isinstance(edge_index, SparseTensor)

    subsets = [node_idx]
    
    for _ in range(hop):
        _, neighbor_idx = edge_index.sample_adj(subsets[-1], -1, replace=False)
        if (max_nodes_per_hop > 0):
            if len(neighbor_idx) > max_nodes_per_hop:
                if use_ppr == False: 
                    # Random sampling when ppr is not used 
                    neighbor_idx = neighbor_idx[torch.randperm(len(neighbor_idx))[:max_nodes_per_hop]]
                else:
                    # Compute sampling probabilities using ppr scores
                    scores = torch.tensor([ppr_scores[node.item()] for node in neighbor_idx], dtype=torch.float32)
                    probabilities = scores / scores.sum()
                    # Sample nodes based on probabilities and select sampled nodes
                    sampled_nodes = torch.multinomial(probabilities, num_samples=max_nodes_per_hop, replacement=False)
                    neighbor_idx = neighbor_idx[sampled_nodes]
                    
        subsets.append(neighbor_idx)

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:len(node_idx)]

    sub_edges = edge_index[subset, :][:, subset].coo()
    row, col, processed_edge_map = sub_edges
    edge_index = torch.stack([row, col], dim=0)

    node_count = subset.size(0)
    edge_index, processed_edge_map = coalesce(edge_index, processed_edge_map, node_count, node_count, "min")

    return subset, edge_index, inv, processed_edge_map

def k_hop_subgraph(
        node_idx: Union[int, list[int], Tensor],
        num_hops: int,
        edge_index: Tensor,
        max_nodes_per_hop=-1,
        relabel_nodes: bool = False,
        num_nodes: Optional[int] = None,
        directed: bool = False,
        use_ppr=True,
        ppr_scores: Optional[dict] = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes) 

    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, int):
        node_idx = [node_idx]
    elif isinstance(node_idx, Tensor):
        if len(node_idx.size()) == 0:
            node_idx = [node_idx.tolist()]
        else:
            node_idx = node_idx.tolist()

    subsets = []
    
    for node in node_idx:
        subsets.append(torch.tensor([node], device=row.device))
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            fringe = col[edge_mask]
            if max_nodes_per_hop > 0:
                if len(fringe) > max_nodes_per_hop:
                    if use_ppr == False:
                        # Random sampling when ppr is not used 
                        fringe = fringe[torch.randperm(len(fringe))[:max_nodes_per_hop]]
                    else: 
                        # Compute sampling probabilities using ppr scores
                        scores = torch.tensor([ppr_scores[node.item()] for node in fringe], dtype=torch.float32)
                        probabilities = scores / scores.sum()
                        # Sample nodes based on probabilities and select sampled nodes
                        sampled_nodes = torch.multinomial(probabilities, num_samples=max_nodes_per_hop, replacement=False)
                        fringe = fringe[sampled_nodes]
            subsets.append(fringe)

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:len(node_idx)]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes,), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask
