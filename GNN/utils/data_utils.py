from torch_geometric.transforms import (
    Compose,
    Cartesian,
    Distance,
    KNNGraph,
    RadiusGraph,
    ToUndirected,
    BaseTransform,
)  # DO NOT IMPORT DELAUNAY, USE OURS
from torch_geometric.utils import to_networkx
import numpy as np
from networkx import is_weakly_connected, connected_components
from warnings import warn
import torch
import re
from collections import OrderedDict
from torch_geometric.data import Data
from itertools import combinations
import os
from typing import TypeVar, Optional, Iterator
from math import ceil
import scipy.spatial

T_co = TypeVar("T_co", covariant=True)


class Delaunay(BaseTransform):
    r"""Computes the delaunay triangulation of a set of points
    (functional name: :obj:`delaunay`)."""

    def __call__(self, data: Data) -> Data:
        if data.pos.size(0) < 2:
            data.edge_index = torch.tensor(
                [], dtype=torch.long, device=data.pos.device
            ).view(2, 0)
        if data.pos.size(0) == 2:
            data.edge_index = torch.tensor(
                [[0, 1], [1, 0]], dtype=torch.long, device=data.pos.device
            )
        elif data.pos.size(0) == 3:
            data.face = torch.tensor(
                [[0], [1], [2]], dtype=torch.long, device=data.pos.device
            )
        if data.pos.size(0) > 3:
            pos = data.pos.cpu().numpy()
            # note: using qhull_options='QJ' with Delaunay can lead to splintering issues.
            tri = scipy.spatial.Delaunay(pos)
            face = torch.from_numpy(tri.simplices)

            data.face = face.t().contiguous().to(data.pos.device, torch.long)

        return data


class BoundedDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Based on https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py...

    Version of DistributedSampler that does not sample empty patches...

    Alternatively, could just use Dataset.indexselect to subset dataset to relevant indices
    (see https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py)

    Effectively, the code does the following:
        # in __init__, subset indices to those with nonzero patches. e.g., drop [3,4,5] from range(9):
        indices = np.array([0,1,2,6,7,8])
        # in __iter__, build rand indices using length of indices to inform valid values
        rand_indices = [0,4,5,2,3,1]
        # if num_ranks == 4, then this array would be extended for divisibility reasons, e.g.:
        rand_indices = [0,4,5,2,3,1,0,0]
        # at this point, we have the following (note the absence of [3,4,5]):
        indices[rand_indices] == array([0, 7, 8, 2, 6, 1, 0, 0])
        # subset to the rank's indices
        rank_rand_indices = rand_indices[my_rank::num_ranks]
        # return iterator over the rank's relevant indices
        return indices[rank_rand_indices]
    """

    def __init__(
        self,
        min_max: tuple,
        dataset: torch.utils.data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        max_count=1e11,
        min_count=1,
    ) -> None:
        assert drop_last == False
        if num_replicas is None or rank is None:
            raise ValueError("specify rank and num_replicas")
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.indices = []
        for sim_idx, sim in enumerate(self.dataset.sim_range):
            if not hasattr(self.dataset, "sim_to_mesh_type_dict"):
                node_counts = self.dataset.node_count_dict()
            else:
                node_counts = self.dataset.node_count_dict(
                    self.dataset.sim_to_mesh_type_dict[sim]
                )
            for patch in range(self.dataset.n_patches):
                if min_max[0] <= node_counts[patch] <= min_max[1]:
                    start = (
                        patch * self.dataset.frames_per_patch
                        + sim_idx * self.dataset.timesteps
                    )
                    assert patch == int(
                        self.dataset.processed_paths[start].split("_")[-3][5:]
                    )
                    assert sim == int(
                        self.dataset.processed_paths[start].split("_")[-2][3:]
                    )
                    self.indices += list(range(start, start + self.dataset.timesteps))
        self.indices = np.array(self.indices)
        self.dataset_length = len(self.indices)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = ceil(self.dataset_length / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_length, generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(self.dataset_length))  # type: ignore[arg-type]

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(self.indices[indices])


def get_row_edges(r):
    L = [(r[0], r[1]), (r[1], r[2]), (r[2], r[0])]
    return L


def tetrahedron_to_triangles(simplex):
    x = combinations(simplex, 3)
    return x


def get_delaunay_edges(
    simplices, non_fluid_indices, add=None, three_d=False, return_kept_idx=False
):
    kept_simplices = []
    kept_idx = []
    S = set(non_fluid_indices)
    for i, t in enumerate(simplices):
        if three_d:
            s = tetrahedron_to_triangles(t)
        else:
            s = [t]
        for triangle in s:
            differ = list(set(triangle).difference(S))
            if len(differ) > 0:
                kept_simplices.append(triangle)
                kept_idx.append(i)
    if add is not None:
        for t in add:
            s = tetrahedron_to_triangles(t)
            for triangle in s:
                if triangle not in kept_simplices:
                    kept_simplices.append(triangle)
    kept_simplices = np.array(kept_simplices)
    edges = list(map(lambda x: get_row_edges(x), kept_simplices))
    edges = [i for sublist in edges for i in sublist]
    if return_kept_idx:
        return np.array(edges).T, kept_idx
    return np.array(edges).T


def triangles_to_edges(triangle_cells):
    """
    Converts a list of triangles to edges.
    For every triangle (u, v, w), we add edges (u, v), (v,  w), (u, w).
    """
    edges = np.vstack(
        (
            triangle_cells[:, [0, 1]],
            triangle_cells[:, [1, 2]],
            triangle_cells[:, [2, 0]],
        )
    )
    receivers = np.amin(edges, axis=1)
    senders = np.amax(edges, axis=1)

    # remove duplicated edges
    unique_edges = np.unique(np.stack((senders, receivers), axis=1), axis=0)
    senders, receivers = unique_edges[:, 0].astype(int), unique_edges[:, 1].astype(int)

    return np.vstack(
        [
            np.expand_dims(np.hstack((senders, receivers)), axis=0),
            np.expand_dims(np.hstack((receivers, senders)), axis=0),
        ]
    )


def add_edges_to_nodes(
    graph,
    N_nodes,
    non_fluid_indices,
    graph_type="knn",
    radius=0.1,
    k=6,
    three_d=False,
    apply_cartesian_transform=True,
    apply_distance_transform=True,
    isolated_vertex_simplices=None,
):
    """This function uses code from fluidgnn/GNN/DatasetClasses/PNNL2020Dataset.py to set up the graph
    based on the user-given graph_type and any associated kwargs (e.g., the 'k' for a knn graph)
    """
    graph_type = graph_type
    transforms = []
    if apply_cartesian_transform:
        transforms.append(Cartesian(norm=False, cat=True))
    if apply_distance_transform:
        transforms.append(Distance(norm=False, cat=True))
    if graph_type in ["radius", "knn", "delaunay"]:
        if graph_type == "radius":
            # note: this construction may be slow due to a thresholding problem (see below)
            transforms.insert(
                0, RadiusGraph(radius, max_num_neighbors=N_nodes)
            )  # setting max_num_neighbors to num nodes guarantees undirected
        elif graph_type == "knn":
            transforms.insert(0, KNNGraph(k, force_undirected=True))
        else:
            # delaunay triangulation handled below
            pass
    else:
        raise Exception("invalid graph type")

    if graph_type != "delaunay":
        transforms = Compose(transforms)
        graph = transforms(graph)

    # additional graph refinements
    # 1) radius: handle thresholding problem
    # 2) delaunay: process faces/simplices, remove interior obstacle edges (done after figuring out if one-hot is necessary)
    # 3) all: remove other-to-source edges #future work: outlet nodes?
    if graph_type == "radius":
        drop_edges = (
            graph.edge_attr[:, -1].numpy() > radius
        )  # distance is last in list of transforms
        graph.edge_index = graph.edge_index[:, ~drop_edges].contiguous()
        graph.edge_attr = graph.edge_attr[~drop_edges].contiguous()
    elif graph_type == "delaunay":
        graph = Delaunay()(graph)
        add = []
        simplices = graph.face.numpy().T

        if isolated_vertex_simplices is not None:
            for isolated_vertex_simplex in isolated_vertex_simplices:
                if isolated_vertex_simplex in simplices.tolist():
                    add.append(isolated_vertex_simplex)
                    print(
                        f"\nforcing inclusion of all-boundary simplex {isolated_vertex_simplex} to avoid disconnected graph"
                    )
        edges = get_delaunay_edges(
            simplices, non_fluid_indices, add=add, three_d=three_d
        )
        graph.edge_index = torch.tensor(edges)
        graph.adj_t = None  # problem in earlier versions of pt geometric
        unique_edges = torch.unique(graph.edge_index, dim=1).size(1)
        print(
            f"\ngraph has {unique_edges} unique edges and {graph.edge_index.size(1)} edges."
        )
        transforms = Compose(
            [
                ToUndirected(reduce="mean"),
                Cartesian(norm=False, cat=True),
                Distance(norm=False, cat=True),
            ]
        )
        graph = transforms(graph)
        unique_edges = torch.unique(graph.edge_index, dim=1).size(1)
        print(
            f"\nafter transforms, graph has {unique_edges} unique edges and {graph.edge_index.size(1)} edges."
        )

    if not is_weakly_connected(to_networkx(graph)):
        warn("disconnected graph")
        subgraphs = sum(
            [
                list(a)
                for a in [
                    c
                    for c in sorted(
                        connected_components(to_networkx(graph).to_undirected()),
                        key=len,
                        reverse=True,
                    )
                ][1:]
            ],
            [],
        )
        if graph_type == "delaunay":
            for simplex in graph.face.numpy().T.tolist():
                for index in set(subgraphs):
                    if index in simplex:
                        print(
                            f"\ndisconnected index {index} found in simplex {simplex}"
                        )
        print(subgraphs)
        assert (
            False
        ), '\nfix your disconnected graph warning by connecting the above node indices with "add" arg above'

    if hasattr(graph, "face"):
        graph.face = None

    return graph


def get_sim_level_info(
    frame,
    partition_method,
    patch_mask=None,
    patch_id=None,
    x_range=None,
    y_range=None,
    z_range=None,
    graph_type="knn",
    radius=0.1,
    k=6,
    thickness=1,
    graph_path=False,
    env=None,
    three_d=False,
    return_mesh_info_only=False,
    isolated_vertex_simplices=None,
    x_coord="X (m)",
    y_coord="Y (m)",
    z_coord="Z (m)",
    fluid_node="fluid",
):
    """PNNL processing.

    Assuming that every sim has a constant (x,y) location for each node, then
    we can find the edge_attr and edge_index for eah sim's subdomain just once -- self.same_xy is used for this.

    This node_type dict should match across subdomains because we do this BEFORE doing the subdomain
    so it has all the node types possible.
    """
    one_hot_dict = OrderedDict()
    for i, key in enumerate(sorted(frame.node_type.unique())):
        one_hot_dict[key] = i
    node_type = torch.nn.functional.one_hot(
        torch.tensor([one_hot_dict[t] for t in frame.node_type])
    )
    assert node_type.shape[1] == len(one_hot_dict.keys())

    # add contact angle
    angle_dict = OrderedDict()
    for i, key in enumerate(sorted(frame.contact_angle.unique())):
        angle_dict[key] = i
    contact_angle = torch.nn.functional.one_hot(
        torch.tensor([angle_dict[t] for t in frame.contact_angle])
    )

    # PTG Graph node construction
    if three_d:
        pos = torch.cat(
            [
                torch.tensor(frame[x_coord].values[:, np.newaxis]),
                torch.tensor(frame[y_coord].values[:, np.newaxis]),
                torch.tensor(frame[z_coord].values[:, np.newaxis]),
            ],
            axis=1,
        )
    else:
        pos = torch.cat(
            [
                torch.tensor(frame[x_coord].values[:, np.newaxis]),
                torch.tensor(frame[y_coord].values[:, np.newaxis]),
            ],
            axis=1,
        )
    ptg_frame = Data(pos=pos, num_nodes=len(pos))

    # PTG Graph edge construction
    non_fluid_indices = np.where(frame.node_type != fluid_node)[0]
    print(
        f"\n{len(non_fluid_indices)/len(frame.node_type)*100}% of indices are non-fluid for graph {graph_path}"
    )
    if graph_path:
        if env is None:
            if not os.path.exists(graph_path):
                ptg_frame = add_edges_to_nodes(
                    ptg_frame,
                    len(frame[x_coord]),
                    non_fluid_indices,
                    graph_type=graph_type,
                    radius=radius,
                    k=k,
                    three_d=three_d,
                    isolated_vertex_simplices=isolated_vertex_simplices,
                )
                torch.save(ptg_frame, graph_path)
        else:
            had_to_build_graph = 0
            if env.rank == 0 and not os.path.exists(graph_path):
                had_to_build_graph = 1
                ptg_frame = add_edges_to_nodes(
                    ptg_frame,
                    len(frame[x_coord]),
                    non_fluid_indices,
                    graph_type=graph_type,
                    radius=radius,
                    k=k,
                    three_d=three_d,
                    isolated_vertex_simplices=isolated_vertex_simplices,
                )
                torch.save(ptg_frame, graph_path)
            if not os.path.exists(graph_path) or had_to_build_graph:
                print(f"Rank {env.rank} waiting for graph to be made by rank 0")
                torch.distributed.barrier()
        ptg_frame = torch.load(graph_path)
    else:
        ptg_frame = add_edges_to_nodes(
            ptg_frame,
            len(frame[x_coord]),
            non_fluid_indices,
            graph_type=graph_type,
            radius=radius,
            k=k,
            three_d=three_d,
            isolated_vertex_simplices=isolated_vertex_simplices,
        )

    if return_mesh_info_only:
        return ptg_frame, node_type, contact_angle, one_hot_dict

    return restrict_to_subgraph(
        ptg_frame,
        partition_method,
        three_d,
        thickness,
        node_type,
        contact_angle,
        one_hot_dict,
        patch_mask=patch_mask,
        patch_id=patch_id,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    )


def restrict_to_subgraph(
    ptg_frame,
    partition_method,
    three_d,
    thickness,
    node_type,
    contact_angle,
    one_hot_dict,
    patch_mask=None,
    patch_id=None,
    x_range=None,
    y_range=None,
    z_range=None,
):
    """
    Warning: modifies ptg_frame in place
    """
    # restrict to subdomain
    if partition_method == "range":
        if three_d:
            subdomain = torch.tensor(
                np.logical_and.reduce(
                    (
                        np.logical_and(
                            x_range[0] <= ptg_frame.pos[:, 0],
                            ptg_frame.pos[:, 0] <= x_range[1],
                        ).numpy(),
                        np.logical_and(
                            y_range[0] <= ptg_frame.pos[:, 1],
                            ptg_frame.pos[:, 1] <= y_range[1],
                        ).numpy(),
                        np.logical_and(
                            z_range[0] <= ptg_frame.pos[:, 2],
                            ptg_frame.pos[:, 2] <= z_range[1],
                        ).numpy(),
                    )
                )
            )
        else:
            subdomain = np.logical_and(
                np.logical_and(
                    x_range[0] <= ptg_frame.pos[:, 0], ptg_frame.pos[:, 0] <= x_range[1]
                ),
                np.logical_and(
                    y_range[0] <= ptg_frame.pos[:, 1], ptg_frame.pos[:, 1] <= y_range[1]
                ),
            )
    elif partition_method == "precomputed":
        patch_mask = torch.tensor(patch_mask, dtype=torch.long)
        subdomain = patch_mask == patch_id
        subdomain = subdomain.long()

    # find subdomain nodes, then add ghost-node padding to them by expanding relevant_indices
    relevant_indices = subdomain_indices = torch.nonzero(subdomain)
    subdomain_node = np.isin(ptg_frame.edge_index[0, :], relevant_indices)

    # augment the subdomain (i.e., the set of relevant_indices) to include indices within "thickness" hops of a relevant node
    for _ in range(thickness):
        # make relevant all relevant node's neighbors -- do this `thickness` times
        relevant_indices = torch.unique(
            ptg_frame.edge_index[
                0,
                np.logical_or(
                    np.isin(ptg_frame.edge_index[0, :], relevant_indices),
                    np.isin(ptg_frame.edge_index[1, :], relevant_indices),
                ),
            ],
            sorted=True,
        )
    relevant_source_node = np.isin(ptg_frame.edge_index[0, :], relevant_indices)
    ghost_nodes = torch.unique(
        ptg_frame.edge_index[0, np.logical_and(relevant_source_node, ~subdomain_node)],
        sorted=True,
    )
    print(
        f"\nSubdomain has {subdomain.sum()} nodes, and {len(ghost_nodes)} ghost nodes."
    )

    # remove inflow nodes from the ghost nodes because we can script them instead of communicating them
    inflow_nodes = np.nonzero(node_type[:, one_hot_dict["liquid_inlet"]] == 1)
    ghost_nodes = ghost_nodes[~np.isin(ghost_nodes, inflow_nodes)]

    # remove from edge data now-irrelevant information
    relevant_neighbor_node = np.isin(ptg_frame.edge_index[1, :], relevant_indices)
    ptg_frame.edge_index = ptg_frame.edge_index[
        :, np.logical_and(relevant_source_node, relevant_neighbor_node)
    ]
    ptg_frame.edge_attr = ptg_frame.edge_attr[
        np.logical_and(relevant_source_node, relevant_neighbor_node)
    ]

    # restrict vars to relevant subsets
    pos = ptg_frame.pos[relevant_indices]
    node_type = node_type[relevant_indices]
    contact_angle = contact_angle[relevant_indices]

    # convert edge_index to the right subset indices
    node_global_idx_to_subset_idx = {
        idx.item(): i for i, idx in enumerate(relevant_indices)
    }
    # ^this dict^ and relevant_indices need to be sorted, and they are because we use python 3.7 and torch.unique, respectively.
    # TODO: use OrderedDict for compatibility
    subset_edge_idx = ptg_frame.edge_index
    for old_idx, new_idx in node_global_idx_to_subset_idx.items():
        subset_edge_idx[subset_edge_idx == old_idx] = new_idx

    ptg_frame.edge_index = subset_edge_idx

    # make dictionary with list of indices that can be sent and list of indices that are ghosts (will be received)
    patch_ghost_index_dict = {
        "can_send": subdomain_indices.squeeze(1),
        "will_need": ghost_nodes,
        "node_global_idx_to_subset_idx": node_global_idx_to_subset_idx,
    }

    is_ghost_boolean_array = torch.isin(relevant_indices, ghost_nodes)

    return (
        ptg_frame.edge_index,
        ptg_frame.edge_attr.float(),
        node_type,
        pos.float(),
        relevant_indices,
        one_hot_dict,
        contact_angle,
        patch_ghost_index_dict,
        is_ghost_boolean_array,
    )


def get_sim_time_from_processed_path(path):
    """This utility method is used to check if sim and time information from a processed .pt file
    matches with its processed path -- it's unlikely but need to check to be sure
    """
    # assumption: processed_path is in this form
    # processed_path = f'PNNL_Subdomain_patch{p}_sim{s}_time{t}.pt'
    # check @processed_file_names above
    def get_sim():
        match = re.search(r"_sim(\d+)", path)
        if match is None:
            raise ValueError(f"No sim is found in processed path: {path}")
        return float(match.group(0)[4:])

    def get_time():
        match = re.search(r"_time(\d+)", path)
        if match is None:
            raise ValueError(f"No time is found in processed path: {path}")
        return float(match.group(0)[5:])

    def get_patch():
        match = re.search(r"_patch(\d+)", path)
        if match is None:
            raise ValueError(f"No sim is found in processed path: {path}")
        return float(match.group(0)[6:])

    return {"patch": get_patch(), "sim": get_sim(), "time": get_time()}


def k_hop_subgraph(node_idx, num_hops, edge_index):
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops, ignoring directed edges.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, and (3) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (:obj:`torch.Tensor`): The central seed node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`)
    """
    relevant_indices = node_idx
    for _ in range(num_hops):
        # make relevant all relevant node's neighbors -- do this `thickness` times
        relevant_indices = torch.unique(
            edge_index[
                0,
                np.logical_or(
                    np.isin(edge_index[0, :], relevant_indices),
                    np.isin(edge_index[1, :], relevant_indices),
                ),
            ],
            sorted=True,
        )
    # preserved edges
    relevant_source_node = np.isin(edge_index[0, :], relevant_indices)
    relevant_neighbor_node = np.isin(edge_index[1, :], relevant_indices)
    edge_mask = np.logical_and(relevant_source_node, relevant_neighbor_node)
    edge_index = edge_index[:, edge_mask]
    return relevant_indices, edge_index, torch.from_numpy(edge_mask)
