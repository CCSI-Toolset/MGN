from torch_geometric.data import Data, Dataset
from torch_geometric.data.dataset import IndexType, files_exist
from tqdm.auto import tqdm
import torch
import os
import numpy as np
from pathlib import Path
from os.path import join, exists
from typing import Union
from GNN.utils.data_utils import get_sim_time_from_processed_path, get_sim_level_info
from GNN.DatasetClasses.base import MGNDataset
from multiprocessing import Pool
import resource


class PNNL_Subdomain(MGNDataset):
    """
    We preprocess the raw PNNL data with either `PNNL_With_Node_Type.py` or
    `PNNL_Raw_Dynamic_Vars_and_Meshes.py`; the latter saves out minimal
    data and is useful for dealing with very large graphs (like 3D data).

    Whether you're using 2D or 3D data, the preprocessed graphs can be too
    large to fit into GPU memory, so we further preprocess the data using
    either `PNNL_Subdomain.py` or `PNNL_Dynamic.py`.

    Use PNNL_Subdomain in tandem with PNNL_With_Node_Type for smaller (
    e.g., non-3D) datasets. Unlike PNNL_Dynamic, this class saves out the
    patch files and thus does not preprocess the data during training.
    """

    def __init__(
        self,
        model_config,
        train_set,
        noise=None,
        noise_gamma=0.0,
        output_type="velocity",
        env=None,
        **kwargs,
    ):
        """
        model_config: configuration settings for the mode, an instance of GNN.ModelConfig.ConfigMGN.ModelConfig
        train_set: Boolean, True if this object represents a training set, False otherwise
        output_type: The variable(s) to be predicted (default: 'velocity')
        noise:  float > 0, noise to add for training
        noise_gamma: float > 0, noise parameter
        patch_mask: used for tailored, irregular patches
        patch_id: used for tailored, irregular patches
        env: the CustomLSFEnvironment object with rank info
        """
        self.x_range = model_config.PNNL_PARAMS[
            f'{"" if train_set else "test_"}' + "x_range"
        ]
        self.y_range = model_config.PNNL_PARAMS[
            f'{"" if train_set else "test_"}' + "y_range"
        ]
        self.three_d = max(["z_range" in k for k in model_config.PNNL_PARAMS])
        if self.three_d:
            print("\nWorking with 3D data!\n")
            self.z_range = model_config.PNNL_PARAMS[
                f'{"" if train_set else "test_"}' + "z_range"
            ]
        self.sim_range = model_config.PNNL_PARAMS[
            f'{"" if train_set else "test_"}' + "sim_range"
        ]
        self.time_range = model_config.PNNL_PARAMS[
            f'{"" if train_set else "test_"}' + "time_range"
        ]
        self.range_of_times = range(self.time_range[0], self.time_range[1] + 1)
        self.output_dim = model_config.PNNL_PARAMS["output_dim"]

        self.root = model_config.PNNL_PARAMS["root_dir"]
        if not files_exist(self.raw_paths):
            raise RuntimeError(
                f"'{self.__class__.__name__}' requires you to initially preprocess raw data, but {self.raw_dir} is empty."
                "See `GNN/DatasetClasses/README.md` or this class's docstring for more information."
            )
        print(f"\nUsing data from {self.root}")

        self.env = env
        self.model_config = model_config
        self.partition_method = model_config.PNNL_PARAMS["partition_method"]
        self.patch_mask = model_config.PNNL_PARAMS.setdefault("patch_mask", None)
        self.patch_id = model_config.PNNL_PARAMS.setdefault("patch_id", None)

        self.source_nodes = (
            model_config.source_node_types
        )  # should be range(1,7) for overfitting experiment
        self.node_type_onehot_dict = {
            "fluid": 0,
            "gas_inlet": 1,
            "inlet_to_outlet_wall": 2,
            "liquid_inlet": 3,
            "outlet": 4,
            "packing": 5,
            "side_wall": 6,
        }
        self.non_source_nodes = tuple(
            set(range(len(self.node_type_onehot_dict))) - set(self.source_nodes)
        )
        assert self.node_type_onehot_dict["fluid"] in self.non_source_nodes, (
            self.non_source_nodes,
            self.node_type_onehot_dict,
        )
        for i in self.source_nodes:
            print(
                f"\nUsing node type {list(self.node_type_onehot_dict.keys())[i]} as a source!!"
            )

        self.output_type = output_type
        # can skip some processing if all sims have the same x,y coordinates
        self.same_xy = model_config.PNNL_PARAMS["same_xy"]
        # replace vel with momentum -- not actually momentum, need densities for that, but this is close
        self.momentum = model_config.PNNL_PARAMS["momentum"]
        # set thickness (# of hops needed to traverse artificial boundary)
        self.thickness = model_config.PNNL_PARAMS.setdefault("thickness", 1)
        # knn, radius, or delaunay
        self.graph_type = model_config.graph_type
        # radius for radius graph
        self.radius = model_config.radius
        # k for knn graph
        self.k = model_config.k

        # partition method
        assert self.partition_method in ("range", "precomputed")
        assert (
            output_type == "velocity"
        ), "Only output type of velocity (or momentum) is supported."

        if self.partition_method == "range":
            assert self.x_range is not None and self.y_range is not None
        elif self.partition_method == "precomputed":
            assert self.patch_mask is not None and self.patch_id is not None
        else:
            raise ValueError(f"Partition method {self.partition_method} is not valid.")

        # Get name for patching method
        if self.partition_method == "range":
            if type(self.x_range) == list:
                assert (
                    type(self.x_range[0]) == tuple
                ), f"x_range should be a 2-item tuple or a list of tuples, you gave {self.x_range}"
                self.n_patches = len(self.x_range)
                self.patch_str = f"{self.partition_method}_{self.n_patches}patches_start_end_{self.x_range[0][0]:.5f}_{self.x_range[-1][1]:.5f}_{self.y_range[0][0]:.5f}_{self.y_range[-1][1]:.5f}"
            elif type(self.x_range) == str:  # automatically make patches
                if not self.three_d:
                    Nx, x_min, x_max = (float(x) for x in self.x_range.split(","))
                    Ny, y_min, y_max = (float(x) for x in self.y_range.split(","))
                    Nx, Ny = int(Nx), int(Ny)
                    self.x_range = sum(
                        (
                            [(start, end)] * Ny
                            for start, end in zip(
                                np.linspace(x_min, x_max, Nx + 1)[:-1],
                                np.linspace(x_min, x_max, Nx + 1)[1:],
                            )
                        ),
                        [],
                    )
                    self.y_range = [
                        (start, end)
                        for start, end in zip(
                            np.linspace(y_min, y_max, Ny + 1)[:-1],
                            np.linspace(y_min, y_max, Ny + 1)[1:],
                        )
                    ] * Nx
                    self.n_patches = len(self.x_range)
                    self.patch_str = f"{self.partition_method}_{self.n_patches}patches_start_end_{self.x_range[0][0]:.5f}_{self.x_range[-1][1]:.5f}_{self.y_range[0][0]:.5f}_{self.y_range[-1][1]:.5f}"
                else:
                    Nx, x_min, x_max = (float(x) for x in self.x_range.split(","))
                    Ny, y_min, y_max = (float(x) for x in self.y_range.split(","))
                    Nz, z_min, z_max = (float(x) for x in self.z_range.split(","))
                    Nx, Ny, Nz = int(Nx), int(Ny), int(Nz)
                    self.x_range = sum(
                        (
                            [(start, end)] * Nz * Ny
                            for start, end in zip(
                                np.linspace(x_min, x_max, Nx + 1)[:-1],
                                np.linspace(x_min, x_max, Nx + 1)[1:],
                            )
                        ),
                        [],
                    )
                    self.y_range = (
                        sum(
                            (
                                [(start, end)] * Nz
                                for start, end in zip(
                                    np.linspace(y_min, y_max, Ny + 1)[:-1],
                                    np.linspace(y_min, y_max, Ny + 1)[1:],
                                )
                            ),
                            [],
                        )
                        * Nx
                    )
                    self.z_range = (
                        [
                            (start, end)
                            for start, end in zip(
                                np.linspace(z_min, z_max, Nz + 1)[:-1],
                                np.linspace(z_min, z_max, Nz + 1)[1:],
                            )
                        ]
                        * Ny
                        * Nx
                    )
                    self.n_patches = len(self.x_range)
                    self.patch_str = f"{self.partition_method}_{self.n_patches}patches_start_end_{self.x_range[0][0]:.5f}_{self.x_range[-1][1]:.5f}_{self.y_range[0][0]:.5f}_{self.y_range[-1][1]:.5f}_{self.z_range[0][0]:.5f}_{self.z_range[-1][1]:.5f}"
            else:
                self.n_patches = 1
                assert (
                    len(self.x_range) == 2
                ), f"x_range should be a 2-item tuple or a list of tuples, you gave {self.x_range}"
                self.patch_str = f"{self.partition_method}_{self.x_range[0]:.5f}_{self.x_range[1]:.5f}_{self.y_range[0]:.5f}_{self.y_range[1]:.5f}"
                self.x_range = [self.x_range]
                self.y_range = [self.y_range]
        elif self.partition_method == "precomputed":
            self.patch_str = f"{self.partition_method}_patchid-{self.patch_id}"
        # add thickness and graph type to patch_str
        self.patch_str = (
            f"{self.patch_str}_thickness{self.thickness}_type{self.graph_type}"
        )

        # TODO: add a warning if min and max of data ranges do not cover the dataset

        # noise
        self.noise = noise
        self.noise_gamma = noise_gamma

        # process and save
        transform = None if "transform" not in kwargs else kwargs["transform"]
        pre_transform = (
            PNNL_Subdomain.pre_transform
            if "pre_transform" not in kwargs
            else kwargs["pre_transform"]
        )
        print(
            f"PNNL_Subdomain init: {self.patch_str}, {self.sim_range}, {self.time_range}, {'transform' in kwargs}, {'pre_transform' in kwargs}"
        )
        super().__init__(
            root=self.root, transform=transform, pre_transform=pre_transform
        )

        # load 'node_type_onehot_dict' if exists -- this should match the dict that has all node types
        if self.node_type_onehot_path.exists():
            constructed_dict = torch.load(self.node_type_onehot_path)
            for i, j in zip(
                self.node_type_onehot_dict.items(), constructed_dict.items()
            ):
                assert i == j, (
                    self.node_type_onehot_dict.items(),
                    constructed_dict.items(),
                )

        # register ._indices
        self._indices = list(range(len(self.processed_paths)))

        if hasattr(self, "update_function"):
            print("Overriding default update function of Dataset")

        # masks for feature normalization
        # normalize everything except for one hot features
        self.node_normalizer_mask = np.zeros(self.num_node_features, dtype=bool)
        self.node_normalizer_mask[self.output_dim :] = True
        self.edge_normalizer_mask = np.zeros(self.num_edge_features, dtype=bool)

        if self.n_patches == 1:
            self.rollout_info = self.base_rollout_info
        else:
            self.rollout_info = self.patch_rollout_info

        n_sims = self.num_sims
        n_steps = self.timesteps
        self.frames_per_patch = n_sims * n_steps

    @property
    def num_node_features(self) -> int:
        if "num_node_features" in self.model_config.PNNL_PARAMS:
            return self.model_config.PNNL_PARAMS["num_node_features"]
        if self.three_d:
            return 14
        return 13

    @property
    def num_edge_features(self) -> int:
        if "num_edge_features" in self.model_config.PNNL_PARAMS:
            return self.model_config.PNNL_PARAMS["num_edge_features"]
        if self.three_d:
            return 4
        return 3

    def node_count_dict(self):
        node_count_path = join(self.processed_dir, "node_count_dict")
        if exists(node_count_path):
            return torch.load(node_count_path)

        if self.env is None or self.env.rank == 0:
            print("building node count dictionary")
            node_count_dict = {}
            for patch, idx in enumerate(
                range(0, self.n_patches * self.frames_per_patch, self.frames_per_patch)
            ):
                d = torch.load(join(self.processed_dir, self.processed_file_names[idx]))
                # processed_file_names end like this: f'PNNL_Dynamic_patch{p}_sim{s}_time{t}.pt'
                assert patch == int(self.processed_file_names[idx].split("_")[2][5:])
                node_count_dict[patch] = d["data"].num_nodes
            print("saving node count dictionary")
            torch.save(node_count_dict, node_count_path)
        if self.env:
            torch.distributed.barrier()
        return torch.load(node_count_path)

    @staticmethod
    def update_function(
        mgn_output_np,
        output_type,
        current_state=None,
        previous_state=None,
        source_data=None,
    ):
        # update function; update momentum based on predicted change ('velocity'), predict pressure
        num_states = current_state.shape[-1]
        # mgn_output_np must correspond to current_state[...,:mgn_output_np.shape[-1]]. in other words,
        # with 'velocity' output_type, if mgn_output_np has 0 error then the following must hold:
        # ground_truth_1[...,:mgn_output_np.shape[-1]] == ground_truth_0[...,:mgn_output_np.shape[-1]] + mgn_output_np
        num_dynamic_features = mgn_output_np.shape[-1]

        with torch.no_grad():
            if output_type == "acceleration":
                assert current_state is not None
                assert previous_state is not None
                next_state = current_state
                c = current_state[..., :num_dynamic_features]
                p = previous_state[..., :num_dynamic_features]
                next_state[..., :num_dynamic_features] = 2 * c - p + mgn_output_np
            elif output_type == "velocity":
                assert current_state is not None
                next_state = current_state
                next_state[..., :num_dynamic_features] += mgn_output_np
            else:  # state
                next_state = mgn_output_np.copy()

            if type(source_data) is dict:
                for key in source_data:
                    next_state[key, :num_states] = source_data[key]
            elif type(source_data) is tuple:
                next_state[source_data[0], :num_dynamic_features] = source_data[1][
                    ..., :num_dynamic_features
                ]
            # else: warning?

        return next_state

    @property
    def num_sims(self):
        return len(self.sim_range)

    @property
    def timesteps(self):
        # we have 1 fewer due to the need to predict deltas
        return len(self.range_of_times[:-1])

    @property
    def node_type_filename(self):
        return "node_type_onehot_dict.pt"

    @property
    def node_type_onehot_path(self):
        return Path(self.processed_dir) / self.node_type_filename

    @property
    def raw_dir(self) -> str:
        return join(self.root, "processed")

    @property
    def raw_file_names(self):
        return [
            "PNNL_{}_{}.pt".format(s, t)
            for s in self.sim_range
            for t in self.range_of_times
        ]

    @property
    def processed_dir(self) -> str:
        if not self.momentum:
            return join(self.root, "processed", "subdomain", self.patch_str)
        return join(self.root, "processed", "subdomain_momentum", self.patch_str)

    @property
    def processed_file_names(self):
        processed_paths = []

        for p in range(self.n_patches):
            for s in self.sim_range:
                for t in self.range_of_times[:-1]:
                    processed_path = f"PNNL_Subdomain_patch{p}_sim{s}_time{t}.pt"
                    processed_paths.append(processed_path)

        return processed_paths

    def get_patch_ghost_index_dict(self, patch):
        # {'can_send': non-ghost-node global indices in patch,
        # 'will_need': ghost node global indices in patch,
        # 'node_global_idx_to_subset_idx': mapping from global index to patch index}

        # use cases are receiving and sending (pseudocode):
        # 1) patch_data[node_global_idx_to_subset_idx[will_need]] = received_data
        # 2) sent_data = patch_data[node_global_idx_to_subset_idx[requested_idx]]
        return torch.load(self.patch_ghost_index_dict_path(patch))

    def patch_ghost_index_dict_path(self, patch):
        return Path(self.processed_dir) / f"ghost_index_dict_patch_{patch}"

    def process(self):
        node_type_onehot_dict = None
        count = 0
        if self.pre_transform is None:
            return
        zipped_ranges = zip(self.x_range, self.y_range, [None] * len(self.x_range))
        isolated_vertex_simplices = None
        if self.three_d:
            isolated_vertex_simplices = [
                [1132416, 1127178, 1130880, 1126788],
                [457466, 458120, 462452, 462738],
                [1132321, 1130790, 1134165, 1133053],
                [1132487, 1133233, 1126806, 1133091],
            ]
            zipped_ranges = zip(self.x_range, self.y_range, self.z_range)
        for p, (x_range, y_range, z_range) in enumerate(zipped_ranges):
            patch_ghost_index_dict = None
            skip_patch, files_in_patch = False, self.num_sims * self.timesteps
            if exists(
                Path(self.processed_dir)
                / self.processed_file_names[(p + 1) * files_in_patch - 1]
            ):
                skip_patch = True
            if self.env != None:
                # have each rank work on separate patches in parallel
                if p % self.env.world_size != self.env.rank:
                    skip_patch = True
                print(
                    f'\nRank {self.env.rank} {"will" if skip_patch else "will not"} skip patch {p}'
                )
            if skip_patch:
                count += files_in_patch
                continue
            for s in tqdm(self.sim_range):
                # see if we processed this sim
                if exists(self.processed_paths[(count - 1) + self.timesteps]):
                    count += self.timesteps
                    continue
                current_frame = None

                def get_start():
                    L = 0
                    R = max(self.range_of_times)
                    if not exists(self.processed_paths[count]):
                        return 0
                    while L <= R:
                        m = (L + R) // 2
                        if exists(self.processed_paths[count + m + 1]):
                            L = m + 1
                        elif not exists(self.processed_paths[count + m]):
                            R = m
                        else:
                            return m + 1
                    assert False

                start = get_start()
                for t in tqdm(self.range_of_times):
                    # t_{k+1}-t_k  is saved at t=t_{k+1} at count k.
                    # at t_k, count will be k-1. if count k exists, skip to t=t_{k+1}
                    if t < start:
                        continue
                    if t == start:
                        count += start
                    # root = self.root.replace('/p/vast1/ccsi/PNNL_2020/t_equal_zero_and_pnnl_node_types', '/usr/workspace/ccsi/PNNL_2020/t_equal_zero_and_pnnl_node_types')
                    next_frame = torch.load(
                        join(self.root, "processed", "PNNL_{}_{}.pt".format(s, t))
                    )
                    if current_frame is None:
                        # if you don't have the same mesh on each sim, or if you are looking at a new patch...
                        if not self.same_xy or patch_ghost_index_dict is None:
                            (
                                edge_index,
                                edge_attr,
                                node_type,
                                pos,
                                relevant_nodes,
                                node_type_onehot_dict,
                                contact_angle,
                                patch_ghost_index_dict,
                                is_ghost_boolean_array,
                            ) = get_sim_level_info(
                                next_frame,
                                self.partition_method,
                                self.patch_mask,
                                self.patch_id,
                                x_range,
                                y_range,
                                z_range,
                                graph_type=self.graph_type,
                                radius=self.radius,
                                k=self.k,
                                thickness=self.thickness,
                                graph_path=Path(self.processed_dir) / "ptg_graph.pt"
                                if (self.env != None and self.same_xy)
                                else None,
                                env=self.env,
                                three_d=self.three_d,
                                isolated_vertex_simplices=isolated_vertex_simplices,
                            )
                            non_source = ~torch.logical_or(
                                is_ghost_boolean_array,
                                torch.isin(
                                    torch.argmax(node_type, dim=1),
                                    torch.tensor(self.source_nodes),
                                ),
                            )
                            # ensure the dict corresponding to the data we're using is the same as the dict with all node types
                            # if this fails, you might be using data different from the PNNL_2020 dataset, or you might have renamed/changed a node type
                            for i, j in zip(
                                self.node_type_onehot_dict.items(),
                                node_type_onehot_dict.items(),
                            ):
                                assert i == j, (
                                    self.node_type_onehot_dict.items(),
                                    node_type_onehot_dict.items(),
                                )
                        next_frame = next_frame.iloc[relevant_nodes]
                    else:
                        next_frame = next_frame.iloc[relevant_nodes]
                        data = self.pre_transform(
                            current_frame,
                            next_frame,
                            edge_index,
                            edge_attr,
                            node_type,
                            contact_angle,
                            pos,
                            self.momentum,
                            output_dim=self.output_dim,
                            non_source=non_source,
                            three_d=self.three_d,
                        )
                        # Save a subdomain graph per patch `p`, per sim `s`, and per time `t`
                        processed_path = self.processed_paths[count]
                        info_dict = get_sim_time_from_processed_path(processed_path)
                        # t-1 because when t=1 then we're saving data for t=0
                        assert (
                            info_dict["time"] == t - 1
                            and info_dict["sim"] == s
                            and info_dict["patch"] == p
                        ), f"Mismatch between processed_path {info_dict} and current patch {p} sim {s} time {t-1}."
                        torch.save(
                            {"data": data, "patch": p, "time": t - 1, "sim": s},
                            processed_path,
                        )
                        count += 1

                    current_frame = next_frame

            # save patch_ghost_index_dict
            torch.save(patch_ghost_index_dict, self.patch_ghost_index_dict_path(p))

        # save node_type_dict
        if self.env != None:
            if self.env.rank == 0:
                torch.save(node_type_onehot_dict, self.node_type_onehot_path)
            torch.distributed.barrier()
        else:
            torch.save(node_type_onehot_dict, self.node_type_onehot_path)

    # define __len__ and get for easier loading with DDP
    # to be used with DistributedSampler
    def __len__(self):
        # assumption: each processed path represents *one* graph
        return len(self.processed_paths)

    def get_non_source_data_mask(self, data):
        # return the nodes that are not scripted/source nodes.
        # NOTE that ghost nodes are counted as scripted/source, and they don't count towards loss
        if hasattr(data, "non_source"):
            non_source = data.non_source  # [num_nodes]
            (bs_times_num_nodes,) = non_source.shape
        else:
            non_source = torch.stack(
                [data[i].non_source for i in range(len(data))]
            )  # [bs, num_nodes]
            bs, num_nodes = non_source.shape
            bs_times_num_nodes = bs * num_nodes
        non_source_data_mask = non_source.view(-1)
        assert len(non_source_data_mask) == bs_times_num_nodes
        return non_source_data_mask

    def get(self, idx):
        data_dict = torch.load(self.processed_paths[idx])
        # data_dict: {data, time, sim}
        return data_dict["data"]

    # define __getitem__ to add noise
    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union["Dataset", Data]:

        data = self.get(idx)
        # data: Data(x, edge_index, edge_attr, y, pos, node_type num_nodes)
        # add noise to data here
        data.dataset_idx = idx

        if self.noise is None:
            return data
        else:
            outputs = data.y  # [num_nodes, out_features]
            node_data = data.x  # [num_nodes, in_features]
            non_scripted_node_mask = self.get_non_source_data_mask(data)

            num_nodes, out_features = outputs.shape
            assert len(self.noise) == out_features

            noise = np.tile(self.noise, (num_nodes, 1))
            noise = torch.tensor(np.random.normal(0, noise)).float()
            # input noise
            # no noise for scripted_nodes
            node_data[non_scripted_node_mask, :out_features] += noise[
                non_scripted_node_mask
            ]

            # output adjustment
            assert self.output_type == "velocity"
            outputs[non_scripted_node_mask, :out_features] -= (
                1.0 - self.noise_gamma
            ) * noise[non_scripted_node_mask]

            data.x = node_data
            data.y = outputs

        return data

    def get_numpy_features(self, i):
        return self.__getitem__(i).x.numpy()

    def patch_rollout_info(
        self, patch_num, sim_num, rollout_start_idx, window_length, sources
    ):
        assert rollout_start_idx == 0  # not supported for patches yet
        assert len(self.processed_file_names) % self.frames_per_patch == 0, (
            self.processed_file_names,
            self.frames_per_patch,
        )
        processed_sim_length = self.timesteps
        assert len(self.processed_file_names) % processed_sim_length == 0, (
            self.processed_file_names,
            processed_sim_length,
        )
        patch_start_idx = (
            patch_num * self.frames_per_patch + sim_num * processed_sim_length
        )
        graph = self.__getitem__(
            patch_start_idx
        )  # test_dataset has n_sim * sim_length elements
        node_coordinates = graph.pos.numpy()
        # for multiprocessing to work, setting resource limit per https://github.com/Project-MONAI/MONAI/issues/701
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
        n_processes = int(len(os.sched_getaffinity(0)) * 0.7)
        pool = Pool(processes=n_processes)
        indices = range(patch_start_idx, patch_start_idx + processed_sim_length)
        data = np.array(
            list(tqdm(pool.imap(self.get_numpy_features, indices), total=len(indices)))
        )
        input_len = data.shape[-1]
        initial_window_data = data[:window_length, :, :input_len]
        onehot_info = (False, len(self.node_type_onehot_dict))
        assert (
            len(self.node_type_onehot_dict) == 7
        ), f"one hot dict had {len(self.node_type_onehot_dict)} keys, expected 7 node types for PNNL"
        rollout_iterations = len(data) - window_length
        fixed_nodes = torch.isin(
            torch.argmax(graph.node_type, dim=1), torch.tensor(sources)
        )
        source_data = (fixed_nodes, data[window_length:, fixed_nodes, :input_len])
        update_function = self.update_function
        return {
            "initial_window_data": initial_window_data,
            "graph": graph,
            "node_types": None,
            "node_coordinates": node_coordinates,
            "onehot_info": onehot_info,
            "rollout_iterations": rollout_iterations,
            "source_data": source_data,
            "shift_source": False,
            "update_function": update_function,
            "num_nodes": len(fixed_nodes),
            "feature_size": input_len,
            "ground_truth": data,
        }

    def base_rollout_info(self, sim_num, rollout_start_idx, window_length, sources):
        processed_sim_length = self.timesteps
        assert len(self.processed_paths) % processed_sim_length == 0, (
            self.processed_paths,
            processed_sim_length,
        )
        sim_start_idx = sim_num * processed_sim_length
        graph = self.__getitem__(
            sim_start_idx
        )  # test_dataset has n_sim * sim_length elements
        node_coordinates = graph.pos.numpy()
        # for multiprocessing to work, setting resource limit per https://github.com/Project-MONAI/MONAI/issues/701
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
        n_processes = int(len(os.sched_getaffinity(0)) * 0.7)
        pool = Pool(processes=n_processes)
        indices = range(
            sim_start_idx + rollout_start_idx, sim_start_idx + processed_sim_length
        )
        data = np.array(
            list(tqdm(pool.imap(self.get_numpy_features, indices), total=len(indices)))
        )
        input_len = data.shape[-1]
        initial_window_data = data[:window_length, :, :input_len]
        onehot_info = (False, len(self.node_type_onehot_dict))
        rollout_iterations = len(data) - window_length
        fixed_nodes = torch.isin(
            torch.argmax(graph.node_type, dim=1), torch.tensor(sources)
        )
        source_data = (fixed_nodes, data[window_length:, fixed_nodes, :input_len])
        update_function = self.update_function
        return {
            "initial_window_data": initial_window_data,
            "graph": graph,
            "node_types": None,
            "node_coordinates": node_coordinates,
            "onehot_info": onehot_info,
            "rollout_iterations": rollout_iterations,
            "source_data": source_data,
            "shift_source": False,
            "update_function": update_function,
            "num_nodes": len(fixed_nodes),
            "feature_size": input_len,
            "ground_truth": data,
        }

    @staticmethod
    def pre_transform(
        current_frame,
        next_frame,
        edge_index,
        edge_attr,
        node_type,
        contact_angle,
        pos,
        momentum,
        output_dim,
        non_source,
        three_d,
    ):
        def get_delta(next_frame, current_frame, var):
            return (
                next_frame[var].values[:, np.newaxis]
                - current_frame[var].values[:, np.newaxis]
            )

        def get_momentum(vel, frac):
            # roughly, the density of the liquid and gas are 1010 and 1.18415, respectively.
            density = frac * 1010 + (1 - frac) * 1.18415
            density /= 1000
            # return (scaled) momentum per unit volume (density * vel)
            return density * vel

        if not momentum:
            input_cols = [
                "Velocity[i] (m/s)",
                "Velocity[j] (m/s)",
                "Volume Fraction of Liq",
            ]
            if three_d:
                input_cols = [
                    "Velocity[i] (m/s)",
                    "Velocity[j] (m/s)",
                    "Velocity[k] (m/s)",
                    "Volume Fraction of Liq",
                ]
        else:
            if not hasattr(current_frame, "momentum_i"):
                current_frame["momentum_i"] = get_momentum(
                    current_frame["Velocity[i] (m/s)"],
                    current_frame["Volume Fraction of Liq"],
                )
                current_frame["momentum_j"] = get_momentum(
                    current_frame["Velocity[j] (m/s)"],
                    current_frame["Volume Fraction of Liq"],
                )
                if three_d:
                    current_frame["momentum_k"] = get_momentum(
                        current_frame["Velocity[k] (m/s)"],
                        current_frame["Volume Fraction of Liq"],
                    )
            next_frame["momentum_i"] = get_momentum(
                next_frame["Velocity[i] (m/s)"], next_frame["Volume Fraction of Liq"]
            )
            next_frame["momentum_j"] = get_momentum(
                next_frame["Velocity[j] (m/s)"], next_frame["Volume Fraction of Liq"]
            )
            input_cols = ["momentum_i", "momentum_j", "Volume Fraction of Liq"]
            if three_d:
                next_frame["momentum_k"] = get_momentum(
                    next_frame["Velocity[k] (m/s)"],
                    next_frame["Volume Fraction of Liq"],
                )
                input_cols = [
                    "momentum_i",
                    "momentum_j",
                    "momentum_k",
                    "Volume Fraction of Liq",
                ]
        input_tensors = [
            torch.tensor(current_frame[col].values[:, np.newaxis]) for col in input_cols
        ]
        # add node_type to input
        input_tensors.append(node_type)
        if contact_angle is not None:
            input_tensors.append(contact_angle)
        x = torch.cat(input_tensors, dim=1).float()
        # y attribute is [num_nodes, *] (these are from Section A.1 of https://arxiv.org/pdf/2010.03409.pdf)
        if not three_d:
            y = torch.cat(
                [
                    torch.tensor(
                        get_delta(
                            next_frame,
                            current_frame,
                            "Velocity[i] (m/s)" if not momentum else "momentum_i",
                        )
                    ),
                    torch.tensor(
                        get_delta(
                            next_frame,
                            current_frame,
                            "Velocity[j] (m/s)" if not momentum else "momentum_j",
                        )
                    ),
                    torch.tensor(
                        get_delta(next_frame, current_frame, "Volume Fraction of Liq")
                    ),
                    torch.tensor(next_frame["Pressure (Pa)"].values[:, np.newaxis]),
                ],
                axis=1,
            ).float()
        else:
            y = torch.cat(
                [
                    torch.tensor(
                        get_delta(
                            next_frame,
                            current_frame,
                            "Velocity[i] (m/s)" if not momentum else "momentum_i",
                        )
                    ),
                    torch.tensor(
                        get_delta(
                            next_frame,
                            current_frame,
                            "Velocity[j] (m/s)" if not momentum else "momentum_j",
                        )
                    ),
                    torch.tensor(
                        get_delta(
                            next_frame,
                            current_frame,
                            "Velocity[k] (m/s)" if not momentum else "momentum_k",
                        )
                    ),
                    torch.tensor(
                        get_delta(next_frame, current_frame, "Volume Fraction of Liq")
                    ),
                    torch.tensor(next_frame["Pressure (Pa)"].values[:, np.newaxis]),
                ],
                axis=1,
            ).float()

        y = y[:, :output_dim]

        ptg_frame = Data(
            x=x.float(),
            y=y.float(),
            pos=pos,
            num_nodes=len(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=node_type,
            non_source=non_source,
        )

        return ptg_frame
