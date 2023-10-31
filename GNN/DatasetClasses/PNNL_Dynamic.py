from torch_geometric.data import Data
from torch_geometric.data.makedirs import makedirs
from torch import distributed as dist
import torch
import sys, pickle
import numpy as np
from pathlib import Path
from os.path import join, exists
from GNN.utils.data_utils import (
    get_sim_time_from_processed_path,
    get_sim_level_info,
    restrict_to_subgraph,
)
from GNN.DatasetClasses.PNNL_Subdomain import PNNL_Subdomain
from copy import deepcopy


class PNNL_Dynamic(PNNL_Subdomain):
    """
    We preprocess the raw PNNL data with either `PNNL_With_Node_Type.py` or
    `PNNL_Raw_Dynamic_Vars_and_Meshes.py`; the latter saves out minimal
    data and is useful for dealing with very large graphs (like 3D data).

    Whether you're using 2D or 3D data, the preprocessed graphs can be too
    large to fit into GPU memory, so we further preprocess the data using
    either `PNNL_Subdomain.py` or `PNNL_Dynamic.py`.

    PNNL_Dynamic is the relevant class when seeking to create patches from
    data initially preprocessed with `PNNL_Raw_Dynamic_Vars_and_Meshes.py`.
    Unlike its counterpart, `PNNL_Dynamic.py` does not save out the data
    for each simulation, timestep, and patch (due to the potential memory
    demands); instead, it loads the full domain data---which was initially
    preprocessed by `PNNL_Raw_Dynamic_Vars_and_Meshes.py`---for a given
    timestep *during training* and subsets it to the patch requested by
    the dataloader (i.e., preprocessing is done dynamically at training time).
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

        self.cached_paths = []
        # process and save
        print("PNNL_Dynamic init\n")
        super().__init__(
            model_config,
            train_set,
            noise=noise,
            noise_gamma=noise_gamma,
            output_type=output_type,
            env=env,
            transform=self.transform,
            pre_transform=PNNL_Dynamic.pre_transform,
            **kwargs,
        )

    def node_count_dict(self, mesh):
        node_count_path = join(self.processed_dir, f"node_count_dict_mesh{mesh}")
        if exists(node_count_path):
            return (torch.load(node_count_path))["all"]

        for sim, mesh_type in self.sim_to_mesh_type_dict.items():
            if sim not in self.sim_range:
                continue
            if mesh_type == mesh:
                sim_with_requested_mesh = sim
                break

        print(
            "Building node count dictionary...\n(this will take a while the first time.)"
        )
        if self.env is None:
            node_count_dict = {}
            in_patch_node_count_dict = {}
            for patch in range(self.n_patches):
                idx = (
                    patch * self.frames_per_patch
                    + self.sim_range.index(sim_with_requested_mesh) * self.timesteps
                )
                # processed_paths end like this: f'PNNL_Dynamic_patch{p}_sim{s}_time{t}.pt'
                assert patch == int(self.processed_paths[idx].split("_")[-3][5:])
                assert sim_with_requested_mesh == int(
                    self.processed_paths[idx].split("_")[-2][3:]
                )
                d = self.get(idx)
                node_count_dict[patch] = d.num_nodes
                in_patch_node_count_dict[patch] = d.non_source.sum()
        else:
            counts = torch.zeros(self.n_patches).to(f"cuda:{self.env.rank%4}")
            in_patch_counts = torch.zeros(self.n_patches).to(f"cuda:{self.env.rank%4}")
            for patch in range(self.n_patches):
                if patch % self.env.world_size != self.env.rank:
                    continue
                idx = (
                    patch * self.frames_per_patch
                    + self.sim_range.index(sim_with_requested_mesh) * self.timesteps
                )
                assert patch == int(self.processed_paths[idx].split("_")[-3][5:])
                assert sim_with_requested_mesh == int(
                    self.processed_paths[idx].split("_")[-2][3:]
                )
                d = self.get(idx)
                counts[patch] = d.num_nodes
                in_patch_counts[patch] = d.non_source.sum()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(in_patch_counts, op=dist.ReduceOp.SUM)
            node_count_dict = {i: count.item() for i, count in enumerate(counts.cpu())}
            in_patch_node_count_dict = {
                i: count.item() for i, count in enumerate(in_patch_counts.cpu())
            }

        print("saving node count dictionary")
        if self.env is None or self.env.rank == 0:
            torch.save(
                {
                    "all": node_count_dict,
                    "without_ghost_source": in_patch_node_count_dict,
                },
                node_count_path,
            )

        return node_count_dict

    @property
    def processed_dir(self) -> str:
        if not self.momentum:
            return join(self.root, "processed", "dynamic_patch_data", self.patch_str)
        return join(
            self.root, "processed", "dynamic_patch_data_momentum", self.patch_str
        )

    @property
    def processed_file_names(self):
        processed_paths = []
        for p in range(self.n_patches):
            for s in self.sim_range:
                for t in self.range_of_times[:-1]:
                    processed_path = f"PNNL_Dynamic_patch{p}_sim{s}_time{t}.pt"
                    processed_paths.append(processed_path)
        return processed_paths

    @property
    def processed_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        path = join(self.processed_dir, f"processed_paths{min(self.sim_range)}.pkl")
        if self.cached_paths:
            return self.cached_paths
        files = self.processed_file_names
        p = [join(self.processed_dir, f) for f in files]
        self.cached_paths = p
        pickle.dump(p, open(path, "wb"))
        return p

    def get_patch_ghost_index_dict(self, patch, sim):
        # {'can_send': non-ghost-node global indices in patch,
        # 'will_need': ghost node global indices in patch,
        # 'node_global_idx_to_subset_idx': mapping from global index to patch index}

        # use cases are receiving and sending (pseudocode):
        # 1) patch_data[node_global_idx_to_subset_idx[will_need]] = received_data
        # 2) sent_data = patch_data[node_global_idx_to_subset_idx[requested_idx]]
        return torch.load(
            self.subgraph_var_path(
                "ghost_index_dict", patch, self.sim_to_mesh_type_dict[sim]
            )
        )

    def subgraph_var_path(self, var, patch, mesh_type):
        path = (
            Path(self.processed_dir)
            / f"{var}"
            / f"mesh_type_{mesh_type}"
            / f"patch_{patch}"
        )
        path.parents[0].mkdir(parents=True, exist_ok=True)
        return path

    def files_exist(self):
        self.frame_path = Path(self.processed_dir) / "ptg_graph.pt"
        self.node_type_path = Path(self.processed_dir) / "node_type.pt"
        self.contact_angle_path = Path(self.processed_dir) / "contact_angle.pt"
        mesh_templates = torch.load(join(self.root, "processed", "PNNL_full_mesh.pt"))
        self.sim_to_mesh_type_dict = mesh_templates["sim_to_mesh_type_dict"]
        if (
            exists(self.frame_path)
            and exists(self.node_type_path)
            and exists(self.contact_angle_path)
        ):
            self.ptg_frame = {
                k: v
                for k, v in (torch.load(self.frame_path, map_location="cpu")).items()
            }
            # we only need the pos attribute at training time
            for mesh_type in self.ptg_frame:
                for var in self.ptg_frame[mesh_type].keys:
                    if var != "pos":
                        delattr(self.ptg_frame[mesh_type], var)
            print(f"\nLoaded the template meshes {self.ptg_frame}")
            self.node_type = torch.load(self.node_type_path, map_location="cpu")
            self.contact_angle = torch.load(self.contact_angle_path, map_location="cpu")
            return True
        return False

    def _process(self):
        if self.files_exist():
            return
        if self.log:
            print("Processing...", file=sys.stderr)
        makedirs(self.processed_dir)
        self.process()
        if self.log:
            print("Done!", file=sys.stderr)

    def process(self):
        # root = self.root.replace('/p/vast1/ccsi/PNNL_2020/t_equal_zero_and_pnnl_node_types', '/usr/workspace/ccsi/PNNL_2020/t_equal_zero_and_pnnl_node_types')
        mesh_templates = torch.load(join(self.root, "processed", "PNNL_full_mesh.pt"))
        mesh_dict = mesh_templates["mesh_dict"]
        self.sim_to_mesh_type_dict = mesh_templates["sim_to_mesh_type_dict"]
        node_types = {}
        contact_angles = {}
        ptg_frames = {}
        isolated_vertex_simplices = None
        if self.three_d:
            isolated_vertex_simplices = [
                [1132416, 1127178, 1130880, 1126788],
                [457466, 458120, 462452, 462738],
                [1132321, 1130790, 1134165, 1133053],
                [1132487, 1133233, 1126806, 1133091],
            ]
        for mesh_type, ptg_frame_no_edges in mesh_dict.items():
            # puts edges onto ptg_frame, returns one-hot versions of node_type and contact_angle variables
            (
                ptg_frame,
                node_type,
                contact_angle,
                node_type_onehot_dict,
            ) = get_sim_level_info(
                ptg_frame_no_edges,
                self.partition_method,
                self.patch_mask,
                self.patch_id,
                x_range=None,
                y_range=None,
                z_range=None,
                graph_type=self.graph_type,
                radius=self.radius,
                k=self.k,
                thickness=self.thickness,
                graph_path=Path(self.processed_dir) / f"ptg_graph_{mesh_type}.pt",
                env=self.env,
                three_d=self.three_d,
                return_mesh_info_only=True,
                isolated_vertex_simplices=isolated_vertex_simplices,
            )
            # ensure the dict corresponding to the data is the same as the dict with all node types
            # if this fails, you might be using data different from the PNNL_2020 dataset, or you might have renamed/changed a node type
            for i, j in zip(
                self.node_type_onehot_dict.items(), node_type_onehot_dict.items()
            ):
                assert i == j, (
                    self.node_type_onehot_dict.items(),
                    node_type_onehot_dict.items(),
                )

            # save node_type_dict
            if self.env != None:
                if self.env.rank == 0:
                    torch.save(node_type_onehot_dict, self.node_type_onehot_path)
                torch.distributed.barrier()
            else:
                torch.save(node_type_onehot_dict, self.node_type_onehot_path)

            node_types[mesh_type] = node_type.cpu()
            contact_angles[mesh_type] = contact_angle.cpu()
            ptg_frames[mesh_type] = ptg_frame.cpu()

        if self.env == None or self.env.rank == 0:
            torch.save(ptg_frames, self.frame_path)
            torch.save(node_types, self.node_type_path)
            torch.save(contact_angles, self.contact_angle_path)
        if self.env:
            torch.distributed.barrier()

        self.files_exist()

    def transform(self, sim, patch, time, patch_bounds):
        """
        loaded from raw data:
            current_frame, next_frame

        computed on the full, raw dataset then refined based on the patch:
            edge_index, edge_attr, node_type, contact_angle, pos, non_source

        variables set by user to modify processing:
            momentum, output_dim, three_d, thickness
        """
        subgraph_vars = {
            "edge_index": None,
            "edge_attr": None,
            "relevant_nodes": None,
            "ghost_index_dict": None,
            "is_ghost_boolean_array": None,
        }

        if all(
            [
                exists(
                    self.subgraph_var_path(var, patch, self.sim_to_mesh_type_dict[sim])
                )
                for var in subgraph_vars
            ]
        ):
            for var in subgraph_vars:
                subgraph_vars[var] = torch.load(
                    self.subgraph_var_path(var, patch, self.sim_to_mesh_type_dict[sim]),
                    map_location="cpu",
                )
        else:
            # need to load ptg_frame with pos AND edge_attr vars on it
            ptg_frame = torch.load(
                self.frame_path.__str__().replace(
                    ".pt", "_" + str(self.sim_to_mesh_type_dict[sim]) + ".pt"
                ),
                map_location="cpu",
            )
            (
                subgraph_vars["edge_index"],
                subgraph_vars["edge_attr"],
                _,
                _,
                subgraph_vars["relevant_nodes"],
                _,
                _,
                subgraph_vars["ghost_index_dict"],
                subgraph_vars["is_ghost_boolean_array"],
            ) = restrict_to_subgraph(
                ptg_frame,
                self.partition_method,
                self.three_d,
                self.thickness,
                # deepcopy(self.ptg_frame[self.sim_to_mesh_type_dict[sim]]), self.partition_method, self.three_d, self.thickness,
                deepcopy(self.node_type[self.sim_to_mesh_type_dict[sim]]),
                deepcopy(self.contact_angle[self.sim_to_mesh_type_dict[sim]]),
                self.node_type_onehot_dict,
                patch_mask=None,
                patch_id=None,  # not supported yet, use range to define patch
                x_range=patch_bounds[0],
                y_range=patch_bounds[1],
                z_range=patch_bounds[2],
            )

            # TODO: if loading these vars is slow, just load rel nodes and use it to find edge_index
            for var, value in subgraph_vars.items():
                torch.save(
                    value,
                    self.subgraph_var_path(var, patch, self.sim_to_mesh_type_dict[sim]),
                )
            del ptg_frame

        pos = self.ptg_frame[self.sim_to_mesh_type_dict[sim]].pos[
            subgraph_vars["relevant_nodes"]
        ]
        node_type = self.node_type[self.sim_to_mesh_type_dict[sim]][
            subgraph_vars["relevant_nodes"]
        ]
        contact_angle = self.contact_angle[self.sim_to_mesh_type_dict[sim]][
            subgraph_vars["relevant_nodes"]
        ]
        non_source = ~torch.logical_or(
            subgraph_vars["is_ghost_boolean_array"],
            torch.isin(torch.argmax(node_type, dim=1), torch.tensor(self.source_nodes)),
        )

        current_frame = {
            k: v[subgraph_vars["relevant_nodes"]]
            for k, v in torch.load(
                join(self.root, "processed", f"PNNL_{sim}_{time}.pt"),
                map_location="cpu",
            ).items()
        }
        next_frame = {
            k: v[subgraph_vars["relevant_nodes"]]
            for k, v in torch.load(
                join(self.root, "processed", f"PNNL_{sim}_{time+1}.pt"),
                map_location="cpu",
            ).items()
        }

        x = self.pre_transform(
            current_frame,
            next_frame,
            subgraph_vars["edge_index"],
            subgraph_vars["edge_attr"],
            node_type,
            contact_angle,
            pos,
            self.momentum,
            self.output_dim,
            non_source,
            self.three_d,
        )
        return x

    def get_time_sim_and_patch(self, idx):
        zipped_ranges = zip(self.x_range, self.y_range, [None] * len(self.x_range))
        if self.three_d:
            zipped_ranges = zip(self.x_range, self.y_range, self.z_range)
        files_in_patch = self.num_sims * self.timesteps
        t = self.range_of_times[idx % self.timesteps]
        s = self.sim_range[(idx // self.timesteps) % self.num_sims]
        p = idx // files_in_patch
        x_range, y_range, z_range = list(zipped_ranges)[p]
        processed_path = self.processed_paths[idx]
        info_dict = get_sim_time_from_processed_path(processed_path)
        assert (
            info_dict["time"] == t and info_dict["sim"] == s and info_dict["patch"] == p
        ), f"Mismatch between processed_path {info_dict} and current patch {p} sim {s} time {t}."
        return s, p, t, x_range, y_range, z_range

    def get(self, idx):
        sim, patch, time, *patch_bounds = self.get_time_sim_and_patch(
            idx
        )  # get subset info
        x = self.transform(sim, patch, time, patch_bounds)
        return x

    def patch_rollout_info(
        self, patch_num, sim_num, rollout_start_idx, window_length, sources
    ):
        assert rollout_start_idx == 0  # not supported for patches yet
        assert len(self.processed_paths) % self.frames_per_patch == 0, (
            self.processed_paths,
            self.frames_per_patch,
        )
        processed_sim_length = self.timesteps
        assert len(self.processed_paths) % processed_sim_length == 0, (
            self.processed_paths,
            processed_sim_length,
        )
        patch_start_idx = (
            patch_num * self.frames_per_patch + sim_num * processed_sim_length
        )
        graph = self.__getitem__(
            patch_start_idx
        )  # test_dataset has n_sim * sim_length elements
        node_coordinates = graph.pos.numpy()

        # Non-Pool Version
        x_data = graph.x.numpy()

        input_len = x_data.shape[-1]
        initial_window_data = x_data[:, :input_len]
        onehot_info = (False, len(self.node_type_onehot_dict))
        fixed_nodes = torch.isin(
            torch.argmax(graph.node_type, dim=1), torch.tensor(sources)
        )
        source_data = (fixed_nodes, x_data[fixed_nodes, :input_len])
        update_function = self.update_function
        return {
            "initial_window_data": initial_window_data,
            "graph": graph,
            "node_types": None,
            "node_coordinates": node_coordinates,
            "onehot_info": onehot_info,
            "source_data": source_data,
            "shift_source": False,
            "update_function": update_function,
            "num_nodes": len(fixed_nodes),
            "feature_size": input_len,
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
            return next_frame[var][:, np.newaxis] - current_frame[var][:, np.newaxis]

        def get_momentum(vel, frac):
            # roughly, the density of the liquid and gas are 1010 and 1.18415, respectively.
            density = frac * 1010 + (1 - frac) * 1.18415
            density /= 1000
            # return (scaled) momentum per unit volume (density * vel)
            return density * vel

        if not momentum:
            input_cols = ["velo_i", "velo_j", "vol_frac"]
            if three_d:
                input_cols = ["velo_i", "velo_j", "velo_k", "vol_frac"]
        else:
            if not hasattr(current_frame, "momentum_i"):
                current_frame["momentum_i"] = get_momentum(
                    current_frame["velo_i"], current_frame["vol_frac"]
                )
                current_frame["momentum_j"] = get_momentum(
                    current_frame["velo_j"], current_frame["vol_frac"]
                )
                if three_d:
                    current_frame["momentum_k"] = get_momentum(
                        current_frame["velo_k"], current_frame["vol_frac"]
                    )
            next_frame["momentum_i"] = get_momentum(
                next_frame["velo_i"], next_frame["vol_frac"]
            )
            next_frame["momentum_j"] = get_momentum(
                next_frame["velo_j"], next_frame["vol_frac"]
            )
            input_cols = ["momentum_i", "momentum_j", "vol_frac"]
            if three_d:
                next_frame["momentum_k"] = get_momentum(
                    next_frame["velo_k"], next_frame["vol_frac"]
                )
                input_cols = ["momentum_i", "momentum_j", "momentum_k", "vol_frac"]
        input_tensors = [current_frame[col][:, np.newaxis] for col in input_cols]
        # add node_type to input
        input_tensors.append(node_type)
        if contact_angle is not None:
            input_tensors.append(contact_angle)
        x = torch.cat(input_tensors, dim=1).float()
        # y attribute is [num_nodes, *] (these are from Section A.1 of https://arxiv.org/pdf/2010.03409.pdf)
        if not three_d:
            y = torch.cat(
                [
                    get_delta(
                        next_frame,
                        current_frame,
                        "velo_i" if not momentum else "momentum_i",
                    ),
                    get_delta(
                        next_frame,
                        current_frame,
                        "velo_j" if not momentum else "momentum_j",
                    ),
                    get_delta(next_frame, current_frame, "vol_frac"),
                ],
                axis=1,
            ).float()
        else:
            y = torch.cat(
                [
                    get_delta(
                        next_frame,
                        current_frame,
                        "velo_i" if not momentum else "momentum_i",
                    ),
                    get_delta(
                        next_frame,
                        current_frame,
                        "velo_j" if not momentum else "momentum_j",
                    ),
                    get_delta(
                        next_frame,
                        current_frame,
                        "velo_k" if not momentum else "momentum_k",
                    ),
                    get_delta(next_frame, current_frame, "vol_frac"),
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
