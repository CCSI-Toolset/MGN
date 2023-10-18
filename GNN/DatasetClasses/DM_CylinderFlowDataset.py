import os

# import sys
import os.path as osp
import pickle

import numpy as np
import torch
from torch import no_grad
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import Cartesian, Distance, Compose

from GNN.utils.mgn_utils import get_sample
from GNN.utils.partition_utils import get_partitioner
from GNN.utils.data_utils import add_edges_to_nodes, triangles_to_edges
from GNN.DatasetClasses.base import MGNDataset


class DM_CylinderFlowDataset(MGNDataset):
    """
    Class to represent the Cylinder Flow dataset used in
    Pfaff, Tobias, et al. "Learning Mesh-Based Simulation with Graph Networks." International Conference on Learning Representations.
    """

    def __init__(
        self,
        model_config,
        train_set,
        output_type="velocity",
        window_length=1,
        noise=None,
        noise_gamma=0.1,
        apply_onehot=False,
        apply_partitioning=False,
        normalize=True,
        shift_source=True,
        **kwargs,
    ):
        """
        model_config: configuration settings for the mode, an instance of GNN.ModelConfig.ConfigMGN.ModelConfig
        train_set: Boolean, True if this object represents a training set, False otherwise
        output_type: The variable(s) to be predicted (default: 'velocity')
        window_length: int, output at time t will be predicted using observations in window [t-window_length, t - 1] (default: 1)
        noise:  float > 0, noise to add for training
        noise_gamma: float > 0, noise parameter
        apply_onehot: Boolean, whether to convert node type categorical variable to one-hot vector (default: False)
        apply_partitioning: Boolean, whether to partition the mesh (used for large meshes) (default: False)
        normalize: Boolean, whether to Standardize data to Normal(0, 1) (default: True)
        shift_source: Boolean, whether to use data from source nodes at time t (True) or at time t-1 (False) to predict output at t
        """
        dataset_params = model_config.DM_CYLINDER_FLOW_PARAMS
        # prepare train / test sets, DM Cylinder Flow
        if train_set:
            self.direc = dataset_params["train_input_dir"]
            self.sample_idx = list(
                range(
                    dataset_params["train_sample_idx_start"],
                    dataset_params["train_sample_idx_end"],
                )
            )
            self._processed_dir = dataset_params["train_processed_dir"]
        else:
            self.direc = dataset_params["test_input_dir"]
            self.sample_idx = list(
                range(
                    dataset_params["test_sample_idx_start"],
                    dataset_params["test_sample_idx_end"],
                )
            )
            self._processed_dir = dataset_params["test_processed_dir"]
        self.num_sims = len(self.sample_idx)
        # whether to apply onehot encoding to node type
        # and various attributes to keep track of node types of interest
        self.apply_onehot = apply_onehot
        self.normal_node_types = torch.tensor(model_config.get_normal_node_types())
        self.boundary_node_types = torch.tensor(model_config.get_boundary_node_types())
        self.source_node_types = torch.tensor(model_config.get_source_node_types())
        self.num_node_types = model_config.get_num_node_types()
        # later, we will relabel node types to be in range(0, K), where K is the
        # number of node types
        self.node_types_relabel_map = {
            orig: new for new, orig in enumerate(dataset_params["node_types_relabel"])
        }
        # output type and normalization
        self.output_type = output_type
        assert (output_type != "acceleration") or (window_length >= 2)
        self.window_length = window_length
        self.shift_source = shift_source
        self.normalize = normalize
        # graph types and parameters
        self.graph_types = model_config.get_graph_types()
        self.radius = model_config.get_radius()
        self.k = model_config.get_k()
        # partitioning settings
        self.force_preprocess = dataset_params["force_preprocess"]
        # only apply partitioning to the train set
        # on the test set, we always evaluate on full domain
        self.apply_partitioning = apply_partitioning if train_set else False
        if self.apply_partitioning:
            pparams = model_config.get_partitioning_params()
        else:
            pparams = {"partitioning_method": "null"}
        self.partitioner = get_partitioner(pparams)
        self.processed_subdir = str(self.partitioner)
        self._processed_dir = osp.join(self._processed_dir, self.processed_subdir)

        # Add noise
        self.noise = noise
        self.noise_gamma = noise_gamma

        # prepare files
        self.process()
        self.dataset_length = self._traj_start_idx[-1]
        self.output_dim = self[0].y.shape[-1]

        # masks for feature normalization
        # normalize everything except for one hot features
        # num_node_features and num_edge_features are being set under the hood in the
        # pytorch geometric Dataset class
        self.node_normalizer_mask = np.zeros(self.num_node_features, dtype=bool)
        if self.apply_onehot:
            self.node_normalizer_mask[-self.num_node_types :] = True
        else:
            self.node_normalizer_mask[-1] = True
        self.edge_normalizer_mask = np.zeros(self.num_edge_features, dtype=bool)

        def update_function(
            mgn_output_np,
            output_type,
            current_state=None,
            previous_state=None,
            source_data=None,
        ):
            """
            update momentum based on predicted change
            """
            num_states = current_state.shape[-1]

            with no_grad():
                if output_type == "acceleration":
                    assert current_state is not None
                    assert previous_state is not None
                    next_state = 2 * current_state - previous_state + mgn_output_np
                elif output_type == "velocity":
                    assert current_state is not None
                    next_state = current_state + mgn_output_np
                else:  # state
                    next_state = mgn_output_np.copy()

                if type(source_data) is dict:
                    for key in source_data:
                        next_state[key, :num_states] = source_data[key]
                elif type(source_data) is tuple:
                    next_state[source_data[0], :num_states] = source_data[1]

            return next_state

        self.update_function = update_function

        super().__init__()

    @property
    def processed_dir(self) -> str:
        """
        where to read/write preprocessed files for partitioning training
        """
        return self._processed_dir

    @property
    def processed_file_names(self) -> str:
        """
        files with information for partitioning, one file per simulation
        """
        return self._processed_file_names

    @property
    def processed_paths(self) -> str:
        """
        same as processed_file_names but prefixed by processed_dir
        """
        return self._processed_paths

    def process(self):
        """
        Function to prepare files with the information needed for partitioning training
        If files already exist, we don't recreate them
        """

        def fname_to_ints(fname):
            """
            Helper function to extract simulation ID, number of timesteps, and number of patches
            from preprocessed file names
            """
            # file names have the format "CylinderFlow_sim###_nsteps###_npatches###.pt"
            fname = fname.replace("CylinderFlow_", "").replace(".pt", "")
            fname = (
                fname.replace("sim", "").replace("nsteps", "").replace("npatches", "")
            )
            traj_idx, n_timesteps, n_patches = tuple([int(i) for i in fname.split("_")])
            return traj_idx, n_timesteps, n_patches

        # check if data has been processed already
        # in that case, populate the file data structures
        if osp.exists(self.processed_dir) and not self.force_preprocess:
            processed_file_names = os.listdir(self.processed_dir)
            processed_file_names = [
                f
                for f in processed_file_names
                if f.endswith(".pt") and fname_to_ints(f)[0] in self.sample_idx
            ]
            # check that all the processed files are there
            # one file per simulation
            assert len(processed_file_names) == self.num_sims, (
                "inconsistent number of preprocessed file. Expected %s, but found %s"
                % (self.num_sims, len(processed_file_names))
            )
            # then check that each file has the correct number of patches
            # and that the expected number of time steps is consistent with the simulation data
            for fname in processed_file_names:
                with open(osp.join(self.processed_dir, fname), "rb") as f:
                    expected_patches = len(pickle.load(f))
                traj_idx, n_timesteps, n_patches = fname_to_ints(fname)
                data_filename = self.direc + "sample_%s.npy" % (traj_idx)
                data_sample = np.load(data_filename, allow_pickle=True).item()
                expected_timesteps = (
                    data_sample["velocity"].shape[0] - self.window_length - 1
                )
                assert n_timesteps == expected_timesteps, (
                    "inconsistent number of timesteps in preprocessed file %s " % fname
                )
                assert n_patches == expected_patches, (
                    "inconsistent number of patches in preprocessed file %s" % fname
                )

            # sort files by simulation ID
            self._processed_file_names = sorted(processed_file_names, key=fname_to_ints)
            self._processed_paths = [
                osp.join(self.processed_dir, fname)
                for fname in self.processed_file_names
            ]
            self._traj_start_idx = [0]
            cur_traj_idx = 0
            for fname in self.processed_file_names:
                traj_idx, n_timesteps, n_patches = fname_to_ints(fname)
                cur_traj_idx += n_timesteps * n_patches
                self._traj_start_idx.append(cur_traj_idx)
        else:
            if not os.path.exists(self.processed_dir):
                os.makedirs(self.processed_dir)
            self._processed_file_names = []
            self._processed_paths = []
            self._traj_start_idx = [0]
            cur_traj_idx = 0
            for traj_idx in self.sample_idx:
                graph = self.get_graph_for_simulation(traj_idx)
                # get patches
                patches = self.partitioner.get_partition(
                    graph, num_edge_types=len(self.graph_types)
                )
                n_patches = len(patches)
                n_timesteps = graph.output.shape[0] - self.window_length - 1
                processed_file_name = f"CylinderFlow_sim{traj_idx}_nsteps{n_timesteps}_npatches{n_patches}.pt"
                processed_path = osp.join(self.processed_dir, processed_file_name)
                with open(processed_path, "wb") as f:
                    pickle.dump(patches, f)
                self._processed_file_names.append(processed_file_name)
                self._processed_paths.append(processed_path)
                cur_traj_idx += n_timesteps * n_patches
                self._traj_start_idx.append(cur_traj_idx)

    def __len__(self):
        """
        number of timesteps per simulation x number of simulations if apply_partioning is False
        otherwise, \sum_{i=1}^{N} n_patches_i x n_timesteps_per_simulation, where the sum is over N simulations
        """
        return self.dataset_length

    def get_non_source_data_mask(self, data):
        """
        returns a boolean tensor, where the ith entry is True if and only if the output for the ith node in the
        mesh should contribute to the computation of the loss
        """
        if type(data) is list:
            mask = ~torch.cat(
                [data_sample.ignore_for_loss_mask for data_sample in data]
            )
        else:
            mask = ~data.ignore_for_loss_mask
        return mask

    def _relabel_node_types(self, node_types):
        """
        relabel the node types, so that they are in the range [0, num_node_types)
        """
        new_node_types = torch.zeros_like(node_types)
        for orig_label, new_label in self.node_types_relabel_map.items():
            new_node_types[node_types == orig_label] = new_label
        return new_node_types

    def __getitem__(self, idx):
        """
        Returns a torch_geometric Dataset for the idx-th item (i.e., a slice of time in a particular simulation)
        """
        # make sure we have a valid index
        assert 0 <= idx < self.dataset_length
        # use binary search to find the simulation file containing the idx-th item
        traj_idx = np.searchsorted(self._traj_start_idx, idx, side="right") - 1
        # read time and patch slice from file, see self.process() above for details on the contents of this file
        path = self.processed_paths[traj_idx]
        fname = os.path.basename(path)
        fname = fname.replace("CylinderFlow_", "").replace(".pt", "")
        fname = fname.replace("sim", "").replace("nsteps", "").replace("npatches", "")
        _, _, n_patches = [int(i) for i in fname.split("_")]
        local_idx = idx - self._traj_start_idx[traj_idx]
        time_idx = local_idx // n_patches
        patch_idx = local_idx % n_patches
        with open(path, "rb") as f:
            patches = pickle.load(f)
            padded_patch, edge_masks, internal_node_mask = patches[patch_idx]
            # force edge_masks to be a list in the case of a single edge type
            # this is for backwards compatibility
            if len(self.graph_types) == 1:
                edge_masks = [edge_masks]
        graph = self.get_graph_for_simulation(traj_idx)
        data = graph.output
        node_types = graph.node_types
        # Find the inflow nodes
        source_nodes = torch.where(torch.isin(node_types, self.source_node_types))[0]
        # get node data for this time point
        node_data, outputs = get_sample(
            data,
            source_node_idx=source_nodes,
            time_idx=time_idx,
            window_length=self.window_length,
            output_type=self.output_type,
            noise_sd=self.noise,
            noise_gamma=self.noise_gamma,
            shift_source=self.shift_source,
        )
        node_data = node_data[0]
        node_data = torch.from_numpy(node_data).float()
        outputs = torch.from_numpy(outputs).float()
        # subselect node types
        patch_node_types = self._relabel_node_types(node_types[padded_patch])
        onehot = F.one_hot(patch_node_types, num_classes=self.num_node_types)
        # prepare node data by subselecting the nodes in this patch and appending node type
        patch_node_data = node_data[padded_patch]
        patch_node_data = torch.cat([patch_node_data, onehot], dim=1)
        # outputs
        patch_outputs = outputs[padded_patch]
        # prepare mask for nodes to ignore when computing loss
        # True if we should exclude a node when computing loss, False otherwise
        # exclude boundary and source nodes
        boundary_nodes = padded_patch[~internal_node_mask]
        ignore_for_loss_mask = torch.isin(padded_patch, boundary_nodes) | torch.isin(
            padded_patch, source_nodes
        )
        patch_graph = Data(
            x=patch_node_data,
            y=patch_outputs,
            num_nodes=len(padded_patch),
            internal_node_mask=internal_node_mask,
            ignore_for_loss_mask=ignore_for_loss_mask,
            dataset_idx=idx,
        )
        # add edges
        num_nodes = graph.pos.shape[0]
        for i, edge_mask in enumerate(edge_masks):
            # relabel edge indices to be in range(len(padded_patch))
            # subselect edges and attributes
            patch_edge_index = graph["edge_index_%s" % i][:, edge_mask]
            patch_edge_attr = graph["edge_attr_%s" % i][edge_mask]
            node_idx = patch_edge_index.new_full((num_nodes,), -1)
            node_idx[padded_patch] = torch.arange(
                padded_patch.size(0), device=patch_edge_index.device
            )
            patch_edge_index = node_idx[patch_edge_index]
            patch_graph["edge_index_%s" % i] = patch_edge_index
            patch_graph["edge_attr_%s" % i] = patch_edge_attr
        patch_graph.edge_index = patch_graph.edge_index_0
        patch_graph.edge_attr = patch_graph.edge_attr_0
        return patch_graph

    def rollout_info(self, sim_num, rollout_start_idx, window_length, sources):
        """
        Returns a dictionary containing data needed to perform rollout for this dataset

        sim_num: int, the ID of the simulation of interest (between 0 and n_simulations)
        rollout_start_idx: index of the timestep from which to start rollout
        window_length: int, output at time t will be predicted using observations in window [t-window_length, t - 1] (default: 1)
        sources: not used, but kept here to be compatible with rollout-generating scripts
                 see rollout_scripts/rollout_full_domains.py
        """
        graph = self.get_graph_for_simulation(sim_num)
        data = graph.output[rollout_start_idx:]
        node_coordinates = graph.pos.numpy()
        node_types = self._relabel_node_types(graph.node_types).numpy()
        feature_size = data.shape[-1]
        initial_window_data = data[:window_length, :, :feature_size]
        num_nodes = len(node_types)
        rollout_iterations = len(data) - window_length
        onehot_info = (self.apply_onehot, self.num_node_types)
        fixed = np.where(node_types != 0)[0]
        source_data = (fixed, data[window_length:, fixed, :feature_size])
        update_function = self.update_function

        return {
            "initial_window_data": initial_window_data,
            "graph": graph,
            "node_types": node_types,
            "node_coordinates": node_coordinates,
            "onehot_info": onehot_info,
            "rollout_iterations": rollout_iterations,
            "source_data": source_data,
            "shift_source": False,
            "update_function": update_function,
            "num_nodes": num_nodes,
            "feature_size": feature_size,
            "ground_truth": data,
        }

    def get_graph_for_simulation(self, traj_idx):
        """
        Creates a torch_geometric Data object for a particular simulation
        The Data contains 1) node coordinates, 2) edges, 3) the output data, and 4) node types

        Inputs:
        traj_idx: the ID of a simulation

        Returns:
        Data object representing the simulation
        """
        # load trajectory
        data_filename = self.direc + "sample_%s.npy" % (traj_idx)
        data_sample = np.load(data_filename, allow_pickle=True).item()
        # velocity
        data = data_sample["velocity"]
        # Node types
        node_types = torch.from_numpy(data_sample["node_type"][0, :, 0]).type(
            torch.LongTensor
        )
        non_fluid_indices = torch.where(~torch.isin(node_types, self.source_node_types))[0]
        # get node coordinates and build graph
        node_coordinates = torch.from_numpy(data_sample["mesh_pos"][0]).float()
        num_nodes = node_coordinates.shape[0]
        graph = Data(pos=node_coordinates, output=data, node_types=node_types)
        # edges
        for i, graph_type in enumerate(self.graph_types):
            subtype_graph = Data(pos=node_coordinates)
            if graph_type == "delaunay":
                # Note: DM data already contains a triangulation
                edge_index = torch.tensor(triangles_to_edges(data_sample["cells"][0]))
                subtype_graph.edge_index = edge_index
                transforms = [
                    Cartesian(norm=False, cat=True),
                    Distance(norm=False, cat=True),
                ]
                transforms = Compose(transforms)
                subtype_graph = transforms(subtype_graph)
            elif graph_type in ["radius", "knn"]:
                subtype_graph = add_edges_to_nodes(
                    subtype_graph, num_nodes, non_fluid_indices,
                    graph_type=graph_type, radius=self.radius, k=self.k
                )
            else:
                raise Exception("invalid graph type: %s" % graph_type)
            graph["edge_index_%s" % i] = subtype_graph.edge_index
            graph["edge_attr_%s" % i] = subtype_graph.edge_attr
        graph.edge_index = graph.edge_index_0
        graph.edge_attr = graph.edge_attr_0
        return graph
