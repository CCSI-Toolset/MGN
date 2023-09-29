from torch_geometric.data import Dataset
import pickle
import numpy as np
import pandas as pd
import torch
import os.path as osp
from collections import OrderedDict
from tqdm import tqdm
import argparse


class PNNL_With_Node_Type(Dataset):
    """
    We preprocess the raw PNNL data with either `PNNL_With_Node_Type.py` or
    `PNNL_Raw_Dynamic_Vars_and_Meshes.py`; the latter saves out minimal
    data and is useful for dealing with very large graphs (like 3D data).

    PNNL_With_Node_Type checks that the data meets certain
    conditions (e.g., the liquid inlet nodes are source nodes and should
    have certain volume fractions and velocities) and adds certain
    necessary variables (like node type) as needed. It saves all this
    data for each timestep.
    """

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        node_type_file=None,
        velocity_file=None,
        sim_file_form=None,
        force_check=False,  # will overwrite stats created during processing
        processed_dir=None,
        sim_range=None,
        time_range=None,
    ):
        """
        Create PNNL dataset with node types
        (see https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/dataset.py)
        :param root: self.processed_dir = root/processed
        :param transform:
        :param pre_transform:
        :param node_type_file: see 02_determine_node_types.ipynb
        :param velocity_file: has velocities for each simulation
        :param sim_file_form: to be completed as '[sim_file_form]/{sim_num_prefix}{sim}/XYZ_Internal_Table_table_{dt*step}0.csv'
        :param processed_dir: defaults to root/processed
        :param sim_range: iterator over sims, e.g. range(1, 51)
        """
        self.custom_processed_dir = processed_dir
        self.root = root
        if not sim_file_form:
            sim_file_form = "/".join(root.split("/")[:-1])
        self.sim_file_form = sim_file_form

        self.n_nodes = None  # number of nodes in the graph
        self.mesh_xy = None  # x,y locations of mesh
        self.sim_range = sim_range if sim_range else range(1, 51)
        self.time_range = time_range if time_range else range(0, 501)
        self.three_d = "3D" in root
        self.sim_num_prefix = "0" if not self.three_d else "Design_"

        # node type from PNNL
        self.node_type_file = node_type_file

        self.node_types = [
            "fluid",
            "side_wall",
            "gas_inlet",
            "outlet",
            "liquid_inlet",
            "packing",
            "inlet_to_outlet_wall",
        ]
        self.phys_vars = [
            "Velocity[i] (m/s)",
            "Velocity[j] (m/s)",
            "Volume Fraction of Liq",
            "Pressure (Pa)",
        ]
        self.yxz = ["Y (m)", "X (m)"]
        self.xyz = ["X (m)", "Y (m)"]
        self.dt = 10
        self.vertical_vel_var = "Velocity[j] (m/s)"
        self.vertical_mesh_var = "Y (m)"
        if self.three_d:
            self.phys_vars = [
                "Velocity[i] (m/s)",
                "Velocity[j] (m/s)",
                "Velocity[k] (m/s)",
                "Volume Fraction of Liq",
                "Pressure (Pa)",
            ]
            self.yxz = ["Y (m)", "X (m)", "Z (m)"]
            self.xyz = ["X (m)", "Y (m)", "Z (m)"]
            self.vertical_vel_var = "Velocity[k] (m/s)"
            self.vertical_mesh_var = "Z (m)"
            self.dt = 50
        self.initialize_trackers()

        if osp.exists(osp.join(self.processed_dir, "value_dict_0.pkl")):
            self.set_attrs_built_during_processing()

        # inlet velocities for each sim
        with open(velocity_file) as fid:
            vels = fid.read().splitlines()
        self.inletVelocity = np.array(list(map(float, vels[1:])))

        super(PNNL_With_Node_Type, self).__init__(root, transform, pre_transform)

        if force_check:
            self.check_processed_data(True)
        self.check_vals_across_frames()

        # make sure that the number of updates to the stats dicts is the number of frames.
        # this would fail if, e.g., processing was stopped after updating a stat dict but
        # before writing out the PyTorch frame, which would cause an extra write to the stat
        # dict when processing resumed.
        assert self.variable_means["n"] == self.len()

    @property
    def processed_dir(self):
        return (
            self.custom_processed_dir
            if self.custom_processed_dir
            else osp.join(self.root, "processed")
        )

    def precision(self, var):
        d = {
            "X (m)": 13,
            "Y (m)": 13,
            "Z (m)": 13,
            "Velocity[i] (m/s)": 12,
            "Velocity[j] (m/s)": 12,
            "Velocity[k] (m/s)": 12,
            "Volume Fraction of Liq": 12,
            "Pressure (Pa)": 12,
        }
        if var == "All":
            if not self.three_d:
                del d["Velocity[k] (m/s)"]
            return d
        return d[var]

    def initialize_trackers(self):
        # dictionaries that for tracking stats,
        # get built during processing...
        # can also be built by running self.check_processed_data(True) (this will overwrite prior vals!)
        self.variable_means = {p: 0 for p in self.phys_vars}
        self.variable_means["n"] = 0
        self.variable_stds = {p: 0 for p in self.phys_vars}
        self.min_yvel_by_node_type = {t: 1e10 for t in self.node_types}
        self.max_yvel_by_node_type = {t: -1e10 for t in self.node_types}
        self.min_xvel_by_node_type = {t: 1e10 for t in self.node_types}
        self.max_xvel_by_node_type = {t: -1e10 for t in self.node_types}
        self.min_zvel_by_node_type = {t: 1e10 for t in self.node_types}
        self.max_zvel_by_node_type = {t: -1e10 for t in self.node_types}
        self.min_pres_by_node_type = {t: 1e10 for t in self.node_types}
        self.max_pres_by_node_type = {t: -1e10 for t in self.node_types}
        self.min_frac_by_node_type = {t: 1e10 for t in self.node_types}
        self.max_frac_by_node_type = {t: -1e10 for t in self.node_types}
        self.min_max_list = [
            self.min_yvel_by_node_type,
            self.max_yvel_by_node_type,
            self.min_xvel_by_node_type,
            self.max_xvel_by_node_type,
            self.min_pres_by_node_type,
            self.max_pres_by_node_type,
            self.min_frac_by_node_type,
            self.max_frac_by_node_type,
        ]
        if self.three_d:
            self.min_max_list += [
                self.min_zvel_by_node_type,
                self.max_zvel_by_node_type,
            ]

    def set_attrs_built_during_processing(self):
        print("setting mesh, node count, node label attrs")
        data = torch.load(osp.join(self.processed_dir, "PNNL_{}_{}.pt".format(1, 0)))
        self.n_nodes = data["X (m)"].size
        self.mesh_xy = data[self.xyz].values
        path = osp.join(self.sim_file_form, self.raw_file_names[1])
        data = pd.read_csv(path)
        if self.node_type_file is not None:
            data = self.merge_boundary_onto(data, self.node_type_file)
        self.boundary_index_to_name_mapping = self.get_index_name_mapping(data)

        print("loading value dicts")
        for i, old_dict in enumerate(self.min_max_list):
            new_dict = pickle.load(
                open(
                    osp.join(self.processed_dir, "{}_{}.pkl".format("value_dict", i)),
                    "rb",
                )
            )
            for k, v in new_dict.items():
                old_dict[k] = v

        self.variable_means = pickle.load(
            open(osp.join(self.processed_dir, "var_mean_dict.pkl"), "rb")
        )
        self.variable_stds = pickle.load(
            open(osp.join(self.processed_dir, "var_std_dict.pkl"), "rb")
        )

    @property
    def raw_file_data(self):
        files = OrderedDict()
        self.n_files = 0
        for sim in self.sim_range:
            sim_num = str(sim).zfill(2) if not self.three_d else sim
            for time in self.time_range:
                files[self.n_files] = {}
                files[self.n_files][
                    "path"
                ] = f"{self.sim_file_form}/{self.sim_num_prefix}{sim_num}/XYZ_Internal_Table_table_{str(time*self.dt)}.csv"
                files[self.n_files]["sim"] = sim
                files[self.n_files]["time"] = time
                self.n_files += 1
        return files

    @property
    def raw_file_names(self):
        paths = []
        for sim in self.sim_range:
            sim_num = str(sim).zfill(2) if not self.three_d else sim
            for time in self.time_range:
                paths.append(
                    f"{self.sim_num_prefix}{sim_num}/XYZ_Internal_Table_table_{str(time*self.dt)}.csv"
                )
        return paths

    def download(self):
        # files already downloaded
        return

    @property
    def raw_dir(self) -> str:
        return osp.join(self.sim_file_form, "")

    @property
    def processed_file_names(self):
        return [
            "PNNL_{}_{}.pt".format(s, t)
            for s in self.sim_range
            for t in self.time_range
        ]

    def process(self):
        i = 0
        while osp.exists(osp.join(self.processed_dir, self.processed_file_names[i])):
            i += 1
        print(f"{i} iteration(s) already processed!")
        print(f"processing remaining {len(self.raw_file_data.values()) - i} iterations")
        assert self.variable_means["n"] == i, (self.variable_means["n"], i)
        for frame in tqdm(list(self.raw_file_data.values())[i:]):
            # sort data by x, then y.
            # confirm all frames have the same # of mesh points self.n_nodes.
            # confirm all mesh point locations are the same, no matter the sim or time.
            # add node_type variable -- requires self.node_type_file to not be None unless 'Boundary Index' was premerged.
            # set pressure to 0 for gas_inlet nodes.
            # hardcode t=0 values, using t=10 as a base mesh.
            # for detailed explanation of this, see https://myconfluence.llnl.gov/x/fbijF or the related 2/22 emails from Jie.
            path = (
                frame["path"]
                if frame["path"].split("_")[-1] != "0.csv"
                else frame["path"].replace("0.csv", f"{self.dt}.csv")
            )
            data = pd.read_csv(path)
            data = data.round(self.precision(var="All"))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(self, data, frame["sim"], frame["time"])

            # check node values in a frame, update stats, save out processed frame
            self.check_vals_in_frame(data, frame["sim"])
            self.value_dict_writer()
            torch.save(
                data,
                osp.join(
                    self.processed_dir,
                    "PNNL_{}_{}.pt".format(frame["sim"], frame["time"]),
                ),
            )
            i += 1

        self.check_processed_data()

    def value_dict_writer(self):
        for i, dic in enumerate(self.min_max_list):
            pickle.dump(
                dic, open("{}/value_dict_{}.pkl".format(self.processed_dir, i), "wb")
            )
        pickle.dump(
            self.variable_means,
            open("{}/var_mean_dict.pkl".format(self.processed_dir), "wb"),
        )
        pickle.dump(
            self.variable_stds,
            open("{}/var_std_dict.pkl".format(self.processed_dir), "wb"),
        )

    def check_processed_data(self, by_frame_first=False):

        # you want this option if the dictionaries like self.min_yvel_by_node_type aren't
        # updated in this instance yet
        if by_frame_first:
            self.initialize_trackers()
            for sim in tqdm(self.sim_range):
                for time in tqdm(self.time_range):
                    data = torch.load(
                        osp.join(self.processed_dir, "PNNL_{}_{}.pt".format(sim, time))
                    )
                    self.check_vals_in_frame(data, sim)

        # mean and var dicts just stored sums---convert them to mean and var now
        self.compute_mean_and_var()
        print(self.variable_means)
        print(self.variable_stds)
        self.value_dict_writer()
        # running this assumes the dictionaries like self.min_yvel_by_node_type have been updated
        # either through the initial processing of the raw data or the preceding loop
        self.check_vals_across_frames()

    def compute_mean_and_var(self):
        for var in self.phys_vars:
            self.variable_means[var] /= self.len() * self.n_nodes
            self.variable_stds[var] /= self.len() * self.n_nodes
            self.variable_stds[var] -= self.variable_means[var] ** 2
            self.variable_stds[var] = self.variable_stds[var] ** 0.5

    def check_vals_across_frames(self):
        # check node values using information from across all frames

        # most node types can reach volume fraction of 1
        for node_type in self.node_types:
            if node_type not in ["inlet_to_outlet_wall", "outlet"]:
                assert round(self.max_frac_by_node_type[node_type], 3) == 1, (
                    node_type,
                    "frac",
                    self.max_frac_by_node_type[node_type],
                )

        # these nodes' velocities can be large
        for node_type in ["fluid", "gas_inlet"]:
            assert self.min_yvel_by_node_type[node_type] < -0.5, (
                node_type,
                "min yvel",
                self.min_yvel_by_node_type[node_type],
            )
            assert self.max_yvel_by_node_type[node_type] > 0.5, (
                node_type,
                "max yvel",
                self.max_yvel_by_node_type[node_type],
            )
            assert self.min_xvel_by_node_type[node_type] < -0.5, (
                node_type,
                "min xvel",
                self.min_xvel_by_node_type[node_type],
            )
            assert self.max_xvel_by_node_type[node_type] > 0.5, (
                node_type,
                "max xvel",
                self.max_xvel_by_node_type[node_type],
            )
            if self.three_d:
                assert self.min_zvel_by_node_type[node_type] < -0.5, (
                    node_type,
                    "min zvel",
                    self.min_zvel_by_node_type[node_type],
                )
                assert self.max_zvel_by_node_type[node_type] > 0.5, (
                    node_type,
                    "max zvel",
                    self.max_zvel_by_node_type[node_type],
                )

        # gas_inlet was set to 0 pressure
        assert (
            self.min_pres_by_node_type["gas_inlet"]
            == self.max_pres_by_node_type["gas_inlet"]
            == 0
        ), (
            node_type,
            self.max_pres_by_node_type["gas_inlet"],
            self.min_pres_by_node_type["gas_inlet"],
        )

        # these nodes' velocities are always zero in x direction (after rounding to 12 places)...
        for node_type in ["outlet", "liquid_inlet"]:
            assert self.min_xvel_by_node_type[node_type] == 0, (
                node_type,
                "min xvel",
                self.min_xvel_by_node_type[node_type],
            )
            assert self.max_xvel_by_node_type[node_type] == 0, (
                node_type,
                "max xvel",
                self.max_xvel_by_node_type[node_type],
            )

            if self.three_d:
                assert self.min_yvel_by_node_type[node_type] == 0, (
                    node_type,
                    "min yvel",
                    self.min_yvel_by_node_type[node_type],
                )
                assert self.max_yvel_by_node_type[node_type] == 0, (
                    node_type,
                    "max yvel",
                    self.max_yvel_by_node_type[node_type],
                )

        # but in the vertical direction, they have special properties, enabling scripting of these nodes' vels
        if not self.three_d:
            assert self.max_yvel_by_node_type["liquid_inlet"] == -0.002, (
                "liquid_inlet",
                "yvel max",
            )
            assert self.min_yvel_by_node_type["liquid_inlet"] == -0.0218, (
                "liquid_inlet",
                "yvel min",
            )
            assert (
                self.max_yvel_by_node_type["outlet"]
                == self.min_yvel_by_node_type["outlet"]
                == 0.01963862
            ), self.min_yvel_by_node_type["outlet"]
        else:
            assert self.max_zvel_by_node_type["liquid_inlet"] == np.round(
                -0.0031641578451782, self.precision(self.vertical_vel_var)
            ), ("liquid_inlet", "zvel max")
            assert self.min_zvel_by_node_type["liquid_inlet"] == np.round(
                -0.158207892258912, self.precision(self.vertical_vel_var)
            ), ("liquid_inlet", "zvel min")
            assert (
                self.max_zvel_by_node_type["outlet"]
                == self.min_zvel_by_node_type["outlet"]
                == np.round(0.0168494133427118, self.precision(self.vertical_vel_var))
            ), self.min_zvel_by_node_type["outlet"]

        # for liquid_inlet, we can also script its vol frac:
        assert (
            self.min_frac_by_node_type["liquid_inlet"]
            == self.max_frac_by_node_type["liquid_inlet"]
            == 1
        ), (
            self.max_frac_by_node_type["liquid_inlet"],
            self.min_frac_by_node_type["liquid_inlet"],
        )

        # these nodes' velocities are always zero
        for node_type in ["side_wall", "packing", "inlet_to_outlet_wall"]:
            assert self.min_yvel_by_node_type[node_type] == 0, (
                node_type,
                "min yvel",
                self.min_yvel_by_node_type[node_type],
            )
            assert self.max_yvel_by_node_type[node_type] == 0, (
                node_type,
                "max yvel",
                self.max_yvel_by_node_type[node_type],
            )
            assert self.min_xvel_by_node_type[node_type] == 0, (
                node_type,
                "min xvel",
                self.min_xvel_by_node_type[node_type],
            )
            assert self.max_xvel_by_node_type[node_type] == 0, (
                node_type,
                "max xvel",
                self.max_xvel_by_node_type[node_type],
            )
            if self.three_d:
                assert self.min_zvel_by_node_type[node_type] == 0, (
                    node_type,
                    "min zvel",
                    self.min_zvel_by_node_type[node_type],
                )
                assert self.max_zvel_by_node_type[node_type] == 0, (
                    node_type,
                    "max zvel",
                    self.max_zvel_by_node_type[node_type],
                )

        print("\nChecks across frames passed!\n")

    def check_vals_in_frame(self, data, sim):
        # check node values in a frame

        # at liquid_inlet, y velocity is inlet velocity (for 2D, else z vel), x velocity is small
        unique_liquid_inlet_vel = data.query(
            'node_type=="liquid_inlet"'
        ).drop_duplicates(subset=[self.vertical_vel_var])
        unique_liquid_inlet_frac = data.query(
            'node_type=="liquid_inlet"'
        ).drop_duplicates(subset=["Volume Fraction of Liq"])
        unique_liquid_inlet_pres = data.query(
            'node_type=="liquid_inlet"'
        ).drop_duplicates(subset=["Pressure (Pa)"])

        # at liquid_inlet, volume fraction and velocity are constant, pressure is not
        assert len(unique_liquid_inlet_vel) == 1, unique_liquid_inlet_vel
        assert len(unique_liquid_inlet_frac) == 1, unique_liquid_inlet_frac
        assert len(unique_liquid_inlet_pres) > 1 or data["time"].values[0] == 0

        # at liquid_inlet, volume fraction is always 1
        assert unique_liquid_inlet_frac["Volume Fraction of Liq"].values[0] == 1

        # at liquid_inlet, y velocity is equal to the negative of the sim's inlet velocity
        assert abs(unique_liquid_inlet_vel[self.vertical_vel_var].values[0]) == abs(
            np.round(self.inletVelocity[sim - 1], self.precision(self.vertical_vel_var))
        ), (
            unique_liquid_inlet_vel[self.vertical_vel_var].values,
            self.inletVelocity,
            sim,
        )

        # at outflow top, y velocity is always 0.019639 (different for 3D)
        unique_outlet_vel = data.query('node_type=="outlet"').drop_duplicates(
            subset=[self.vertical_vel_var]
        )
        assert len(unique_outlet_vel) == 1, unique_outlet_vel
        if self.three_d:
            assert unique_outlet_vel["Velocity[k] (m/s)"].values[0] == np.round(
                0.0168494133427118, self.precision(self.vertical_vel_var)
            ), unique_outlet_vel["Velocity[k] (m/s)"].values[0]
        else:
            assert (
                unique_outlet_vel["Velocity[j] (m/s)"].values[0] == 0.01963862
            ), unique_outlet_vel["Velocity[j] (m/s)"].values[0]

        # track maximum and minimum for each channel
        for node_type in self.node_types:
            if self.three_d:
                mn = data.query("node_type==@node_type")["Velocity[k] (m/s)"].min()
                mx = data.query("node_type==@node_type")["Velocity[k] (m/s)"].max()
                if self.min_zvel_by_node_type[node_type] > mn:
                    self.min_zvel_by_node_type[node_type] = mn
                if self.max_zvel_by_node_type[node_type] < mx:
                    self.max_zvel_by_node_type[node_type] = mx

            mn = data.query("node_type==@node_type")["Velocity[j] (m/s)"].min()
            mx = data.query("node_type==@node_type")["Velocity[j] (m/s)"].max()
            if self.min_yvel_by_node_type[node_type] > mn:
                self.min_yvel_by_node_type[node_type] = mn
            if self.max_yvel_by_node_type[node_type] < mx:
                self.max_yvel_by_node_type[node_type] = mx

            mn = data.query("node_type==@node_type")["Velocity[i] (m/s)"].min()
            mx = data.query("node_type==@node_type")["Velocity[i] (m/s)"].max()
            if self.min_xvel_by_node_type[node_type] > mn:
                self.min_xvel_by_node_type[node_type] = mn
            if self.max_xvel_by_node_type[node_type] < mx:
                self.max_xvel_by_node_type[node_type] = mx

            mn = data.query("node_type==@node_type")["Pressure (Pa)"].min()
            mx = data.query("node_type==@node_type")["Pressure (Pa)"].max()
            if self.min_pres_by_node_type[node_type] > mn:
                self.min_pres_by_node_type[node_type] = mn
            if self.max_pres_by_node_type[node_type] < mx:
                self.max_pres_by_node_type[node_type] = mx

            mn = data.query("node_type==@node_type")["Volume Fraction of Liq"].min()
            mx = data.query("node_type==@node_type")["Volume Fraction of Liq"].max()
            if self.min_frac_by_node_type[node_type] > mn:
                self.min_frac_by_node_type[node_type] = mn
            if self.max_frac_by_node_type[node_type] < mx:
                self.max_frac_by_node_type[node_type] = mx

        # track mean and std
        for var in self.phys_vars:
            self.variable_means[var] += data[var].values.sum()
            self.variable_stds[var] += (data[var].values ** 2).sum()
        self.variable_means["n"] += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        sim = idx // len(self.time_range) + 1
        time = idx % len(self.time_range)
        data = torch.load(
            osp.join(self.processed_dir, "PNNL_{}_{}.pt".format(sim, time))
        )
        return data

    def get_index_name_mapping(self, merged):

        d = {}

        unique_indices = {x: float(x) for x in merged["Boundary Index"].unique()}
        found = 0
        for index, float_index in unique_indices.items():
            if float_index > 1e10:
                found += 1
                d[index] = "fluid"
        assert (
            found == 1
        )  # only the fluid node type should have a large 'Boundary Index'

        max_y = max(merged[self.vertical_mesh_var])
        mins = merged.groupby("Boundary Index").min()[self.vertical_mesh_var]
        found = 0
        for b_index, min_ in mins.items():
            if min_ == max_y:
                found += 1
                d[b_index] = "outlet"
        assert (
            found == 1
        )  # only the outlet node type should have its min equal to the max

        min_y = min(merged[self.vertical_mesh_var])
        found = 0
        for b_index, min_ in mins.items():
            if min_ == min_y:
                found += 1
                d[b_index] = "gas_inlet"
        assert found == 1  # only the gas_inlet node type should reach the min

        max_x = max(merged["X (m)"])
        maxes = merged.groupby("Boundary Index").max()["X (m)"]
        found = 0
        for b_index, max_ in maxes.items():
            if max_ == max_x:
                found += 1
                d[b_index] = "side_wall"
        assert found == 1  # only the side wall node type should have the max X location

        vels = merged.groupby("Boundary Index")[self.vertical_vel_var].unique()
        found = 0
        for b_index, vel_list in vels.items():
            if len(vel_list) == 1:
                if vel_list[0] < 0:
                    found += 1
                    d[b_index] = "liquid_inlet"
        assert (
            found == 1
        )  # only the liquid_inlet node type has the same (negative) velocity at all nodes

        found = 0
        smallest = 1e10
        for b_index, min_ in mins.items():
            if min_ < smallest:
                if b_index not in d:
                    smallest = min_
        for b_index, min_ in mins.items():
            if min_ == smallest:
                found += 1
                d[b_index] = "packing"
        assert (
            found == 1
        )  # the packing node type appears at lower locations than the other remaining node type

        found = 0
        for b_index, vel_list in vels.items():
            if b_index not in d:
                found += 1
                d[b_index] = "inlet_to_outlet_wall"
        assert found == 1  # only this node type remains

        assert len(d) == len(merged["Boundary Index"].unique())

        return d

    def merge_boundary_onto(self, single_frame, node_type_file):
        boundary = pd.read_csv(
            node_type_file, usecols=["X (m)", "Y (m)", "Boundary Index"]
        )
        boundary["round_X"] = np.round(boundary["X (m)"], self.precision("X (m)"))
        boundary["round_Y"] = np.round(boundary["Y (m)"], self.precision("Y (m)"))
        boundary = boundary.drop(["X (m)", "Y (m)"], axis=1)

        single_frame["round_X"] = np.round(
            single_frame["X (m)"], self.precision("X (m)")
        )
        single_frame["round_Y"] = np.round(
            single_frame["Y (m)"], self.precision("Y (m)")
        )

        merged = single_frame.merge(boundary, how="outer", on=["round_X", "round_Y"])
        assert len(merged) == len(
            single_frame
        )  # every obs in each dataset had exactly one match
        assert (
            pd.isna(merged["Boundary Index"]).sum() == 0
        )  # every obs in base dataset has a boundary
        merged = merged.drop(["round_X", "round_Y"], axis=1)
        return merged


def assign_node_type_drop_index(frame, mapping):
    frame["node_type"] = [mapping[b_index] for b_index in frame["Boundary Index"]]
    frame = frame.drop(["Boundary Index"], axis=1)
    return frame


def add_contact_angle(frame):
    # these are being added as strings because most node types don't have a meaningful contact angle
    contact_angle_dict = {
        "inlet_to_outlet_wall": "180",
        "side_wall": "33.5",
        "packing": "33.5",
        "fluid": "N/A",
        "outlet": "N/A",
        "gas_inlet": "N/A",
        "liquid_inlet": "N/A",
    }
    frame["contact_angle"] = [
        contact_angle_dict[node_type] for node_type in frame["node_type"]
    ]


def pre_transform(self, data, sim, time):
    # sort data by y then x
    data = data.sort_values(by=self.yxz)
    data["sim"] = [sim] * data["X (m)"].size
    data["time"] = [time] * data["X (m)"].size
    data["node"] = range(data["X (m)"].size)
    if self.n_nodes is None:
        self.n_nodes = data["X (m)"].size
    else:
        # confirm that all frames have the same number of nodes
        # note that these nodes are not always in the same place
        assert data["X (m)"].size == self.n_nodes
    if self.mesh_xy is None:
        # had to round data to ensure that we have the same mesh locations (x,y) across sims,
        # make sure this didn't introduce duplicates
        assert len(data[data[self.yxz].duplicated(keep=False)]) == 0
        self.mesh_xy = data[self.xyz].values
    else:
        # confirm that all x,y locations are the same across timesteps/sims
        assert np.all(data[self.xyz].values == self.mesh_xy)

    ###########################################################################
    # define node types (aka 'Boundary Index'), unless they came pre-merged onto the data
    if self.node_type_file is not None:
        data = self.merge_boundary_onto(data, self.node_type_file)
    # convert these 'Boundary Index' values into something we can understand
    if not hasattr(self, "boundary_index_to_name_mapping"):
        self.boundary_index_to_name_mapping = self.get_index_name_mapping(data)
    data = assign_node_type_drop_index(data, self.boundary_index_to_name_mapping)
    # end define node types
    ###########################################################################

    # hardcode t=0
    if time == 0:
        for var in [
            self.phys_vars[:-2]
        ]:  # for each velocity var -- i, j, and (if 3D) k
            data.loc[
                (data["node_type"] != "liquid_inlet") & (data["node_type"] != "outlet"),
                var,
            ] = 0
        data.loc[data["node_type"] != "liquid_inlet", "Volume Fraction of Liq"] = 0
        data["Pressure (Pa)"] = 0

    # hardcode pressure for gas inlet, but first make sure it's small relative to usual pressure vals
    p1 = (
        abs(self.min_pres_by_node_type["fluid"])
        if self.min_pres_by_node_type["fluid"] < 0
        else 100
    )
    p2 = (
        self.max_pres_by_node_type["fluid"]
        if self.max_pres_by_node_type["fluid"] > 0
        else 100
    )
    big_p = max(p1, p2)
    assert (
        max(abs(data.query('node_type=="gas_inlet"')["Pressure (Pa)"].values))
        < big_p / 100
    ), max(abs(data.query('node_type=="gas_inlet"')["Pressure (Pa)"].values))
    data.loc[data["node_type"] == "gas_inlet", "Pressure (Pa)"] = 0

    # add contact angle
    add_contact_angle(data)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a CCSI dataset")
    parser.add_argument(
        "-r",
        "--root",
        help="Destination for raw and to-be processed data",
        required=True,
    )
    args = vars(parser.parse_args())
    print(f"Processing data from {args['root']}")
    raw_data_folders = [
        "/usr/workspace/ccsi/PNNL_2020/t_equal_zero_and_pnnl_node_types/",
        "/data/ccsi/PNNL_2021_07_20_w_boundaries/",
        "/usr/workspace/ccsi/PNNL_3D_2022_09_15/",
        "/p/vast1/ccsi/PNNL_3D_2023_01_15/",
        "/global/cfs/cdirs/m4313/pnnl_globus/3D/",
        "/global/cfs/cdirs/m4313/pnnl_globus/test_3D/",
        "/global/cfs/cdirs/m4313/pnnl_globus/test_2D/",
    ]
    assert (
        args["root"] in raw_data_folders
    ), f"unknown raw data location {args['root']} -- extend list of raw_data_folders."

    node_type_file = None  # node type is already on some datasets
    velocity_file = (
        None  # if we do not have this file, we will create it using the data
    )
    processed_dir = None  # if not specified, store processed data in {root}/processed
    sim_range = None
    time_range = None
    if (
        args["root"]
        == "/usr/workspace/ccsi/PNNL_2020/t_equal_zero_and_pnnl_node_types/"
    ):
        node_type_file = "/usr/workspace/ccsi/test_export_boundary.csv"
        velocity_file = "/usr/workspace/ccsi/PNNL_2020/inlet_velocities.txt"
    if args["root"] == "/data/ccsi/PNNL_2021_07_20_w_boundaries/":
        velocity_file = "/data/ccsi/pnnl_liquid_inlet/liquid_inlet_velocity.txt"
    if args["root"] == "/usr/workspace/ccsi/PNNL_3D_2022_09_15/":
        processed_dir = "/p/vast1/ccsi/PNNL_3D_2022_09_15/processed"
    if args["root"] == "/global/cfs/cdirs/m4313/pnnl_globus/3D/":
        processed_dir = "/pscratch/sd/b/bartolds/pnnl_data/processed/"
    if "test" in args["root"]:
        processed_dir = (
            "/pscratch/sd/b/bartolds/test_pnnl_data_with_node_type/processed/"
        )
        sim_range = range(1, 2)
        time_range = range(0, 2)
        if "2D" in args["root"]:
            node_type_file = (
                "/global/cfs/cdirs/m4313/pnnl_globus/2D/test_export_boundary.csv"
            )
            processed_dir = (
                "/pscratch/sd/b/bartolds/test_pnnl_data_with_node_type_2D/processed/"
            )
            velocity_file = "/global/cfs/cdirs/m4313/pnnl_globus/2D/vels.txt"
    if not velocity_file:
        velocity_file = osp.join(args["root"], "vels.txt")

    PNNL_With_Node_Type(
        root=args["root"],
        pre_transform=pre_transform,
        node_type_file=node_type_file,
        velocity_file=velocity_file,
        processed_dir=processed_dir,
        sim_range=sim_range,
        time_range=time_range,
    )
