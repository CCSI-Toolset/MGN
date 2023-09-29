from torch_geometric.data import Dataset
import pickle
import numpy as np
import pandas as pd
import torch
import os.path as osp
from collections import OrderedDict
from tqdm import tqdm
from time import sleep
import argparse
from multiprocessing import Pool, Manager
import resource
import os


class PNNL_Raw_Dynamic_Vars_and_Meshes(Dataset):
    """
    We preprocess the raw PNNL data with either `PNNL_With_Node_Type.py` or
    `PNNL_Raw_Dynamic_Vars_and_Meshes.py`; the latter saves out minimal
    data and is useful for dealing with very large graphs (like 3D data).

    PNNL_Raw_Dynamic_Vars_and_Meshes checks that the data meets certain
    conditions (e.g., the liquid inlet nodes are source nodes and should
    have certain volume fractions and velocities) and adds certain
    necessary variables (like node type) as needed. It only saves out
    the dynamic variables for each timestep; variables that are constant
    for a given mesh are saved out only once.
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
        proc=0,
        n_proc=1,
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
        :param proc: this code processes data in parallel with multiprocessing. use the proc argument to also use distributed processing (i.e., instantiate this class on 2 different nodes, each with a unique value for proc, starting with proc=0 on the first node and increasing by 1 for additional nodes)
        :param n_proc: see the proc argument. this is the total number of processes used
        """
        self.custom_processed_dir = processed_dir
        self.root = root
        if not sim_file_form:
            sim_file_form = "/".join(root.split("/")[:-1])
        self.sim_file_form = sim_file_form

        self.proc = proc
        self.n_proc = n_proc

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
        if not self.three_d:
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
        else:
            self.phys_vars = [
                "Velocity[i] (m/s)",
                "Velocity[j] (m/s)",
                "Velocity[k] (m/s)",
                "Volume Fraction of Liq",
                "Pressure (Pa)",
            ]
            self.yxz = ["Y (m)", "X (m)", "Z (m)"]
            self.xyz = ["X (m)", "Y (m)", "Z (m)"]
            self.dt = 50
            self.vertical_vel_var = "Velocity[k] (m/s)"
            self.vertical_mesh_var = "Z (m)"
        self.initialize_trackers()

        if osp.exists(osp.join(self.processed_dir, "value_dict_0.pkl")):
            self.set_attrs_built_during_processing()

        # inlet velocities for each sim
        self.inletVelocity = {}
        save_vels = 1
        if osp.exists(velocity_file):
            with open(velocity_file) as fid:
                vels = fid.read().splitlines()
            assert (
                len(vels) == len(self.sim_range) + 1
            ), "You are preprocessing {len(self.sim_range)} sims but only have {len(vels)-1} velocities available."
            self.inletVelocity = {
                self.sim_range[i]: v
                for i, v in enumerate(np.array(list(map(float, vels[1:]))))
            }
            save_vels = 0
        if not self.inletVelocity:
            self.inletVelocity = Manager().dict({})
        print(velocity_file, osp.exists(velocity_file), save_vels)

        super(PNNL_Raw_Dynamic_Vars_and_Meshes, self).__init__(
            root, transform, pre_transform
        )

        if self.proc == 0:
            self.inletVelocity = dict(self.inletVelocity)
            if save_vels and self.inletVelocity:
                with open(velocity_file, "w") as fid:
                    fid.write("vel")
                    for s in self.sim_range:
                        fid.write("\n")
                        fid.write(str(self.inletVelocity[s]))

            if force_check:
                self.check_processed_data(True)
            self.check_processed_data()

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
        manager = Manager()
        self.min_yvel_by_node_type = manager.dict({t: 1e10 for t in self.node_types})
        self.max_yvel_by_node_type = manager.dict({t: -1e10 for t in self.node_types})
        self.min_xvel_by_node_type = manager.dict({t: 1e10 for t in self.node_types})
        self.max_xvel_by_node_type = manager.dict({t: -1e10 for t in self.node_types})
        self.min_pres_by_node_type = manager.dict({t: 1e10 for t in self.node_types})
        self.max_pres_by_node_type = manager.dict({t: -1e10 for t in self.node_types})
        self.min_frac_by_node_type = manager.dict({t: 1e10 for t in self.node_types})
        self.max_frac_by_node_type = manager.dict({t: -1e10 for t in self.node_types})
        self.min_max_list = manager.list()
        self.min_max_list += [
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
            self.min_zvel_by_node_type = manager.dict(
                {t: 1e10 for t in self.node_types}
            )
            self.max_zvel_by_node_type = manager.dict(
                {t: -1e10 for t in self.node_types}
            )
            self.min_max_list += [
                self.min_zvel_by_node_type,
                self.max_zvel_by_node_type,
            ]

    def unpack_trackers(self):
        # recover these attributes after we convert them to non-manager objects
        if self.three_d:
            [
                self.min_yvel_by_node_type,
                self.max_yvel_by_node_type,
                self.min_xvel_by_node_type,
                self.max_xvel_by_node_type,
                self.min_pres_by_node_type,
                self.max_pres_by_node_type,
                self.min_frac_by_node_type,
                self.max_frac_by_node_type,
                self.min_zvel_by_node_type,
                self.max_zvel_by_node_type,
            ] = self.min_max_list
        else:
            [
                self.min_yvel_by_node_type,
                self.max_yvel_by_node_type,
                self.min_xvel_by_node_type,
                self.max_xvel_by_node_type,
                self.min_pres_by_node_type,
                self.max_pres_by_node_type,
                self.min_frac_by_node_type,
                self.max_frac_by_node_type,
            ] = self.min_max_list

    def set_attrs_built_during_processing(self):
        print("setting mesh, node count, node label attrs")
        d = torch.load(osp.join(self.processed_dir, "PNNL_full_mesh.pt"))
        self.mesh_dict = d["mesh_dict"]
        self.sim_to_mesh_type_dict = d["sim_to_mesh_type_dict"]
        path = osp.join(self.sim_file_form, self.raw_file_names[1])
        data = pd.read_csv(path)
        if self.node_type_file is not None:
            data = self.merge_boundary_onto(data, self.node_type_file)
        self.boundary_index_to_name_mapping = self.get_index_name_mapping(data)

        print("loading value dicts")
        self.min_max_list = list(self.min_max_list)
        for i in range(len(self.min_max_list)):
            self.min_max_list[i] = dict(self.min_max_list[i])
        for i, old_dict in enumerate(self.min_max_list):
            new_dict = pickle.load(
                open(
                    osp.join(self.processed_dir, "{}_{}.pkl".format("value_dict", i)),
                    "rb",
                )
            )
            for k, v in new_dict.items():
                old_dict[k] = v
        self.unpack_trackers()

    @property
    def raw_file_data(self):
        files = OrderedDict()
        self.n_files = 0
        for sim in self.sim_range:
            path = self.sim_file_form
            if sim == 1:
                path = self.sim_file_form
            sim_num = str(sim).zfill(2) if not self.three_d else sim
            for time in self.time_range:
                files[self.n_files] = {}
                files[self.n_files][
                    "path"
                ] = f"{path}/{self.sim_num_prefix}{sim_num}/XYZ_Internal_Table_table_{str(time*self.dt)}.csv"
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

        if (
            not osp.exists(osp.join(self.processed_dir, "PNNL_full_mesh.pt"))
            and self.proc == 0
        ):
            for sim, frame in enumerate(
                list(self.raw_file_data.values())[:: len(self.time_range)]
            ):
                assert frame["sim"] == self.sim_range[sim], (
                    frame["sim"],
                    self.sim_range[sim],
                )
                self.build_mesh_dict(frame, frame["sim"])
            torch.save(
                {
                    "mesh_dict": self.mesh_dict,
                    "sim_to_mesh_type_dict": self.sim_to_mesh_type_dict,
                },
                osp.join(self.processed_dir, "PNNL_full_mesh.pt"),
            )
        elif not osp.exists(osp.join(self.processed_dir, "PNNL_full_mesh.pt")):
            print("Waiting for full mesh to be made")
            self.wait_for_file(osp.join(self.processed_dir, "PNNL_full_mesh.pt"))
        d = torch.load(osp.join(self.processed_dir, "PNNL_full_mesh.pt"))
        self.mesh_dict = d["mesh_dict"]
        self.sim_to_mesh_type_dict = d["sim_to_mesh_type_dict"]

        unprocessed = list(self.raw_file_data.values())[i:]
        unprocessed = unprocessed[self.proc :: self.n_proc]
        print(f"process {self.proc} processing {len(unprocessed)} files")
        # for multiprocessing to work, setting resource limit per https://github.com/Project-MONAI/MONAI/issues/701
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
        n_processes = int(len(os.sched_getaffinity(0)) * 0.75)
        pool = Pool(processes=n_processes)
        list(
            tqdm(pool.imap(self.load_process_save, unprocessed), total=len(unprocessed))
        )

        self.min_max_list = list(self.min_max_list)
        for i in range(len(self.min_max_list)):
            self.min_max_list[i] = dict(self.min_max_list[i])
        self.unpack_trackers()
        self.value_dict_writer()
        self.wait_for_file("val_dicts")

    def wait_for_file(self, file):
        if file == "val_dicts":
            while (
                sum(
                    [
                        osp.exists(
                            osp.join(self.processed_dir, f"value_dict_7_proc{p}.pkl")
                        )
                        for p in range(self.n_proc)
                    ]
                )
                < self.n_proc
            ):
                sleep(5)
            return
        while not osp.exists(file):
            sleep(3)

    def load_process_save(self, frame):

        # hardcode t=0 values, using t=dt as a base mesh.
        path = (
            frame["path"]
            if frame["path"].split("_")[-1] != "0.csv"
            else frame["path"].replace("0.csv", f"{self.dt}.csv")
        )
        data = self.pre_transform(self, path, frame["sim"], frame["time"])

        # check node values in a frame, update stats, save out processed frame (only dynamic vars for efficient loading)
        self.check_vals_in_frame(data, frame["sim"])
        velo_i = data["Velocity[i] (m/s)"]
        velo_j = data["Velocity[j] (m/s)"]
        vol_frac = data["Volume Fraction of Liq"]
        velo_i_tensor = torch.tensor(velo_i.to_numpy())
        velo_j_tensor = torch.tensor(velo_j.to_numpy())
        vol_frac_tensor = torch.tensor(vol_frac.to_numpy())
        if not self.three_d:
            res2 = {
                "velo_i": velo_i_tensor,
                "velo_j": velo_j_tensor,
                "vol_frac": vol_frac_tensor,
            }
        else:
            velo_k = data["Velocity[k] (m/s)"]
            velo_k_tensor = torch.tensor(velo_k.to_numpy())
            res2 = {
                "velo_i": velo_i_tensor,
                "velo_j": velo_j_tensor,
                "velo_k": velo_k_tensor,
                "vol_frac": vol_frac_tensor,
            }
        torch.save(
            res2,
            osp.join(
                self.processed_dir, "PNNL_{}_{}.pt".format(frame["sim"], frame["time"])
            ),
        )

    def value_dict_stacker(self):
        print("stacking value dicts")
        for p in range(self.n_proc):
            for i, master_dict in enumerate(self.min_max_list):
                loc = osp.join(self.processed_dir, f"value_dict_{i}_proc{p}.pkl")
                worker_dict = pickle.load(open(loc, "rb"))
                use_mx = (
                    i % 2 == 1
                )  # master_dict is tracking a max if this is true, otherwise a min

                for node_type in self.node_types:
                    val = worker_dict[node_type]
                    update = val < master_dict[node_type]
                    if use_mx:
                        update = val > master_dict[node_type]
                    if update:
                        master_dict[node_type] = val

        for i, dic in enumerate(self.min_max_list):
            pickle.dump(
                dic, open("{}/value_dict_{}.pkl".format(self.processed_dir, i), "wb")
            )

    def value_dict_writer(self):
        for i, dic in enumerate(self.min_max_list):
            pickle.dump(
                dic,
                open(
                    "{}/value_dict_{}_proc{}.pkl".format(
                        self.processed_dir, i, self.proc
                    ),
                    "wb",
                ),
            )

    def check_processed_data(self, by_frame_first=False):

        # you want this option if the dictionaries like self.min_yvel_by_node_type aren't
        # updated in this instance yet
        if by_frame_first:
            self.initialize_trackers()
            for i, sim in enumerate(tqdm(self.sim_range)):
                mesh = self.mesh_dict[self.sim_to_mesh_type_dict[sim]]
                for time in tqdm(self.time_range):
                    data = torch.load(
                        osp.join(self.processed_dir, "PNNL_{}_{}.pt".format(sim, time))
                    )
                    data = pd.DataFrame(data)
                    data["node_type"] = mesh["node_type"].values
                    if not self.three_d:
                        data = data.rename(
                            columns={
                                "velo_i": "Velocity[i] (m/s)",
                                "velo_j": "Velocity[j] (m/s)",
                                "vol_frac": "Volume Fraction of Liq",
                            }
                        )
                    else:
                        data = data.rename(
                            columns={
                                "velo_i": "Velocity[i] (m/s)",
                                "velo_j": "Velocity[j] (m/s)",
                                "velo_k": "Velocity[k] (m/s)",
                                "vol_frac": "Volume Fraction of Liq",
                            }
                        )
                    self.check_vals_in_frame(data, sim, time)
            self.value_dict_writer()

        self.value_dict_stacker()
        # running this assumes the dictionaries like self.min_yvel_by_node_type have been updated
        # either through the initial processing of the raw data or the preceding loop
        self.check_vals_across_frames()

    def check_vals_across_frames(self):
        # check node values using information from across all frames

        # most node types can reach volume fraction of 1
        for node_type in self.node_types:
            if node_type not in ["inlet_to_outlet_wall", "outlet"]:
                assert round(self.max_frac_by_node_type[node_type], 2) == 1, (
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

    def check_vals_in_frame(self, data, sim, time=None):
        # check node values in a frame

        # at liquid_inlet, y velocity is inlet velocity (for 2D, else z vel), x velocity is small
        unique_liquid_inlet_vel = data.query(
            'node_type=="liquid_inlet"'
        ).drop_duplicates(subset=[self.vertical_vel_var])
        unique_liquid_inlet_frac = data.query(
            'node_type=="liquid_inlet"'
        ).drop_duplicates(subset=["Volume Fraction of Liq"])

        # at liquid_inlet, volume fraction and velocity are constant, pressure is not
        assert len(unique_liquid_inlet_vel) == 1, (sim, time, unique_liquid_inlet_vel)
        assert len(unique_liquid_inlet_frac) == 1, unique_liquid_inlet_frac

        # at liquid_inlet, volume fraction is always 1
        assert (
            unique_liquid_inlet_frac["Volume Fraction of Liq"].values[0] == 1
        ), unique_liquid_inlet_frac["Volume Fraction of Liq"].values[0]

        # at liquid_inlet, y velocity is equal to the negative of the sim's inlet velocity
        if sim in self.inletVelocity:
            assert (
                self.inletVelocity[sim]
                == unique_liquid_inlet_vel[self.vertical_vel_var].values[0]
            )
        else:
            self.inletVelocity[sim] = unique_liquid_inlet_vel[
                self.vertical_vel_var
            ].values[0]

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

            mn = data.query("node_type==@node_type")["Volume Fraction of Liq"].min()
            mx = data.query("node_type==@node_type")["Volume Fraction of Liq"].max()
            if self.min_frac_by_node_type[node_type] > mn:
                self.min_frac_by_node_type[node_type] = mn
            if self.max_frac_by_node_type[node_type] < mx:
                self.max_frac_by_node_type[node_type] = mx

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

    def build_mesh_dict(self, frame, sim):
        # find where mesh point locations are different across sims.
        # add variables specific to mesh (node type and contact angle)

        path = (
            frame["path"]
            if frame["path"].split("_")[-1] != "0.csv"
            else frame["path"].replace("0.csv", f"{self.dt}.csv")
        )
        data = pd.read_csv(path)
        data = data.round(self.precision(var="All"))

        # sort data by y then x
        data = data.sort_values(by=self.yxz)
        # had to round data to ensure that we have the same mesh locations (x,y) across sims,
        # make sure this didn't introduce duplicates
        assert len(data[data[self.yxz].duplicated(keep=False)]) == 0

        # define node types (aka 'Boundary Index'), unless they came pre-merged onto the data
        if self.node_type_file is not None:
            data = self.merge_boundary_onto(data, self.node_type_file)
        # convert these 'Boundary Index' values into something we can understand
        if not hasattr(self, "boundary_index_to_name_mapping"):
            self.boundary_index_to_name_mapping = self.get_index_name_mapping(data)
        data = assign_node_type_drop_index(data, self.boundary_index_to_name_mapping)

        # add contact angle
        add_contact_angle(data)

        # build dict of mesh positions
        if not hasattr(self, "mesh_dict"):
            self.mesh_dict = {
                0: data[["X (m)", "Y (m)", "Z (m)", "node_type", "contact_angle"]]
            }  # type of mesh: numpy array representing mesh
            self.sim_to_mesh_type_dict = {sim: 0}  # sim number: type of mesh
        else:
            mesh_type_exists = 0
            for mesh_type, mesh in self.mesh_dict.items():
                if len(data) == len(mesh):
                    if np.all(data[self.xyz].values == mesh[self.xyz].values):
                        self.sim_to_mesh_type_dict[sim] = mesh_type
                        mesh_type_exists = 1
                        break
                    else:
                        print(
                            "\nWarning: two sims have the same mesh size but different node locations. Maybe each has the same mesh but numerical error prevented merging."
                        )
            if not mesh_type_exists:
                mesh_type = max(self.mesh_dict.keys()) + 1
                print(f"\nCreating mesh type {mesh_type}")
                self.mesh_dict[mesh_type] = data[
                    ["X (m)", "Y (m)", "Z (m)", "node_type", "contact_angle"]
                ]
                self.sim_to_mesh_type_dict[sim] = mesh_type


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


def pre_transform(self, path, sim, time):
    # sort data by x, then y.
    # confirm mesh is constant across time given a sim
    # add node_type variable -- requires self.node_type_file to not be None unless 'Boundary Index' was premerged.
    # set pressure to 0 for gas_inlet nodes.
    # for detailed explanation of this, see https://myconfluence.llnl.gov/x/fbijF or the related 2/22 emails from Jie.

    if self.three_d:
        data = pd.read_csv(
            path,
            usecols=[
                "Velocity[i] (m/s)",
                "Velocity[j] (m/s)",
                "Velocity[k] (m/s)",
                "Volume Fraction of Liq",
                "X (m)",
                "Y (m)",
                "Z (m)",
            ],
        )
    else:
        data = pd.read_csv(
            path,
            usecols=[
                "Velocity[i] (m/s)",
                "Velocity[j] (m/s)",
                "Volume Fraction of Liq",
                "X (m)",
                "Y (m)",
                "Z (m)",
            ],
        )
    data = data.round(self.precision(var="All"))
    # sort data by y then x
    data = data.sort_values(by=self.yxz)

    # get relevant mesh dict
    mesh = self.mesh_dict[self.sim_to_mesh_type_dict[sim]]
    assert len(data) == len(mesh), (len(data), len(mesh))
    assert np.all(data[self.xyz].values == mesh[self.xyz].values)

    # hardcode t=0
    if time == 0:
        for var in [
            self.phys_vars[:-2]
        ]:  # for each velocity var -- i, j, and (if 3D) k
            data.loc[
                (mesh["node_type"].values != "liquid_inlet")
                & (mesh["node_type"].values != "outlet"),
                var,
            ] = 0
        data.loc[
            mesh["node_type"].values != "liquid_inlet", "Volume Fraction of Liq"
        ] = 0
        # data['Pressure (Pa)'] = 0 # for now, we are not using pressure

    data["node_type"] = mesh["node_type"].values

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
        "/global/cfs/cdirs/m4313/pnnl_globus/test_2D/",
        "/global/cfs/cdirs/m4313/pnnl_globus/test_3D/",
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
        processed_dir = "/pscratch/sd/b/bartolds/test_pnnl_data/processed/"
        sim_range = range(1, 2)
        time_range = range(0, 2)
        if "2D" in args["root"]:
            node_type_file = (
                "/global/cfs/cdirs/m4313/pnnl_globus/2D/test_export_boundary.csv"
            )
            processed_dir = "/pscratch/sd/b/bartolds/test_pnnl_data_2D/processed/"
    if not velocity_file:
        velocity_file = osp.join(args["root"], "vels.txt")

    PNNL_Raw_Dynamic_Vars_and_Meshes(
        root=args["root"],
        pre_transform=pre_transform,
        node_type_file=node_type_file,
        velocity_file=velocity_file,
        processed_dir=processed_dir,
        sim_range=sim_range,
        time_range=time_range,
    )
