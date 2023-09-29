from importlib import import_module
import os
import pickle
from sys import path

import argparse
import numpy as np
from pathlib import Path

import torch
from torch import tensor
from torch_geometric.data import Data

#########
# local application imports
# get path to root of the project
mgn_code_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
path.append(mgn_code_dir)

from GNN.ModelConfig.ConfigMGN import ModelConfig
from GNN.MeshGraphNets import MeshGraphNets
from GNN.utils.train_utils import load_model
from GNN.utils.dist import allreduce_hook_sum
from GNN.utils.patch_sync_utils import (
    generate_non_ghost_nodes_dict,
    generate_ghost_nodes_info_dict,
)

import math
import shutil

from GNN.utils.cluster import CustomLSFEnvironment
from torch.nn.parallel import DistributedDataParallel
from torch import distributed
import datetime

"""
example usage:
    python rollout_scripts/rollout.py config_files/delaunay_fixed/pnnl_3D.ini
"""

#########
# load config file
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Config file to train model. See ../configfiles for examples",
    )
    parser.add_argument(
        "-test",
        "--test_set",
        action="store_true",
        help="Rollout and/or visualize using test instead of train data",
    )
    parser.add_argument(
        "--save_all_timesteps",
        action="store_true",
        help="Rollout and/or visualize using test instead of train data",
    )
    return parser


def main(args, env):

    m = ModelConfig(args.config_file)

    assert (
        m.get_world_size() == env.world_size
    ), f"World size {m.get_world_size()} in config does not match world size {env.world_size} in this current run"

    print("ENV:", env.__dict__)
    device = torch.device(f"cuda:{env.local_rank}")
    torch.distributed.init_process_group(
        "nccl",
        init_method="env://",
        timeout=datetime.timedelta(seconds=3600),
        world_size=env.world_size,
        rank=env.rank,
    )
    print("Rank {0}/{1} running on device {2}".format(env.rank, env.world_size, device))

    #########
    # prepare train or test set
    data_params = (
        m.get_train_data_params() if not args.test_set else m.get_test_data_params()
    )

    ds_class_name = m.get_class_name()
    DS = getattr(import_module("GNN.DatasetClasses." + ds_class_name), ds_class_name)
    dataset = DS(model_config=m, train_set=not args.test_set, **data_params)
    dataset.noise = None  # don't add noise during rollouts

    #########
    # build model
    mgn_dim = m.get_mgn_dim()
    model = MeshGraphNets(
        dataset.num_node_features,
        dataset.num_edge_features,
        dataset.output_dim,
        out_dim_node=mgn_dim,
        out_dim_edge=mgn_dim,
        hidden_dim_node=mgn_dim,
        hidden_dim_edge=mgn_dim,
        hidden_dim_processor_node=mgn_dim,
        hidden_dim_processor_edge=mgn_dim,
        hidden_dim_decoder=mgn_dim,
        node_normalizer_mask=dataset.node_normalizer_mask,
        edge_normalizer_mask=dataset.edge_normalizer_mask,
        mp_iterations=m.get_mp_iterations(),
        mlp_norm_type=m.get_mlp_norm_type(),
        output_type=m.get_output_type(),
        connection_type=m.get_connection_type(),
        connection_alpha=m.get_connection_alpha(),
        connection_aggregation=m.get_connection_aggregation(),
        integrator=m.get_integrator(),
    )

    #########
    # load model
    checkpoint_dir = m.get_checkpoint_dir()

    model = load_model(checkpoint_dir, model_filename="best_val_model.pt", model=model)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[env.local_rank])
    base_model = model.module

    if m.get_ddp_hook() == "sum":
        model.register_comm_hook(None, allreduce_hook_sum)
    elif m.get_ddp_hook() == "mean":
        # default behavior, pass
        pass
    else:
        raise NotImplementedError(m.get_ddp_hook())

    #########
    # rollout

    out_folder = Path(m.test_output_dir) / m.expt_name
    out_folder.mkdir(exist_ok=True, parents=True)

    # global sim indexed
    sim_to_mesh_type_dict = torch.load(Path(m.PNNL_PARAMS["sim_to_mesh_dict_path"]))[
        "sim_to_mesh_type_dict"
    ]

    node_info_dicts_file = out_folder / "node_info_dicts"
    if node_info_dicts_file.exists():
        print("|---Loading Saved Nodes Info Dicts---|")
        node_info_dicts = torch.load(node_info_dicts_file)
    else:
        mesh_types = ["mesh_type_0", "mesh_type_1", "mesh_type_2"]
        node_info_dicts = {}
        for m_idx, mesh_type in enumerate(mesh_types):
            mesh_type_path = Path(m.PNNL_PARAMS["patch_dict_base_path"]) / mesh_type

            print("|---Generating Nodes Info Dicts for Mesh {0}---|".format(m_idx))
            mesh_type_node_dicts = {
                "non_ghost_nodes_dict": generate_non_ghost_nodes_dict(mesh_type_path),
                "ghost_nodes_info_dict": generate_ghost_nodes_info_dict(mesh_type_path),
            }
            node_info_dicts[m_idx] = mesh_type_node_dicts
        print("|---Saving Nodes Info Dicts for all Meshes ---|")
        if env.rank == 0:
            torch.save(node_info_dicts, node_info_dicts_file)

    num_dynamic_feats = dataset.output_dim

    # minus 1 since starting state counts
    num_time_steps = dataset.time_range[1] - 1

    # each rank will be in charge of rolling out/syncing specific patches
    rank_workload = math.ceil(dataset.n_patches / env.world_size)
    all_rank_j_ranges = []
    for j in range(0, dataset.n_patches, rank_workload):
        j_indices_range = range(
            j,
            j + rank_workload
            if j + rank_workload <= dataset.n_patches
            else dataset.n_patches,
        )
        all_rank_j_ranges.append(j_indices_range)

    rank_j_ranges = all_rank_j_ranges[env.rank]
    print(
        "Rank {0}/{1} is in charge of patches: {2}".format(
            env.rank, env.world_size, rank_j_ranges
        )
    )

    print("-" * 25)
    print(
        "Starting Rollouts on | {0} Sims | {1} Timesteps | {2} Patches |".format(
            dataset.num_sims, num_time_steps, rank_j_ranges
        )
    )
    print("-" * 25)

    # Process One Simulation at a time
    for i in range(dataset.num_sims):

        distributed.barrier()

        global_sim_idx = dataset.sim_range[i]
        mesh_id = sim_to_mesh_type_dict[global_sim_idx]
        non_ghost_nodes_dict = node_info_dicts[mesh_id]["non_ghost_nodes_dict"]
        ghost_nodes_info_dict = node_info_dicts[mesh_id]["ghost_nodes_info_dict"]
        tot_nodes = sum([len(v["global_idx"]) for v in non_ghost_nodes_dict.values()])

        sim_out_folder = out_folder / "simulation_{0}".format(i)
        sim_out_folder.mkdir(exist_ok=True, parents=True)

        aggregated_full_domain_out_folder = sim_out_folder / "aggregated_full_domain/"
        aggregated_full_domain_out_folder.mkdir(exist_ok=True)

        print("-" * 10)
        print("| Simulation {0}/{1} |".format(i, dataset.num_sims))
        print("-" * 10)

        rollout_outfile = sim_out_folder / "rollout.pk"

        if rollout_outfile.exists():
            print("Already Rolled Out Simulation {0}, Skipping...".format(i))
            continue

        patch_infos = {}

        distributed.barrier()

        # Iterate Single Timesteps at a time
        for k in range(num_time_steps):

            distributed.barrier()

            if env.rank == 0:
                with open(sim_out_folder / "progress.txt", mode="a") as f:
                    f.write(
                        "| Simulation {0}/{1} | Timestep {2}/{3} |".format(
                            i, dataset.num_sims, k, num_time_steps
                        )
                    )
                    f.write("\n")

            print("-" * 5)
            print(
                "| Simulation {0}/{1} | Timestep {2}/{3} |".format(
                    i, dataset.num_sims, k, num_time_steps
                )
            )
            print("-" * 5)

            unsynced_timestep_out_folder = sim_out_folder / "unsynced_timestep"
            unsynced_timestep_out_folder.mkdir(exist_ok=True)

            synced_timestep_out_folder = sim_out_folder / "synced_timestep"
            synced_timestep_out_folder.mkdir(exist_ok=True)

            full_domain_state_outfile = (
                aggregated_full_domain_out_folder
                / "full_domain_state_timestep_{0}.pt".format(k)
            )

            distributed.barrier()

            print("|--- Predicting All Patches (Unsync)---|")
            # Rollout Individual Patches
            for j in rank_j_ranges:
                # check to see if we've run this before
                outfile = unsynced_timestep_out_folder / "patch_{0}.pt".format(j)

                if j in patch_infos:
                    d = patch_infos[j]
                else:
                    d = dataset.rollout_info(
                        patch_num=j,
                        sim_num=i,
                        rollout_start_idx=m.rollout_start_idx,
                        window_length=data_params["window_length"],
                        sources=m.source_node_types,
                    )
                    patch_infos[j] = d

                if d["num_nodes"] == 0:
                    continue

                source_data_k_j = (
                    d["source_data"][0],
                    tensor(d["source_data"][1], device=device),
                )

                graph = d["graph"]
                input_graph = Data(
                    edge_index=graph.edge_index, edge_attr=graph.edge_attr
                ).to(device)
                initial_window_data = tensor(
                    d["initial_window_data"], device=device
                ).unsqueeze(0)

                if k == 0:
                    current_window_data = initial_window_data
                else:
                    previous_synced_outfile = (
                        synced_timestep_out_folder / "patch_{0}.pt".format(j)
                    )
                    previous_synced_state = torch.load(previous_synced_outfile)

                    # update dynamic features only
                    current_window_data = initial_window_data
                    current_window_data[
                        0, :, :num_dynamic_feats
                    ] = previous_synced_state

                next_state = base_model.rollout_step(
                    device=device,
                    input_graph=input_graph,
                    current_window_data=current_window_data,
                    node_types=None,
                    node_coordinates=None,
                    onehot_info=d["onehot_info"],
                    source_data=source_data_k_j,
                    update_function=d["update_function"],
                    shift_source=d["shift_source"],
                    update_func_kwargs={"output_type": m.get_output_type()},
                )

                # save dynamic features only
                next_state = next_state[:, :num_dynamic_feats]
                next_state = next_state.cpu()

                assert next_state.shape == (
                    d["num_nodes"],
                    num_dynamic_feats,
                ), next_state.shape

                torch.save(next_state, outfile)

            distributed.barrier()

            unsynced_states = {}
            print("|---Loading All Unsynced Patches---|")
            for j in range(dataset.n_patches):
                current_patch = "patch_{0}".format(j)
                current_patch_state_file = (
                    unsynced_timestep_out_folder / "{0}.pt".format(current_patch)
                )
                if current_patch_state_file.exists():
                    current_patch_state = torch.load(current_patch_state_file)
                    unsynced_states[j] = current_patch_state

            print("|--- Updating All Patches (Sync)---|")
            for j in rank_j_ranges:
                current_patch = "patch_{0}".format(j)
                if j in unsynced_states:
                    updated_patch_state = unsynced_states[j].clone()
                else:
                    continue

                for needed_patch in ghost_nodes_info_dict[current_patch]:
                    needed_patch_idx = int(needed_patch.split("patch_")[1])
                    needed_patch_state = unsynced_states[needed_patch_idx]
                    needed_patch_local_idx = ghost_nodes_info_dict[current_patch][
                        needed_patch
                    ]["source_local_idx"]
                    current_patch_local_idx = ghost_nodes_info_dict[current_patch][
                        needed_patch
                    ]["dest_local_idx"]
                    updated_patch_state[current_patch_local_idx] = needed_patch_state[
                        needed_patch_local_idx
                    ]

                torch.save(
                    updated_patch_state,
                    synced_timestep_out_folder / "{0}.pt".format(current_patch),
                )

            distributed.barrier()

            if env.rank == 0:

                if not args.save_all_timesteps and (k < (num_time_steps - 1)):
                    print("-" * 5)
                    print(
                        "|---Skipping Full Domain Aggregation Until Last Timestep---|"
                    )
                    print("-" * 5)
                elif args.save_all_timesteps or (k == (num_time_steps - 1)):
                    print("-" * 5)
                    print(
                        "| Simulation {0}/{1} | Full Domain Aggregation Timestep {2}/{3} |".format(
                            i, dataset.num_sims, k, num_time_steps
                        )
                    )
                    print("-" * 5)
                    # Aggregate Synced Patch States into Full Domain State
                    full_domain_state = torch.zeros(tot_nodes, num_dynamic_feats)
                    for j in range(dataset.n_patches):
                        current_patch = "patch_{0}".format(j)
                        current_patch_state_file = (
                            synced_timestep_out_folder / "{0}.pt".format(current_patch)
                        )
                        if current_patch_state_file.exists():
                            current_patch_state = torch.load(current_patch_state_file)
                            global_idx = non_ghost_nodes_dict[current_patch][
                                "global_idx"
                            ]
                            current_patch_non_ghost_node_local_idx = (
                                non_ghost_nodes_dict[current_patch]["local_idx"]
                            )
                            current_patch_non_ghost_node_states = current_patch_state[
                                current_patch_non_ghost_node_local_idx
                            ]
                            full_domain_state[
                                global_idx
                            ] = current_patch_non_ghost_node_states

                    torch.save(full_domain_state, full_domain_state_outfile)

            distributed.barrier()

        if env.rank == 0:
            print("|---Removing last timestep folders---|")
            shutil.rmtree(synced_timestep_out_folder)
            shutil.rmtree(unsynced_timestep_out_folder)

        distributed.barrier()

        if env.rank == 0:
            if not args.save_all_timesteps:
                # if not saving all timesteps, save just the final full domain state as the "rollout.pk"
                # note that `full_domain_state` contains the full domain aggregation from the last timestep, (num_time_steps - 1)
                # (the reset for this variable occurs in the inner loop of k -> num_time_steps)
                with open(rollout_outfile, mode="wb") as f:
                    pickle.dump(full_domain_state, f)

            elif args.save_all_timesteps:
                print("-" * 5)
                print("|---Saving the Full Domain Aggregation Across All Timesteps---|")
                print("-" * 5)
                # Combine Full Domain States across Timesteps
                all_rollouts = []
                for k in range(num_time_steps):
                    current_full_domain_state_file = (
                        aggregated_full_domain_out_folder
                        / "full_domain_state_timestep_{0}.pt".format(k)
                    )
                    current_full_domain_state = torch.load(
                        current_full_domain_state_file
                    )
                    all_rollouts.append(current_full_domain_state)

                # Save simulation rollout
                sim_rollout = np.array(torch.stack(all_rollouts))
                with open(rollout_outfile, mode="wb") as f:
                    pickle.dump(sim_rollout, f)

        distributed.barrier()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    env = CustomLSFEnvironment()

    main(args, env)
