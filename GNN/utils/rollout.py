from importlib import import_module
import os
import pickle
from sys import path
import matplotlib
import argparse
import numpy as np
import torch

# for parallel:
from torch_geometric.nn import DataParallel

#########
# local application imports
# get path to root of the project
mgn_code_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
path.append(mgn_code_dir)

from GNN.ModelConfig.ConfigMGN import ModelConfig
from GNN.MeshGraphNets import MeshGraphNets
from GNN.MultiMeshGraphNets import MultiMeshGraphNets
from GNN.utils.train_utils import get_free_gpus, load_model
from GNN.utils import plot_utils

"""
example usage:
    python GNN/utils/rollout.py config_files/test_pnnl_subdomain.ini --cpu --viz_rollout_num=1 --GT
"""

#########
# load config file
def build_parser_full_domains():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Config file to train model. See ../configfiles for examples",
    )
    parser.add_argument(
        "--rollout_num",
        default=None,
        help="Index of the rollout you want to run, all are run by default",
    )
    parser.add_argument(
        "-test",
        "--test_set",
        action="store_true",
        help="Rollout and/or visualize using test instead of train data",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force rollout to run on the CPU (helpful if you have memory constraints)",
    )
    parser.add_argument(
        "--nobar", action="store_true", help="do not show the progress bar"
    )
    parser.add_argument(
        "--viz_rollout_num",
        default=None,
        help="Index of the simulation you want to visualize a rollout for",
    )
    parser.add_argument(
        "--GT",
        action="store_true",
        help="Visualize the ground truth instead of the predicted rollout",
    )
    return parser


def main_full_domains(args):
    dataset_str = "train" if not args.test_set else "test"
    m = ModelConfig(args.config_file)
    already_ran = False
    if args.rollout_num is not None:
        if os.path.exists(
            f"{m.test_output_dir}/{m.expt_name}/rollout_{dataset_str}_{args.rollout_num}.pk"
        ):
            already_ran = True

    if not already_ran:
        run_and_save_rollout(args, m)

    ######
    # visualize rollout
    if args.viz_rollout_num is not None:
        loaded = pickle.load(
            open(
                f'{m.test_output_dir}/{m.expt_name}/rollout_{dataset_str}_{args.viz_rollout_num}{"_GT" if args.GT else ""}.pk',
                "rb",
            )
        )
        data = loaded["pred"][:, :, :3]
        graph = pickle.load(
            open(
                f"{m.test_output_dir}/{m.expt_name}/rollout_{dataset_str}_{args.viz_rollout_num}.pk",
                "rb",
            )
        )["coords"]
        graph = matplotlib.tri.Triangulation(graph[:, 0], graph[:, 1])
        print(f"Saving visualization to {m.test_output_dir}/{m.expt_name}")
        plot_utils.plot_rollout_separate(
            data,
            graph,
            save_dir=f"{m.test_output_dir}/{m.expt_name}",
            outname=f'rollout_{dataset_str}_{args.viz_rollout_num}{"_GT" if args.GT else ""}.mp4',
        )

    return f"{m.test_output_dir}/{m.expt_name}"


def minimal_rollout(dataset, model, i, m, device):
    """
    runs a rollout assuming you already have a dataset and model loaded.
    assumes rollout starts at iter 0 and window_length is 1.
    i is the simulation in the dataset you want to run a rollout for.
    m is the ModelConfig object.

    returns subset of rollout for performance-monitoring purposes
    """
    old_noise = dataset.noise
    dataset.noise = None  # don't add noise during rollouts
    try:  # get rid of wrapper on model if needed
        model = model.module
    except:
        pass
    d = dataset.rollout_info(i, 0, 1, m.source_node_types)
    ground_truth = d["ground_truth"][1:]
    steps = len(ground_truth) - 1
    steps_to_keep = np.array([int(steps * x) for x in [0.02, 0.2, 0.8, 1]])
    rollout_output = {
        "pred": model.rollout(
            device,
            d["initial_window_data"],
            d["graph"],
            node_types=d["node_types"],
            node_coordinates=None,
            onehot_info=d["onehot_info"],
            rollout_iterations=d["rollout_iterations"],
            source_data=d["source_data"],
            shift_source=d["shift_source"],
            update_function=d["update_function"],
            update_func_kwargs={"output_type": m.get_output_type()},
        )[steps_to_keep],
        "actual": ground_truth[steps_to_keep],
        "coords": d["node_coordinates"],
        "steps": steps_to_keep,
    }
    dataset.noise = old_noise
    return rollout_output


def run_and_save_rollout(args, m):
    #########
    # prepare train or test set
    data_params = (
        m.get_train_data_params() if not args.test_set else m.get_test_data_params()
    )
    dataset_str = "train" if not args.test_set else "test"

    ds_class_name = m.get_class_name()
    DS = getattr(import_module("GNN.DatasetClasses." + ds_class_name), ds_class_name)
    if hasattr(m, "PNNL_PARAMS"):
        print("Rolling out on patch limits rather than individual patches")
        data_prefix = f'{"" if not args.test_set else "test_"}'
        if type(m.PNNL_PARAMS[data_prefix + "x_range"]) is str:
            x_min, x_max = [
                float(x) for x in m.PNNL_PARAMS[data_prefix + "x_range"].split(",")[1:]
            ]
            y_min, y_max = [
                float(y) for y in m.PNNL_PARAMS[data_prefix + "y_range"].split(",")[1:]
            ]
        else:
            x_min, x_max = np.min(m.PNNL_PARAMS[data_prefix + "x_range"]), np.max(
                m.PNNL_PARAMS[data_prefix + "x_range"]
            )
            y_min, y_max = np.min(m.PNNL_PARAMS[data_prefix + "y_range"]), np.max(
                m.PNNL_PARAMS[data_prefix + "y_range"]
            )
        m.PNNL_PARAMS[data_prefix + "x_range"] = [(x_min, x_max)]
        m.PNNL_PARAMS[data_prefix + "y_range"] = [(y_min, y_max)]
    dataset = DS(model_config=m, train_set=not args.test_set, **data_params)
    dataset.noise = None  # don't add noise during rollouts

    #########
    # build model
    mgn_dim = m.get_mgn_dim()
    model_arch = m.get_model_arch()
    assert model_arch in [
        "MeshGraphNets",
        "MultiMeshGraphNets",
    ], "Invalid `model_arch`, set to either 'MeshGraphNets' or 'MultiMeshGraphNets'"

    if model_arch == "MeshGraphNets":
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
            connection_aggregation=m.get_aggregation(),
            graph_processor_type=m.get_graph_processor_type(),
            integrator=m.get_integrator(),
        )
    elif model_arch == "MultiMeshGraphNets":
        model = MultiMeshGraphNets(
            dataset.num_node_features,
            dataset.num_edge_features,
            dataset.output_dim,
            num_edge_types=len(dataset.graph_types),
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
            graph_processor_type=m.get_graph_processor_type(),
            integrator=m.get_integrator(),
        )

    #########
    # device settings
    free_gpus = get_free_gpus(percent_threshold=5)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        free_gpus = list(
            set(free_gpus).intersection(
                list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            )
        )
    device = torch.device(
        "cuda:" + str(free_gpus[0])
        if torch.cuda.is_available() and len(free_gpus) > 0
        else "cpu"
    )
    if args.cpu:
        device = "cpu"
    if len(free_gpus) == 0 and device != "cpu":
        raise Exception("No free GPUs")
    print("Running in device %s" % device)
    use_parallel = m.get_use_parallel() and len(free_gpus) > 1 and device != "cpu"
    use_parallel = False  # for now, just run on 1 process

    #########
    # load model
    checkpoint_dir = m.get_checkpoint_dir()

    model = load_model(checkpoint_dir, model_filename="best_val_model.pt", model=model)
    model = model.to(device)
    base_model = model
    if use_parallel:
        model = DataParallel(model, device_ids=free_gpus)
        base_model = model.module

    #########
    # rollout
    show_progressbar = not args.nobar
    # create rollout dir if needed
    outdir = f"{m.test_output_dir}/{m.expt_name}/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in range(dataset.num_sims):
        # if user wants a specific sim to be rolled out, skip other sims
        if args.rollout_num is not None:
            if i != int(args.rollout_num):
                continue
        # check to see if we've run this rollout before
        outfile = f"{m.test_output_dir}/{m.expt_name}/rollout_{dataset_str}_{i}.pk"
        if os.path.exists(outfile):
            continue

        print("Getting rollout starting point and source data...")
        d = dataset.rollout_info(
            i, m.rollout_start_idx, data_params["window_length"], m.source_node_types
        )

        print(f"Starting rollout {i}")
        rollout_output = {
            "pred": base_model.rollout(
                device,
                d["initial_window_data"],
                d["graph"],
                node_types=d["node_types"],
                node_coordinates=None,
                onehot_info=d["onehot_info"],
                rollout_iterations=d["rollout_iterations"],
                source_data=d["source_data"],
                shift_source=d["shift_source"],
                update_function=d["update_function"],
                update_func_kwargs={"output_type": m.get_output_type()},
                progressbar=show_progressbar,
            ),
            "coords": d["node_coordinates"],
        }

        # save rollout output to test_dir
        # rollout_output is a numpy array of shape (rollout_iterations, num_nodes, feature_size)
        with open(outfile, "wb") as f:
            assert rollout_output["pred"].shape == (
                d["rollout_iterations"],
                d["num_nodes"],
                d["feature_size"],
            ), rollout_output["pred"].shape
            pickle.dump(rollout_output, f)

        # save out ground truth
        outfile = f"{m.test_output_dir}/{m.expt_name}/rollout_{dataset_str}_{i}_GT.pk"
        with open(outfile, "wb") as f:
            rollout_output["pred"] = d["ground_truth"][1:]
            rollout_output[
                "coords"
            ] = 'Get coords from prediction file; i.e., remove "_GT" from this filename.'
            assert rollout_output["pred"].shape == (
                d["rollout_iterations"],
                d["num_nodes"],
                d["feature_size"],
            ), rollout_output["pred"].shape
            pickle.dump(rollout_output, f)


if __name__ == "__main__":
    parser = build_parser_full_domains()
    args = parser.parse_args()
    main_full_domains(args)
