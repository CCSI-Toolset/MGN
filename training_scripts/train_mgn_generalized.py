from importlib import import_module
import os
from sys import path
import time
import argparse
import tqdm

import torch
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# for parallel:
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel

#########
# local application imports
# get path to root of the project
mgn_code_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
path.append(mgn_code_dir)

from GNN.ModelConfig.ConfigMGN import ModelConfig
from GNN.MeshGraphNets import MeshGraphNets
from GNN.MultiMeshGraphNets import MultiMeshGraphNets
from GNN.utils.train_utils import (
    train_epoch,
    val_epoch,
    get_free_gpus,
    save_model,
    load_model,
    load_model_train_vars,
    get_lr_scheduler,
    interval_actions,
    setup_sbp_train_loader,
)
from GNN.utils.utils import get_delta_minute


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Config file to train model. See ../configfiles for examples",
    )
    return parser


def run(args):

    # ----- Config/Parser -----

    m = ModelConfig(args.config_file)

    train_data_params = m.get_train_data_params()
    test_data_params = m.get_test_data_params()

    # ----- Datasets -----

    ds_class_name = m.get_class_name()
    DS = getattr(import_module("GNN.DatasetClasses." + ds_class_name), ds_class_name)
    train_dataset = DS(model_config=m, train_set=True, **train_data_params)
    test_dataset = DS(model_config=m, train_set=False, **test_data_params)

    # ----- Device Settings -----

    batch_size = m.get_batch_size()
    free_gpus = get_free_gpus()

    device = torch.device(
        "cuda:" + str(free_gpus[0])
        if torch.cuda.is_available() and len(free_gpus) > 0
        else "cpu"
    )
    if len(free_gpus) == 0 and device != "cpu":
        raise Exception("No free GPUs")

    use_parallel = m.get_use_parallel() and len(free_gpus) > 1 and device != "cpu"

    # hard-coded to be set to False since this script is for single node usage!
    use_ddp = False
    # hard-coded to be set to False due to stability issues
    use_amp = False
    use_sbp = m.get_use_sbp()

    if use_parallel:
        train_loader = DataListLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # ----- Model & Training Setup -----

    mgn_dim = m.get_mgn_dim()

    model_arch = m.get_model_arch()

    assert model_arch in [
        "MeshGraphNets",
        "MultiMeshGraphNets",
    ], "Invalid `model_arch`, set to either 'MeshGraphNets' or 'MultiMeshGraphNets'"

    if model_arch == "MeshGraphNets":
        model = MeshGraphNets(
            train_dataset.num_node_features,
            train_dataset.num_edge_features,
            train_dataset.output_dim,
            out_dim_node=mgn_dim,
            out_dim_edge=mgn_dim,
            hidden_dim_node=mgn_dim,
            hidden_dim_edge=mgn_dim,
            hidden_dim_processor_node=mgn_dim,
            hidden_dim_processor_edge=mgn_dim,
            hidden_dim_decoder=mgn_dim,
            node_normalizer_mask=train_dataset.node_normalizer_mask,
            edge_normalizer_mask=train_dataset.edge_normalizer_mask,
            mp_iterations=m.get_mp_iterations(),
            mlp_norm_type=m.get_mlp_norm_type(),
            output_type=m.get_output_type(),
            connection_type=m.get_connection_type(),
            connection_alpha=m.get_connection_alpha(),
            connection_aggregation=m.get_connection_aggregation(),
            graph_processor_type=m.get_graph_processor_type(),
            integrator=m.get_integrator(),
        )
    elif model_arch == "MultiMeshGraphNets":
        model = MultiMeshGraphNets(
            train_dataset.num_node_features,
            train_dataset.num_edge_features,
            train_dataset.output_dim,
            num_edge_types=len(train_dataset.graph_types),
            out_dim_node=mgn_dim,
            out_dim_edge=mgn_dim,
            hidden_dim_node=mgn_dim,
            hidden_dim_edge=mgn_dim,
            hidden_dim_processor_node=mgn_dim,
            hidden_dim_processor_edge=mgn_dim,
            hidden_dim_decoder=mgn_dim,
            node_normalizer_mask=train_dataset.node_normalizer_mask,
            edge_normalizer_mask=train_dataset.edge_normalizer_mask,
            mp_iterations=m.get_mp_iterations(),
            mlp_norm_type=m.get_mlp_norm_type(),
            output_type=m.get_output_type(),
            connection_type=m.get_connection_type(),
            connection_alpha=m.get_connection_alpha(),
            connection_aggregation=m.get_connection_aggregation(),
            graph_processor_type=m.get_graph_processor_type(),
            integrator=m.get_integrator(),
        )

    loss_function = torch.nn.MSELoss()

    # reload model, if available
    checkpoint_dir = m.get_checkpoint_dir()
    load_prefix = m.get_load_prefix()

    model = load_model(
        checkpoint_dir=checkpoint_dir,
        model_filename="{0}_model.pt".format(load_prefix),
        model=model,
    )

    # place on device
    model = model.to(device)
    if use_parallel:
        # We put model on DP after loading checkpoint. Be sure to save model without the DP attached.
        model = DataParallel(model, device_ids=free_gpus)

    # ----- Optimizer Settings -----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=m.get_lr(), weight_decay=m.get_weight_decay()
    )
    lr_scheduler = get_lr_scheduler(
        model_config=m, train_loader=train_loader, optimizer=optimizer
    )

    lr_scheduler, optimizer, last_epoch, info, best_loss = load_model_train_vars(
        checkpoint_dir=checkpoint_dir,
        scheduler_filename="{0}_scheduler.pt".format(load_prefix),
        scheduler=lr_scheduler,
        optimizer_filename="{0}_optimizer.pt".format(load_prefix),
        optimizer=optimizer,
        epoch_filename="{0}_epoch.pt".format(load_prefix),
        info_filename="{0}_info.pt".format(load_prefix),
        best_loss_filename="best_val_loss.pt",
    )

    # ----- Training Prep -----

    # tensorboard settings
    tb_dir = m.get_tensorboard_dir()
    tb_rate = m.get_tb_rate()
    # log_rate = m.get_log_rate()

    # create a summary writer.
    tb_writer = None if not m.get_use_tensorboard() else SummaryWriter(tb_dir)

    # ----- Train -----

    # training loop
    if best_loss is None:
        best_loss = val_epoch(
            val_loader=test_loader,
            model=model,
            loss_function=loss_function,
            device=device,
            parallel=use_parallel,
            ddp=use_ddp,
            amp=use_amp,
            time_limit_hours=m.get_validation_time_limit(),
            sample_limit=m.get_sample_limit(),
        )

    # find training time, and (if resuming training) where training left off
    steps = m.get_steps()
    if steps:
        pbar = tqdm.tqdm(total=steps, position=0, leave=True, unit_scale=True)
        global_step = info["global_step"]
        pbar.update(global_step)
    else:
        pbar = tqdm.tqdm(total=m.get_epochs(), position=0, leave=True)
        global_step = (last_epoch + 1) * len(
            train_loader
        )  # assumes batch size does not change on continuation; assumes entire epoch finishes before each model save
        pbar.update(last_epoch + 1)

    grad_accum_steps = m.get_grad_accum_steps()
    sample_wise_losses = []

    start_time = time.time()

    while pbar.n < pbar.total:
        epoch = global_step // len(train_loader) if steps else pbar.n
        pbar.set_description(f"Step {global_step}" if steps else f"Epoch {epoch}")
        prior_step = global_step

        if use_sbp:
            train_loader = setup_sbp_train_loader(
                epoch=epoch,
                sbp_start_epoch=m.get_sbp_start_epoch(),
                sbp_rate=m.get_sbp_rate(),
                sbp_percent=m.get_sbp_percent(),
                sbp_randomess=m.get_sbp_randomness(),
                sample_wise_losses=sample_wise_losses,
                orig_train_dataset=train_dataset,
                batch_size=batch_size,
                use_parallel=use_parallel,
            )

        train_loss, global_step, sample_wise_losses = train_epoch(
            train_loader=train_loader,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            parallel=use_parallel,
            ddp=use_ddp,
            grad_accum_steps=grad_accum_steps,
            amp=use_amp,
            tb_writer=tb_writer,
            tb_rate=tb_rate,
            global_step=global_step,
            steps=steps,
            time_limit_hours=m.get_train_time_limit(),
        )
        pbar.refresh()
        val_loss = val_epoch(
            val_loader=test_loader,
            model=model,
            loss_function=loss_function,
            device=device,
            parallel=use_parallel,
            ddp=use_ddp,
            amp=use_amp,
            tb_writer=tb_writer,
            global_step=global_step,
            epoch=epoch,
            time_limit_hours=m.get_validation_time_limit(),
            sample_limit=m.get_sample_limit(),
        )
        pbar.refresh()

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(
                checkpoint_dir=checkpoint_dir,
                model_filename="best_val_model.pt",
                model=model,
                scheduler_filename="best_val_scheduler.pt",
                scheduler=lr_scheduler,
                optimizer_filename="best_val_optimizer.pt",
                optimizer=optimizer,
                epoch_filename="best_val_epoch.pt",
                epoch=epoch,
                info_filename="best_val_info.pt",
                info={"global_step": global_step} if steps else None,
                best_loss_filename="best_val_loss.pt",
                best_loss=best_loss,
            )

        if "ReduceLROnPlateau" in str(lr_scheduler.__class__):
            lr_scheduler.step(val_loss)

        if tb_writer is not None:
            tb_writer.add_scalar(
                "Opt/lr",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=global_step,
            )
            tb_writer.add_scalar(
                "Profile/epoch_time",
                scalar_value=get_delta_minute(start_time),
                global_step=global_step,
            )

        save_model(
            checkpoint_dir=checkpoint_dir,
            model_filename="last_epoch_model.pt",
            model=model,
            scheduler_filename="last_epoch_scheduler.pt",
            scheduler=lr_scheduler,
            optimizer_filename="last_epoch_optimizer.pt",
            optimizer=optimizer,
            epoch_filename="last_epoch_epoch.pt",
            epoch=epoch,
            info_filename="last_epoch_info.pt",
            info={"global_step": global_step} if steps else None,
        )

        pbar.update(1 if not steps else global_step - prior_step)

        # perform `interval_actions` every m.get_save_interval() epochs/steps
        if m.get_save_interval() is not None:
            interval_actions(
                pbar,
                m,
                checkpoint_dir,
                model,
                lr_scheduler,
                optimizer,
                epoch,
                global_step,
                steps,
                test_dataset,
                tb_writer,
                device,
            )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
