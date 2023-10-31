from tqdm import tqdm
from torch import no_grad, cat, save, load, distributed, tensor, is_tensor
from os import popen, environ, makedirs, listdir
from os.path import join, exists
from time import time
import random

# import logging

from torch_geometric.data import DataLoader, DataListLoader
from torch.utils.data import Subset
from GNN.utils.ExpLR import ExpLR
from GNN.utils.plot_utils import interval_plot

from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch import autocast


def train_epoch(
    train_loader,
    model,
    loss_function,
    optimizer,
    lr_scheduler,
    device,
    parallel=False,
    ddp=False,
    grad_accum_steps=1,
    amp=False,
    tb_writer=None,
    tb_rate=1,
    global_step=0,
    env=None,
    steps=None,
    time_limit_hours=1,
):
    """
    Runs one training epoch

    train_loader: train dataset loader
    model: pytorch model
    loss_function: loss function
    optimizer: optimizer
    lr_scheduler: learning rate scheduler
    device: pytorch device
    parallel: if True, train on parallel gpus
    ddp: whether to use distributed training
    tb_writer: tensorboard writer
    tb_rate: tensorboard logging rate
    global_step: current step/batch number
    env: CustomLSFEnvironment from GNN.utils.cluster
    steps: number of training steps (when not using epoch-based training)
    """

    # autocast gpu device must be "cuda" without any attached device ids
    amp_device = "cpu" if device == "cpu" else "cuda"
    sample_wise_losses = []
    total_loss = 0
    running_loss = 0
    start_time = time()
    lim_steps = steps if steps else 1e15  # need a numeric version of steps for if con.

    model.train()

    tqdm_itr = tqdm(train_loader, position=1, desc="Training", leave=True)

    for i, data_list in enumerate(tqdm_itr):
        with autocast(amp_device, enabled=amp):
            if not parallel:
                data_list = data_list.to(device)

            predicted = model(data_list)

            if parallel:
                y = cat([data.y for data in data_list]).to(predicted.device)
                m = model.module
            else:
                y = data_list.y
                m = model

            # if ddp is on, we need to go down on more level of .module
            if ddp:
                m = m.module

            assert hasattr(m, "_output_normalizer")
            y = m._output_normalizer(y, accumulate=m.training)

            # implement "get_non_source_data_mask"
            dataset = train_loader.dataset

            mask = dataset.get_non_source_data_mask(data_list)
            loss = loss_function(predicted[mask], y[mask])

            # normalize loss for gradient accumulation
            loss = loss / grad_accum_steps

        loss.backward()

        # step every grad_accum_steps, or at final iteration of epoch to account for the last few remaining batches
        if ((i + 1) % grad_accum_steps == 0) or (i + 1 == len(train_loader)):
            optimizer.step()
            if "ReduceLROnPlateau" not in str(lr_scheduler.__class__):
                lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        total_loss += loss.item()
        running_loss += loss.item()

        # gather sample-wise losses for possible selective backprop during the main loop
        sample_wise_loss = get_sample_wise_losses(
            ins=data_list,
            outs=predicted,
            mask=mask,
            loss_function=loss_function,
            use_parallel=parallel,
        )
        sample_wise_losses += sample_wise_loss

        if global_step % tb_rate == 0:
            if ddp:
                running_loss = tensor(running_loss, device=device)
                distributed.all_reduce(running_loss)  # does sum reduction by default...
                running_loss = (
                    running_loss.item() / env.world_size
                )  # ...divide by world size to get mean loss
            # if using ddp, only rank 0 will have a tb_writer!
            if tb_writer is not None:
                running_loss = running_loss / tb_rate
                tb_writer.add_scalar(
                    "Loss/train", scalar_value=running_loss, global_step=global_step
                )
            tqdm_itr.set_postfix_str("Train loss: %f" % (total_loss / (i + 1)))
            # reset loss for all ranks
            running_loss = 0.0

        # 1 hour hard-coded limit
        elapsed = (time() - start_time) / 3600
        if (elapsed > time_limit_hours or global_step >= lim_steps) and (
            (i + 1) % grad_accum_steps == 0
        ):
            total_loss = total_loss / (i + 1)
            if ddp:
                total_loss = tensor(total_loss, device=device)
                distributed.all_reduce(total_loss)  # does sum reduction by default...
                total_loss = (
                    total_loss.item() / env.world_size
                )  # ...divide by world size to get mean loss
            return total_loss, global_step, sample_wise_losses

    total_loss = total_loss / len(train_loader)
    if ddp:
        total_loss = tensor(total_loss, device=device)
        distributed.all_reduce(total_loss)  # does sum reduction by default...
        total_loss = (
            total_loss.item() / env.world_size
        )  # ...divide by world size to get mean loss

    return total_loss, global_step, sample_wise_losses


def val_epoch(
    val_loader,
    model,
    loss_function,
    device,
    parallel=False,
    ddp=False,
    amp=False,
    tb_writer=None,
    global_step=0,
    epoch=0,
    env=None,
    time_limit_hours=1,
    sample_limit=1e6,
):
    """
    Runs one validation epoch

    val_loader: validation dataset loader
    model: pytorch model
    loss_function: loss function
    device: pytorch device
    parallel: if True, run on parallel gpus
    tb_writer: tensorboard writer
    global_step: current training step/batch number
    epoch: current training epoch
    env: CustomLSFEnvironment from GNN.utils.cluster
    time_limit_hours: in hours, an upper limit on how long to spend inside this function
    sample_limit: in batches, an upper limit on how many batches this function runs (interacts with env.world_size)
    """

    # autocast gpu device must be "cuda" without any attached device ids
    amp_device = "cpu" if device == "cpu" else "cuda"
    total_loss = 0
    start_time = time()

    model.eval()

    tqdm_itr = tqdm(val_loader, position=1, desc="Validation", leave=True)

    for i, data_list in enumerate(tqdm_itr):
        with autocast(amp_device, enabled=amp):
            if not parallel:
                data_list = data_list.to(device)

            with no_grad():
                predicted = model(data_list)
                if parallel:
                    y = cat([data.y for data in data_list]).to(predicted.device)
                    m = model.module
                else:
                    y = data_list.y
                    m = model

                if ddp:
                    m = m.module

                assert hasattr(m, "_output_normalizer")
                y = m._output_normalizer(y, accumulate=m.training)

                # implement "get_non_source_data_mask"
                dataset = val_loader.dataset

                mask = dataset.get_non_source_data_mask(data_list)
                loss = loss_function(predicted[mask], y[mask])

            total_loss += loss.item()

            tqdm_itr.set_description("Validation")
            tqdm_itr.set_postfix_str("Val loss: %f" % (loss.item()))

        elapsed = (time() - start_time) / 3600
        sample_limit_divisor = 1 if not env else env.world_size
        if elapsed > time_limit_hours or (i + 1) > sample_limit / sample_limit_divisor:
            val_loss = total_loss / (i + 1)
            if ddp:
                val_loss = tensor(val_loss, device=device)
                distributed.all_reduce(val_loss)  # does sum reduction by default...
                val_loss = (
                    val_loss.item() / env.world_size
                )  # ...divide by world size to get mean loss
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "Loss/val", scalar_value=val_loss, global_step=global_step
                )
                tb_writer.add_scalar(
                    "Loss/val_epoch", scalar_value=val_loss, global_step=epoch
                )
            return val_loss

    val_loss = total_loss / len(val_loader)

    if ddp:
        val_loss = tensor(val_loss, device=device)
        distributed.all_reduce(val_loss)  # does sum reduction by default...
        val_loss = (
            val_loss.item() / env.world_size
        )  # ...divide by world size to get mean loss
    if tb_writer is not None:
        tb_writer.add_scalar("Loss/val", scalar_value=val_loss, global_step=global_step)
        tb_writer.add_scalar("Loss/val_epoch", scalar_value=val_loss, global_step=epoch)

    return val_loss


def get_free_gpus(mem_threshold=100, percent_threshold=None):
    """
    Gets current free gpus

    mem_threshold: maximum allowed memory currently in use to be considered a free gpu
    percent_threshold: maximum allowed memory (as percentage of total) currently in use
                     to be considered a free gpu
    """

    with popen("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used") as f:
        gpu_info = f.readlines()
    memory_used = [int(x.split()[2]) for x in gpu_info]
    free_gpus = [i for i, mem in enumerate(memory_used) if mem <= mem_threshold]
    if percent_threshold is not None:
        with popen("nvidia-smi -q -d Memory |grep -A4 GPU|grep Total") as f:
            gpu_info = f.readlines()
        memory_total = [int(x.split()[2]) for x in gpu_info]
        free_gpus = [
            i
            for i, (used, total) in enumerate(zip(memory_used, memory_total))
            if 100 * used / total <= percent_threshold
        ]

    if "CUDA_VISIBLE_DEVICES" in environ:
        free_gpus = list(
            set(free_gpus).intersection(
                list(map(int, environ["CUDA_VISIBLE_DEVICES"].split(",")))
            )
        )

    return free_gpus


def interval_actions(
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
):
    """
    Actions to perform every n steps/epochs of training.

    Currently, we save the model and compute rollout IA if the dataset is PNNL's.
    """
    interval_dir = join(checkpoint_dir, "interval_models")
    makedirs(interval_dir, exist_ok=True)
    files = listdir(interval_dir)
    latest_saved_step = 0
    for f in files:
        step = int(f.split("_")[1])
        if step > latest_saved_step:
            latest_saved_step = step
    if pbar.n - latest_saved_step >= m.get_save_interval():
        prefix = "iter_" + str(pbar.n)
        save_model(
            checkpoint_dir=interval_dir,
            model_filename=f"{prefix}_model.pt",
            model=model,
            scheduler_filename=f"{prefix}_scheduler.pt",
            scheduler=lr_scheduler,
            optimizer_filename=f"{prefix}_optimizer.pt",
            optimizer=optimizer,
            epoch_filename=f"{prefix}_epoch.pt",
            epoch=epoch,
            info_filename=f"{prefix}_info.pt",
            info={"global_step": global_step} if steps else None,
        )
        if m.get_run_rollout_at_interval():
            from GNN.utils.ia_utils import Metrics
            from GNN.utils.rollout import minimal_rollout as rollout_func

            # run a rollout on the 0th (first) test simulation
            rollout = rollout_func(test_dataset, model, 0, m, device)
            # compute error on third output variable (vol_frac_col=2)
            # TODO: let user define the variable to be analyzed
            IA_dict = Metrics(
                rollout["pred"], rollout["actual"], rollout["coords"], vol_frac_col=2
            ).IA_rel_error
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "IA/IA relative error",
                    scalar_value=IA_dict["IA_error"],
                    global_step=global_step,
                )
                tb_writer.add_scalar(
                    "IA/IA pred",
                    scalar_value=IA_dict["IA_pred"],
                    global_step=global_step,
                )
                tb_writer.add_scalar(
                    "IA/IA act", scalar_value=IA_dict["IA_act"], global_step=global_step
                )
                tb_writer.add_figure(
                    "Viz/Rollout (Predicted vs. Actual)",
                    interval_plot(
                        rollout["pred"][:, :, 2],
                        rollout["actual"][:, :, 2],
                        rollout["coords"],
                        rollout["steps"],
                    ),
                    global_step=global_step,
                )


def save_model(
    checkpoint_dir,
    model_filename,
    model,
    scheduler_filename=None,
    scheduler=None,
    optimizer_filename=None,
    optimizer=None,
    epoch_filename=None,
    epoch=None,
    config_filename=None,
    config=None,
    info_filename=None,
    info=None,
    best_loss_filename=None,
    best_loss=None,
):
    """
    Saves model checkpoint
    """

    # strips all module wrappers from the model, catches all module wrap cases (DataParallel, DistributedDataParallel, etc)
    # while loop allows handling of multiple nested wrappers.
    while True:
        try:
            model = model.module
        except:
            break

    save(model.state_dict(), join(checkpoint_dir, model_filename))

    if (scheduler_filename is not None) and (scheduler is not None):
        save(scheduler.state_dict(), join(checkpoint_dir, scheduler_filename))

    if (optimizer_filename is not None) and (optimizer is not None):
        save(optimizer.state_dict(), join(checkpoint_dir, optimizer_filename))

    if (epoch_filename is not None) and epoch is not None:
        save(epoch, join(checkpoint_dir, epoch_filename))

    if (config_filename is not None) and (config is not None):
        save(config, join(checkpoint_dir, config_filename))

    if (info_filename is not None) and (info is not None):
        save(info, join(checkpoint_dir, info_filename))

    if (best_loss_filename is not None) and (best_loss is not None):
        save(best_loss, join(checkpoint_dir, best_loss_filename))


def load_model(checkpoint_dir, model_filename, model):
    if exists(join(checkpoint_dir, model_filename)):
        print("RELOADING MODEL STATE DICT")
        sd = load(join(checkpoint_dir, model_filename), map_location="cpu")
        new = {}
        for k, v in sd.items():
            if k[:7] == "module.":
                new[k[7:]] = v
            else:
                new[k] = v
        model.load_state_dict(new)

    return model


def load_model_train_vars(
    checkpoint_dir,
    scheduler_filename=None,
    scheduler=None,
    optimizer_filename=None,
    optimizer=None,
    epoch_filename=None,
    info_filename=None,
    best_loss_filename=None,
):
    """
    Loads model checkpoint
    """

    if (scheduler_filename is not None) and exists(
        join(checkpoint_dir, scheduler_filename)
    ):
        print("RELOADING SCHEDULER STATE DICT")
        scheduler.load_state_dict(
            load(join(checkpoint_dir, scheduler_filename), map_location="cpu")
        )

    if (optimizer_filename is not None) and exists(
        join(checkpoint_dir, optimizer_filename)
    ):
        print("RELOADING OPTIMIZER STATE DICT")
        optimizer.load_state_dict(
            load(join(checkpoint_dir, optimizer_filename), map_location="cpu")
        )

    if (epoch_filename is not None) and exists(join(checkpoint_dir, epoch_filename)):
        print(f"RELOADING {epoch_filename[:4]} EPOCH")
        last_epoch = load(join(checkpoint_dir, epoch_filename), map_location="cpu")
    else:
        last_epoch = -1

    if (info_filename is not None) and exists(join(checkpoint_dir, info_filename)):
        print(f"RELOADING INFO FROM {info_filename[:4]} EPOCH")
        info = load(join(checkpoint_dir, info_filename), map_location="cpu")
    else:
        info = {"global_step": 0}

    if (best_loss_filename is not None) and exists(
        join(checkpoint_dir, best_loss_filename)
    ):
        best_loss = load(join(checkpoint_dir, best_loss_filename))
    else:
        best_loss = None

    return scheduler, optimizer, last_epoch, info, best_loss


def get_lr_scheduler(model_config, train_loader, optimizer):

    scheduler = model_config.get_scheduler()
    scheduler_params = model_config.get_scheduler_params()

    epochs = model_config.get_epochs()
    steps = model_config.get_steps()
    if steps:
        epochs = -(-steps // len(train_loader))
    else:
        steps = epochs * len(train_loader)

    if scheduler == "ExpLR":
        if "decay_steps" not in scheduler_params:
            num_decay_steps = steps * scheduler_params["scheduler_decay_interval"]
            scheduler_params["decay_steps"] = num_decay_steps
        return ExpLR(
            optimizer,
            decay_steps=scheduler_params["decay_steps"],
            gamma=scheduler_params["gamma"],
            min_lr=scheduler_params["min_lr"],
        )

    elif scheduler == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=scheduler_params["max_lr"],
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )
    elif scheduler == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, patience=scheduler_params["patience"])

    else:
        raise Exception("LR scheduler not recognized")


def check_lr_scheduler(lr_scheduler, config, num_decay_steps, global_step):
    # check if current lr_sched configs are different from checkpoint's lr sched and then override
    if lr_scheduler.decay_steps != num_decay_steps:
        print(
            f"WARNING: lr_sched decay_steps is different from num_train_steps. Overriding..: {lr_scheduler.decay_steps} to be {num_decay_steps}"
        )
        lr_scheduler.decay_steps = num_decay_steps

    if "gamma" in config.get_scheduler_params():
        config_gamma = config.get_scheduler_params()["gamma"]
        if lr_scheduler.gamma != config_gamma:
            print(
                f"WARNING: lr_sched gamma is different from gamma. Overriding..: {lr_scheduler.gamma} to be {config_gamma}"
            )
            lr_scheduler.gamma = config.get_scheduler_gamma()

    if lr_scheduler.last_epoch != global_step:
        print(
            f"WARNING: lr_sched last_epoch is different from global_step . Overriding..: {lr_scheduler.last_epoch} to be {global_step}"
        )
        lr_scheduler.last_epoch = global_step


def get_max_epochs_steps(
    dataloader,
    env,
    max_steps=None,
    max_epochs=None,
    accumulate_grad_batches=1,
    stage="fit",
    one_dataset_per_rank=False,
    use_distributed_sampler=False,
):
    if stage == "fit":
        # calculate train_steps for each ddp process
        total_devices = env.world_size
        if one_dataset_per_rank or use_distributed_sampler:
            # if one_dataset_per_rank, or
            # using distributed sampler (current length of dataloader would have been divided among world_size already)
            # then no need to divide number of train_batches by total_devices
            train_batches = len(dataloader)
        else:
            train_batches = len(dataloader) // total_devices

        if max_steps is not None:
            # epochs can be fractional
            epochs = float(max_steps) / train_batches
        elif max_epochs is not None:
            epochs = max_epochs
        else:
            raise ValueError("Either max_epochs or max_steps should be defined.")

        max_epochs = epochs
        max_steps = (epochs * train_batches) // accumulate_grad_batches
        print(
            f"Total train step: max_steps: {max_steps}, max_epochs: {max_epochs}, length dataloader: {len(dataloader)}, total devices: {total_devices}, accumulate grad batches: {accumulate_grad_batches}"
        )
        return max_epochs, max_steps

    else:
        raise NotImplementedError()


def get_sample_wise_losses(ins, outs, mask, loss_function, use_parallel):

    sample_wise_loss = []

    if use_parallel:
        slice_indices = [0]
        for sample in ins:
            slice_indices.append(slice_indices[-1] + len(sample.x))
        y = cat([sample.y for sample in ins]).to(outs.device)
    else:
        slice_indices = ins.ptr
        y = ins.y

    for i in range(len(slice_indices) - 1):

        slice_range = (slice_indices[i], slice_indices[i + 1])

        sample_y_label = y[slice_range[0] : slice_range[1]]
        sample_y_pred = outs[slice_range[0] : slice_range[1]]

        sample_dataset_idx = ins[i].dataset_idx
        if is_tensor(sample_dataset_idx):
            sample_dataset_idx = sample_dataset_idx.item()

        sample_loss = loss_function(sample_y_pred, sample_y_label).item()

        sample_wise_loss.append(
            {"dataset_idx": sample_dataset_idx, "loss": sample_loss}
        )

    return sample_wise_loss


def sort_sample_wise_losses(sample_wise_loss):
    sorted_sample_wise_loss = sorted(
        sample_wise_loss, key=lambda x: x["loss"], reverse=True
    )

    return sorted_sample_wise_loss


# set up train loader based on sbp or full epochs
def setup_sbp_train_loader(
    epoch,
    sbp_start_epoch,
    sbp_rate,
    sbp_percent,
    sbp_randomess,
    sample_wise_losses,
    orig_train_dataset,
    batch_size,
    use_parallel,
):

    # Every nth epoch will be a full epoch, use full (original) train dataloader
    if (
        (epoch % sbp_rate == 0)
        or (epoch < sbp_start_epoch)
        or (len(sample_wise_losses) == 0)
    ):
        if use_parallel:
            train_loader = DataListLoader(
                orig_train_dataset, batch_size=batch_size, shuffle=True
            )
        else:
            train_loader = DataLoader(
                orig_train_dataset, batch_size=batch_size, shuffle=True
            )

        return train_loader

    # sbp epochs, use subset based on sbp_percent highest loss samples.
    else:
        sorted_sample_wise_losses = sort_sample_wise_losses(sample_wise_losses)
        highest_sample_loss_indices = [
            s["dataset_idx"] for s in sorted_sample_wise_losses
        ]

        sbp_k = int(sbp_percent * len(orig_train_dataset))
        sbp_rand = int(sbp_randomess * len(orig_train_dataset))

        top_sbp_percent_samples = highest_sample_loss_indices[:sbp_k]

        # randomly modify sbp_randomess percent of samples.
        random.shuffle(top_sbp_percent_samples)
        top_sbp_percent_samples = top_sbp_percent_samples[:-sbp_rand]

        # finds missing values from an existing list of values, uses a set operation: difference
        unused_sample_indices = list(
            set(range(len(orig_train_dataset))).difference(top_sbp_percent_samples)
        )
        random.shuffle(unused_sample_indices)
        random_sbp_sample_indices = unused_sample_indices[:sbp_rand]

        top_sbp_percent_samples += random_sbp_sample_indices

        subset_train_dataset = Subset(orig_train_dataset, top_sbp_percent_samples)
        if use_parallel:
            train_loader = DataListLoader(
                subset_train_dataset, batch_size=batch_size, shuffle=True
            )
        else:
            train_loader = DataLoader(
                subset_train_dataset, batch_size=batch_size, shuffle=True
            )

        return train_loader
