
from tqdm import tqdm
from torch import no_grad, cat, save, load
from os import popen, path
from os.path import join, exists
import pickle

def train_epoch(train_loader, model, loss_function, 
        optimizer, lr_scheduler,
        device, parallel=False, 
        tb_writer=None, tb_rate=1, total_steps=0):

    '''
    Runs one training epoch

    train_loader: train dataset loader
    model: pytorch model
    loss_function: loss function
    optimizer: optimizer
    device: pytorch device
    parallel: if True, train on parallel gpus
    tb_writer: tensorboard writer
    tb_rate: tensorboard logging rate
    total_steps: current step/batch number
    '''

    total_loss = 0
    running_loss = 0

    model.train()

    tqdm_itr = tqdm(train_loader, position=1, desc='Training', leave=True)

    for i, data_list in enumerate(tqdm_itr):
        if not parallel:
            data_list = data_list.to(device)

        optimizer.zero_grad()
        predicted = model(data_list)
        if parallel:
            y = cat([data.y for data in data_list]).to(predicted.device)
            m = model.module
        else:
            y = data_list.y
            m = model

        if hasattr(m, '_output_normalizer'):
            y = m._output_normalizer(y, accumulate=m.training)

        loss = loss_function(predicted, y)
        loss.backward()
        optimizer.step()
        if not 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
            lr_scheduler.step()

        total_loss += loss.item()
        running_loss += loss.item()
        total_steps += 1

        tqdm_itr.set_postfix_str('Train loss: %f' % (total_loss / (i + 1)))

        if (tb_writer is not None) and (total_steps % tb_rate == 0):
            tb_writer.add_scalar("Loss/train", scalar_value=running_loss / tb_rate, global_step=total_steps)
            running_loss = 0.0

    return total_loss / len(train_loader), total_steps


def val_epoch(val_loader, model, loss_function, device, parallel=False,
    tb_writer=None, total_steps=0, epoch=0):
    
    '''
    Runs one validation epoch

    val_loader: validation dataset loader
    model: pytorch model
    loss_function: loss function
    device: pytorch device
    parallel: if True, run on parallel gpus
    tb_writer: tensorboard writer
    total_steps: current training step/batch number
    epoch: current training epoch
    '''

    total_loss = 0

    model.eval()

    tqdm_itr = tqdm(val_loader, position=1, desc='Validation', leave=True)

    for i, data_list in enumerate(tqdm_itr):
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

            if hasattr(m, '_output_normalizer'):
                y = m._output_normalizer(y, accumulate=m.training)

            loss = loss_function(predicted, y)

            total_loss += loss.item()

            tqdm_itr.set_description('Validation')
            tqdm_itr.set_postfix_str('Val loss: %f' % (total_loss / (i + 1)))

        val_loss = total_loss / len(val_loader)

        if (tb_writer is not None):
            tb_writer.add_scalar("Loss/val", scalar_value=val_loss, global_step=total_steps)
            tb_writer.add_scalar("Loss/val_epoch", scalar_value=val_loss, global_step=epoch)


    return val_loss

def get_free_gpus(mem_threshold=100):

    '''
    Gets current free gpus

    mem_threshold: maximum allowed memory currently in use to be considered a free gpu
    '''

    with popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used') as f:
        gpu_info = f.readlines()
    memory_available = [int(x.split()[2]) for x in gpu_info]
    free_gpus = [i for i, mem in enumerate(memory_available) if mem <= mem_threshold]

    return free_gpus

def save_model(checkpoint_dir, 
        model_filename, model, 
        scheduler_filename=None, scheduler=None,
        epoch_filename=None, epoch=None):

    '''
    Saves model checkpoint
    '''

    save(model.state_dict(), join(checkpoint_dir, model_filename))

    if (scheduler_filename is not None) and (scheduler is not None):
        save(scheduler.state_dict(), join(checkpoint_dir, scheduler_filename))

    if epoch is not None:
        save(epoch, join(checkpoint_dir, epoch_filename))

def load_model(checkpoint_dir, 
        model_filename, model,  
        scheduler_filename=None, scheduler=None,  
        epoch_filename=None):

    '''
    Loads model checkpoint
    '''

    if exists(join(checkpoint_dir, model_filename)):
        model.load_state_dict(load(join(checkpoint_dir, model_filename)))

    if exists(join(checkpoint_dir, scheduler_filename)):
        scheduler.load_state_dict(load(join(checkpoint_dir, scheduler_filename)))

    if exists(join(checkpoint_dir, epoch_filename)):
        last_epoch = load(join(checkpoint_dir, epoch_filename))
    else:
        last_epoch = -1

    return model, scheduler, last_epoch
