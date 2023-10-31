from torch import tensor, cat, int64, is_tensor
from torch.nn.functional import one_hot
from numpy import ones, squeeze, diff, tile
from numpy.random import normal


def process_node_window(
    node_data,
    node_coordinates=None,
    node_types=None,
    apply_onehot=False,
    onehot_classes=0,
    inlet_velocity=None,
):

    """
    Concatenates with node features with one-hot encoding of node types

    node_data: time x node x feature array
    node_coordinates: node x dimension array; do not use if node_data already has preprocessed node coordinate information
    node_types: node x one-hot dim array; do not use if node_data already has preprocessed node type information
    apply_onehot: boolean
    onehot_classes: integer
    #inlet_velocity: float; we'll use it as a node feature for now, unless there's a better place to use it

    """
    num_nodes = node_data.shape[1]
    node_types_ = (
        one_hot(tensor(node_types, dtype=int64).flatten(), onehot_classes)
        if apply_onehot
        else node_types
    )
    if hasattr(node_types_, "device"):
        node_types_ = node_types_.to(node_data.device)
    node_data = [node_data.transpose(1, 0).reshape(num_nodes, -1)]

    if node_coordinates is not None:
        node_data.append(
            node_coordinates
            if is_tensor(node_coordinates)
            else tensor(node_coordinates)
        )

    if node_types_ is not None:
        node_data.append(node_types_ if is_tensor(node_types_) else tensor(node_types_))

    if inlet_velocity is not None:
        node_data.append(tensor(inlet_velocity * ones((num_nodes, 1))))

    if len(node_data) > 1:
        return cat(node_data, axis=-1)

    return node_data[0]


def get_sample(
    dataset,
    source_node_idx,
    time_idx,
    window_length=5,
    output_type="acceleration",
    noise_sd=None,
    noise_gamma=0.1,
    shift_source=False,
):

    """
    Returns position data window (with noise) and output velocity

    source_node_idx: source node indices
    time_idx: current time index
    window_length: input window length
    output_type: output type; one of 'state', 'velocity', or 'acceleration'
    noise_sd: noise standard deviation
    noise_gamma: noise gamma (see noise details in arXiv:2010.03409)
    shift_source: if True, shift input source nodes ahead by one timestep; noise is not included
    """

    mask = ones(dataset.shape[1], dtype=bool)
    mask[source_node_idx] = False

    node_data = dataset[time_idx : (time_idx + window_length), :, :].copy()

    # for sources, shift window ahead by 1 timepoint, do not add noise (masked below)
    # also need to update MeshGraphNets.py update_function and rollout functions (and loss?)
    # to do: add a config parameter to turn this off
    if shift_source:
        node_data[:, source_node_idx, :] = dataset[
            (time_idx + 1) : (time_idx + window_length + 1), source_node_idx, :
        ].copy()

    # compute output
    if output_type == "acceleration":
        outputs = dataset[
            (time_idx + window_length - 2) : (time_idx + window_length + 1), :, :
        ]
        outputs = squeeze(diff(diff(outputs, axis=0), axis=0), axis=0)
    elif output_type == "velocity":
        outputs = dataset[
            (time_idx + window_length - 1) : (time_idx + window_length + 1), :, :
        ]
        outputs = squeeze(diff(outputs, axis=0), axis=0)
    else:
        outputs = dataset[time_idx + window_length, :, :].copy()

    # add noise to position and output
    if noise_sd is not None:
        noise = tile(noise_sd, (node_data.shape[1], 1))
        noise = normal(0, noise)
        # input noise
        node_data[-1][mask] += noise[mask]
        # output adjustment

        if output_type == "acceleration":
            # acceleration_p = x_{t+1} - 2 * x_{t} + x_{t-1} - 2 * noise
            # acceleration_v = x_{t+1} - 2 * x_{t} + x_t{-1} - noise
            # adjustment = 2 * gamma * noise + (1-gamma) * noise = noise * (1 + gamma)
            outputs[mask] -= (1 + noise_gamma) * noise[mask]
        elif output_type == "velocity":
            # velocity_adj = x_{t+1} - (x_{t} + noise)
            outputs[mask] -= noise[mask]
        # else: nothing for state

    return node_data, outputs
