import numpy as np
from abc import abstractmethod
import torch
from torch_geometric.data import Dataset


def update_function(
    mgn_output_np,
    output_type,
    current_state=None,
    previous_state=None,
    source_data=None,
) -> np.ndarray:
    """creates the next state from the output of the MGN"""
    num_states = current_state.shape[-1]

    with torch.no_grad():
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
        # else: warning?

    return next_state


class MGNDataset(Dataset):
    def __init__(self, **kwargs):
        # if this subclasses Dataset, then we need to call super().__init__()
        """Create a dataset compatible with MGN."""
        if not hasattr(self, "update_function"):
            print("Adding default update function to Dataset")
            self.update_function = update_function

        super().__init__(**kwargs)

    @abstractmethod
    def get_non_source_data_mask(self, data) -> torch.Tensor:
        """Given a torch.Tensor, return a boolean torch.Tensor indicating the rows that
        are not source node rows.

        This mask is used to ensure we only compute the loss on nodes that aren't sources.

        The following example assumes the node type onehot is in columns #4:6, and the
        source node type is 1.

        source_node_type = 1
        node_type_onehot_start_idx = 4
        node_type_onehot_end_idx = 6

        if type(data) is list:
            # if using parallel, each sample in a batch is still represented as an element in a list. need to manually concat.
            one_hot_node_types = torch.cat([data_sample.x[:, node_type_onehot_start_idx:node_type_onehot_end_idx+1] for data_sample in data])
        else:
            # if not using parallel, all samples in a batch are concat'd together.
            one_hot_node_types = data.x[:, node_type_onehot_start_idx:node_type_onehot_end_idx+1]

        non_source_node_mask = one_hot_node_types.argmax(dim=1) != source_node_type

        return non_source_node_mask

        """

        pass

    @abstractmethod
    def __len__(self):
        """Return the number of graphs in the dataset.

        This is usually N_timesteps_per_simulation * N_simulations
        """
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return the graph at index `idx`.

        Preprocessing should go here. For example, adding noise to the graph data.
        """
        pass

    def len(self):
        r"""Returns the number of graphs stored in the dataset. One of the abstract methods for PyG 2.3 Dataset class"""
        pass

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`. One of the abstract methods for PyG 2.3 Dataset class"""
        pass
