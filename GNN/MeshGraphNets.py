from numpy import array, zeros
from torch import cat, no_grad, tensor, stack
from torch.nn.functional import pad
from torch_geometric.data import Data
from time import time
from tqdm import tqdm
from functools import partial

from .utils.mgn_utils import process_node_window
from .GNN import GNN
from .utils.Normalizer import Normalizer


class MeshGraphNets(GNN):
    def __init__(
        self,
        # data attributes:
        in_dim_node,  # includes data window, node type, inlet velocity
        in_dim_edge,  # distance and relative coordinates
        out_dim,  # includes x-velocity, y-velocity, volume fraction, pressure (or a subset)
        # encoding attributes:
        out_dim_node=128,
        out_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        node_normalizer_mask=None,
        edge_normalizer_mask=None,
        # graph processor attributes:
        mp_iterations=15,
        hidden_dim_processor_node=128,
        hidden_dim_processor_edge=128,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
        # decoder attributes:
        hidden_dim_decoder=128,
        hidden_layers_decoder=2,
        output_type="acceleration",
        output_normalizer_mask=None,
        # adaptive mesh
        use_adaptive_mesh=False,  # not implemented yet
        normalize_output=True,
        connection_type="FullResidualConnection",
        connection_alpha=0.5,
        connection_aggregation="concat",
        graph_processor_type="Original",
        integrator="euler",
        **kwargs,
    ):

        """
        Original MeshGraphNets model (arXiv:2010.03409), with additional enhancements; default values are based on the paper/supplement

        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        out_dim: output dimension
        out_dim_node: encoded node feature dimension
        out_dim_edge: encoded edge feature dimension
        hidden_dim_node: node encoder MLP dimension
        hidden_dim_edge: edge encoder MLP dimension
        hidden_layers_node: number of node encoder MLP layers
        hidden_layers_edge: number of edge encoder MLP layers
        node_normalizer_mask: boolean mask 0=normalize, 1=do not normalize
        edge_normalizer_mask: boolean mask 0=normalize, 1=do not normalize
        mp_iterations: number of message passing iterations
        hidden_dim_processor_node: MGN node processor MLP dimension
        hidden_dim_processor_edge: MGN edge processor MLP dimension
        hidden_layers_processor_node: number of MGN node processor MLP layers
        hidden_layers_processor_edge: number of MGN edge processor MLP layers
        mlp_norm_type: MLP normalization type ('LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm')
        hidden_dim_decoder: decoder MLP dimension
        hidden_layers_decoder: decoder MLP layers
        output_type: output type ('state', 'velocity', 'acceleration')
        output_normalizer_mask: boolean mask 0=normalize, 1=do not normalize
        use_adaptive_mesh: if True, use adaptive; not functional yet
        normalize_output: if True, normalize the output
        connection_type: one of [FullResidualConnection, ResidualConnection, InitialConnection, DenseConnection]
        connection_alpha: float for Residual & Initial connection types only, can be in range [0, 1]
        connection_aggregation: for dense connection types only, can be one of [concat, maxpool, attention]
        integrator: how the change to the input is calculated
        """

        super(MeshGraphNets, self).__init__(
            in_dim_node,
            in_dim_edge,
            out_dim,
            out_dim_node,
            out_dim_edge,
            hidden_dim_node,
            hidden_dim_edge,
            hidden_layers_node,
            hidden_layers_edge,
            mp_iterations,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
            hidden_dim_decoder,
            hidden_layers_decoder,
            output_type,
            connection_type,
            connection_alpha,
            connection_aggregation,
            graph_processor_type,
            **kwargs,
        )

        self.in_dim_node = in_dim_node
        self.in_dim_edge = in_dim_edge
        self.out_dim = out_dim

        self.use_adaptive_mesh = use_adaptive_mesh

        self._node_normalizer = Normalizer(
            size=in_dim_node, ignore_mask=node_normalizer_mask, name="node_normalizer"
        )
        self._edge_normalizer = Normalizer(
            size=in_dim_edge, ignore_mask=edge_normalizer_mask, name="edge_normalizer"
        )
        self.normalize_output = normalize_output
        if normalize_output:
            self._output_normalizer = Normalizer(
                size=out_dim,
                ignore_mask=output_normalizer_mask,
                name="output_normalizer",
            )

        self.full_grad = "euler" in integrator or "_fg" in integrator
        integrator = integrator.lower().replace("_fg", "")
        integrators = {
            "euler": self._forward,
            "heun": self.heun_forward,
            "midpoint": self.midpoint_forward,
            "heun_third": partial(
                self.butcher_tableau_forward, tableau_builder("heun_third")
            ),
            "ssprk3": partial(self.butcher_tableau_forward, tableau_builder("ssprk3")),
        }
        assert integrator in integrators, (integrator.lower(), integrators.keys())
        self.forward = integrators[integrator]
        print(
            f'using {integrator} integrator, full gradient is set to {self.full_grad or integrator in ["heun_third","ssprk3"]}'
        )

        # if self.use_adaptive_mesh:
        #     pass

    def _forward(self, graph, force_accum_off=False):
        accumulate = self.training and not force_accum_off

        if type(graph) is tuple:
            x, edge_index, edge_attr = graph
        else:
            x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

        # encode node/edge features
        out = self.node_encoder(self._node_normalizer(x, accumulate=accumulate))
        edge_attr = self.edge_encoder(
            self._edge_normalizer(edge_attr, accumulate=accumulate)
        )

        # if self.use_adaptive_mesh:
        #     pass

        # message passing
        out, _ = self.graph_processor(out, edge_index, edge_attr)

        # decode
        out = self.node_decoder(out)

        return out

    def midpoint_forward(self, graph):
        """
        WARNING: assumes a "velocity" (aka derivative) update...
        I.e., the MGN's forward must be f=dy/dt.

        Given delta_t (aka "h") = 1,  this method returns f_2, where:
            y_{t+1} = y_t + f_2,
            f_2 = f(y_t + 0.5 * f_1),
            f_1 = f(y_t)
        """
        if not self.full_grad:
            ### low-memory, in-place graph.x update that may provide less learning signal ###
            # get f_1 = dy/dt
            with no_grad():
                f_1 = self._forward(graph)

            # to get f_2, first update y_t with f_1
            if self.normalize_output:  # f_1 = normalized dy/dt
                graph.x[..., : self.out_dim] += 0.5 * self._output_normalizer.inverse(
                    f_1
                )
            else:  # f_1 = dy/dt
                graph.x[..., : self.out_dim] += 0.5 * f_1
            # compute f_2
            f_2 = self._forward(graph, force_accum_off=True)
        else:
            ### high-memory version ###
            f_1 = self._forward(graph)

            # to get f_2, first update y_t with f_1
            other_dim = graph.x.shape[-1] - self.out_dim
            if self.normalize_output:  # f_1 = normalized dy/dt
                f_2 = self._forward(
                    (
                        graph.x + 0.5 * self.PIN(f_1, other_dim),
                        graph.edge_index,
                        graph.edge_attr,
                    ),
                    force_accum_off=True,
                )
            else:  # f_1 = dy/dt
                f_2 = self._forward(
                    (
                        graph.x + 0.5 * pad(f_1, (0, other_dim)),
                        graph.edge_index,
                        graph.edge_attr,
                    ),
                    force_accum_off=True,
                )

        return f_2

    def heun_forward(self, graph):
        """
        WARNING: assumes a "velocity" (aka derivative) update...
        I.e., the MGN's forward must be f=dy/dt.

        Given delta_t (aka "h") = 1,  this method returns f_3, where:
            y_{t+1} = y_t + f_3,
            f_3 = (f_1 + f_2)/2,
            f_2 = f(y_t + f_1),
            f_1 = f(y_t)
        """
        if not self.full_grad:
            ### low-memory, in-place graph.x update that may provide less learning signal ###
            # get f_1 = dy/dt
            with no_grad():
                f_1 = self._forward(graph)

            # to get f_2, first update y_t with f_1
            if self.normalize_output:  # f_1 = normalized dy/dt
                graph.x[..., : self.out_dim] += self._output_normalizer.inverse(f_1)
            else:  # f_1 = dy/dt
                graph.x[..., : self.out_dim] += f_1
            # compute f_2
            f_2 = self._forward(graph, force_accum_off=True)
        else:
            ### high-memory version ###
            f_1 = self._forward(graph)

            # to get f_2, first update y_t with f_1
            other_dim = graph.x.shape[-1] - self.out_dim
            if self.normalize_output:  # f_1 = normalized dy/dt
                f_2 = self._forward(
                    (
                        graph.x + self.PIN(f_1, other_dim),
                        graph.edge_index,
                        graph.edge_attr,
                    ),
                    force_accum_off=True,
                )
            else:  # f_1 = dy/dt
                f_2 = self._forward(
                    (
                        graph.x + pad(f_1, (0, other_dim)),
                        graph.edge_index,
                        graph.edge_attr,
                    ),
                    force_accum_off=True,
                )

        return (f_1 + f_2) / 2

    def butcher_tableau_forward(self, params, graph):
        """
        WARNING: assumes a "velocity" (aka derivative) update...
        I.e., the MGN's forward must be f=dy/dt.

        See the following for more information:
        https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/numerics/solvers/ode.py
        """
        a, bsol = params

        # k_i = normalized dy/dt
        assert (
            self.normalize_output
        ), "MeshGraphNets.butcher_tableau_forward only implemented for normalize_output=True"
        pad = graph.x.shape[-1] - self.out_dim

        k1 = self._forward(graph)

        k2 = (
            self._forward(
                (
                    graph.x + (a[0][0] * self.PIN(k1, pad) if a[0][0] else 0),
                    graph.edge_index,
                    graph.edge_attr,
                ),
                force_accum_off=True,
            )
            if sum(abs(bsol[1:]))
            else 0
        )

        k3 = (
            self._forward(
                (
                    graph.x
                    + (
                        a[1][0] * (self.PIN(k1, pad) if a[1][0] else 0)
                        + a[1][1] * (self.PIN(k2, pad) if a[1][1] else 0)
                    ),
                    graph.edge_index,
                    graph.edge_attr,
                ),
                force_accum_off=True,
            )
            if sum(abs(bsol[2:]))
            else 0
        )

        k4 = (
            self._forward(
                (
                    graph.x
                    + (
                        a[2][0] * (self.PIN(k1, pad) if a[2][0] else 0)
                        + a[2][1] * (self.PIN(k2, pad) if a[2][1] else 0)
                        + a[2][2] * (self.PIN(k3, pad) if a[2][2] else 0)
                    ),
                    graph.edge_index,
                    graph.edge_attr,
                ),
                force_accum_off=True,
            )
            if sum(abs(bsol[3:]))
            else 0
        )

        return bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4

    def PIN(self, k, other_dim):
        """
        padding and inverse normalization of the input to get dy/dt with zero padding at location of constant vars
        """
        return pad(self._output_normalizer.inverse(k), (0, other_dim))

    # Default update/rollout functions based on state, velocity, or acceleration
    # For customized updates, extend this class and override this function
    # Alternatively (preferred), write a new update_function as part within the dataset class that is used
    # (see cylinder flow dataset example)
    def update_function(
        self, mgn_output_np, current_state=None, previous_state=None, source_data=None
    ):
        """
        Default state update function;
        Extend and override this function, or add as a dataset class attribute

        mgn_output_np: MGN output
        current_state: Current state
        previous_state: Previous state (for acceleration-based updates)
        source_data: Source/scripted node data
        """

        with no_grad():
            if self.output_type == "acceleration":
                assert current_state is not None
                assert previous_state is not None
                next_state = 2 * current_state - previous_state + mgn_output_np
            elif (
                self.output_type == "velocity"
            ):  # or self.output_type == 'velocity_liquidfraction':
                assert current_state is not None
                # the default approach assumes all variables are updated using the same type
                # and the mgn output dimension is equal to the number of state variables
                next_state = current_state + mgn_output_np
            elif self.output_type == "state":
                next_state = mgn_output_np.copy()
            else:
                # add this in case self.output_type is misconfigured
                raise NotImplementedError(self.output_type)

            # Scripting
            if type(source_data) is dict:
                for key in source_data:
                    next_state[key] = source_data[key]
            elif type(source_data) is tuple:
                next_state[source_data[0]] = source_data[1]
            else:
                raise NotImplementedError()
            # else: warning?

        return next_state

    def rollout_step(
        self,
        device,
        input_graph,
        current_window_data,
        node_types,
        node_coordinates,
        onehot_info=None,
        inlet_velocity=None,
        source_data=None,
        update_function=None,  # if None, use the default update_function (above; uses same variables for input and output)
        shift_source=False,
        update_func_kwargs={},
    ):

        assert (self.output_type != "acceleration") or (
            current_window_data.shape[-1] >= 2
        )

        self.eval()
        update_func = (
            self.update_function if update_function is None else update_function
        )

        with no_grad():

            # process/reshape data
            node_data = process_node_window(
                current_window_data,
                node_coordinates,
                node_types,
                apply_onehot=onehot_info[0],
                onehot_classes=onehot_info[1],
                inlet_velocity=inlet_velocity,
            )

            # apply MGN model
            node_data = node_data.squeeze()
            input_graph.x = node_data
            if self.normalize_output:
                mgn_output = self._output_normalizer.inverse(self.forward(input_graph))
            else:
                mgn_output = self.forward(input_graph)

            # if self.use_adaptive_mesh:
            #     pass

            # update state
            next_state = update_func(
                mgn_output_np=mgn_output,
                current_state=current_window_data[-1],
                previous_state=current_window_data[-2]
                if len(current_window_data) > 1
                else None,
                source_data=source_data,
                **update_func_kwargs,
            )

        return next_state

    def rollout(
        self,
        device,
        initial_window_data,
        graph,
        node_types,
        node_coordinates,
        onehot_info=None,
        inlet_velocity=None,
        rollout_iterations=1,
        source_data=None,
        update_function=None,  # if None, use the default update_function (above; uses same variables for input and output)
        shift_source=False,
        progressbar=True,
        update_func_kwargs={},
    ):
        """
        graph: torch_geometric.data.Data object with the following attributes (see PNNLGraphDataset.py for graph construction):
               x: node x feature array (volume fraction, pressure, node type, inlet velocity, etc.)
               edge_index: 2 x edge array
               edge_attr: edge x feature matrix (distance, relative coordinates)
        initial_window_data: window x nodes x features array
        node_types: node x one-hot dim array
        node_coordinates: node x dimension array
        inlet_velocity: float
        rollout_iterations: int; number of timepoints to simulate
        source_data: dict({node id: time x features np array}) or tuple (node ids, time x node x feature np array); simulated nodes/boundary conditions
        shift_source: if True, replace source/known data at time t before updating rest of t-1 data to time t
        show_progressbar: whether to show a progress bar (True) or not. Default is True
        """

        current_window_data = tensor(initial_window_data, device=device)
        input_graph = Data(edge_index=graph.edge_index, edge_attr=graph.edge_attr).to(
            device
        )
        input_len = self.node_encoder.model[0].in_features

        rollout_output = []

        iter_obj = range(rollout_iterations)
        if progressbar:
            iter_obj = tqdm(iter_obj)
        start = time()
        for i in iter_obj:

            # TODO: source data is large and requires movement to GPU. Replace with a function for source node values?
            if type(source_data) is dict:
                source_data_i = cat(
                    [source_data[id][i, :] for id in source_data.keys()]
                )
                source_data_i = (list(source_data.keys()), source_data_i)
            elif type(source_data) is tuple:
                source_data_i = (
                    source_data[0],
                    tensor(source_data[1][i, :, :], device=device),
                )
                source_node_mask, source_node_data = source_data_i
                # source_node_mask: [num_nodes]
                # source_node_data: [time, num_nodes, features]
            else:
                source_data_i = None

            if (source_data_i is not None) and shift_source:
                current_window_data[:-1, source_data_i[0]] = current_window_data[
                    1:, source_data_i[0]
                ]
                current_window_data[-1, source_data_i[0]] = source_data_i[1]

            next_state = self.rollout_step(
                device=device,
                input_graph=input_graph,
                current_window_data=current_window_data,
                node_types=node_types,
                node_coordinates=node_coordinates,
                onehot_info=onehot_info,
                inlet_velocity=inlet_velocity,
                source_data=source_data_i,
                update_function=update_function,
                shift_source=shift_source,
                update_func_kwargs=update_func_kwargs,
            )

            current_window_data = cat(
                [current_window_data[1:], next_state[:, :input_len].unsqueeze(0)], dim=0
            )

            rollout_output.append(next_state.cpu())

        seconds = time() - start
        print(f"\n***********\nRollout took {seconds} seconds!\n************\n")
        return array(stack(rollout_output))


def tableau_builder(method):
    base_tableau_a = zeros((3, 3))
    base_tableau_b = zeros(4)
    # row 0,1,2 is used to compute k2,k3,k4, respectively.
    # column 0,1,2 gives constants for k1,k2,k3, respectively.
    if method == "heun_third":
        base_tableau_a[0][0] = 1 / 3
        base_tableau_a[1][1] = 2 / 3
        base_tableau_b[0] = 1 / 4
        base_tableau_b[2] = 3 / 4
    elif method == "ssprk3":
        base_tableau_a[0][0] = 1
        base_tableau_a[1][:2] = 1 / 4
        base_tableau_b[0:2] = 1 / 6
        base_tableau_b[2] = 2 / 3
    else:
        assert False, f"{method} not implemented"
    return base_tableau_a, base_tableau_b
