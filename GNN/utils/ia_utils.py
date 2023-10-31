import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
import re, gc
from torch import cuda
from GNN.utils.rollout import main as rollout_function
from copy import copy
from scipy.interpolate import griddata


def interpData(x, y, z, Nx=None, Ny=None, method="linear"):
    """
    This function takes 3 lists of points (x,y,z) and maps them to a
    rectangular grid. Either Nx or Ny must be set or delta_x must be set.
    e.g.

    x = y = z = np.random.rand(30)
    grid_x, grid_y, grid_z = interpData(x,y,z,Nx=128,Ny=128)
    """

    eps = 1e-4  # needed to make sure that the interpolation does not have nans.
    grid_x, grid_y = np.mgrid[
        x.min() + eps : x.max() - eps : Nx * 1j, y.min() + eps : y.max() - eps : Ny * 1j
    ]
    grid_z = griddata(np.array([x, y]).T, z, (grid_x, grid_y), method=method)
    return grid_x, grid_y, grid_z


class Metrics:
    """
    Defines relative IA and volume-fraction-field errors in terms of
    ground-truth and predicted rollouts and mesh coordinates (user inputs).

    Other metrics can be added later, e.g. RMSE.
    """

    def __init__(
        self,
        rollout_pred=None,
        rollout_act=None,
        rollout_coords=None,
        vol_frac_col=2,
        last_frame_only=True,
        interpolation_sizes=None,
    ):
        self.rollout_pred = rollout_pred
        if type(rollout_pred) == str:
            self.rollout_pred = pickle.load(open(rollout_pred, "rb"))["pred"]
        self.rollout_act = rollout_act
        if type(rollout_act) == str:
            self.rollout_act = pickle.load(open(rollout_act, "rb"))["pred"]
        self.rollout_coords = rollout_coords
        if type(rollout_coords) == str:
            self.rollout_coords = pickle.load(open(rollout_coords, "rb"))["coords"]
        self.vol_frac_col = vol_frac_col
        self.last_frame_only = last_frame_only
        if interpolation_sizes:
            print(
                f"\nInterpolating. Original mesh shape was {self.rollout_coords.shape}."
            )
            assert last_frame_only
            x = self.rollout_coords[:, 0]
            y = self.rollout_coords[:, 1]
            z = self.rollout_pred[-1, :, self.vol_frac_col]
            grid_x, grid_y, grid_z = interpData(
                x, y, z, Nx=interpolation_sizes[0], Ny=interpolation_sizes[0]
            )
            self.rollout_pred = grid_z.reshape(1, -1, 1)
            self.rollout_coords = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
            print(
                f"\nNew shape is {self.rollout_coords.shape}. Now {grid_x.shape} x {grid_y.shape}. Vol frac data has shape [timesteps, nodes, x*y] = {self.rollout_pred.shape}."
            )

            z = self.rollout_act[-1, :, self.vol_frac_col]
            grid_x, grid_y, grid_z = interpData(
                x, y, z, Nx=interpolation_sizes[0], Ny=interpolation_sizes[0]
            )
            self.rollout_act = grid_z.reshape(1, -1, 1)
            self.vol_frac_col = 0  # we only interpolated the vol_frac_col, so now this is 0 to grab the first and only col

        assert (
            len(self.rollout_pred.shape) == 3
        ), "expected [num timesteps, num nodes, num features]"
        assert (
            len(self.rollout_act.shape) == 3
        ), "expected [num timesteps, num nodes, num features]"
        assert np.all(
            self.rollout_pred.shape == self.rollout_act.shape
        ), "pred and actual shapes must match"

    @property
    def IA_rel_error(self):
        return self.get_IA_data(
            self.rollout_pred,
            self.rollout_act,
            self.rollout_coords,
            self.vol_frac_col,
            self.last_frame_only,
        )

    @property
    def VF_rel_error(self):
        return self.get_VF_data(
            self.rollout_pred, self.rollout_act, self.vol_frac_col, self.last_frame_only
        )

    @staticmethod
    def get_IA_data(
        rollout_pred, rollout_act, rollout_coords, vol_frac_col, last_frame_only=True
    ):
        if last_frame_only:
            IA_pred = computeInterfacialArea2D(
                rollout_coords, rollout_pred[-1][:, vol_frac_col]
            )
            IA_act = computeInterfacialArea2D(
                rollout_coords, rollout_act[-1][:, vol_frac_col]
            )
        else:
            IA_pred = np.array(
                [
                    computeInterfacialArea2D(
                        rollout_coords, rollout_pred[i][:, vol_frac_col]
                    )
                    for i in range(len(rollout_pred))
                ]
            )
            IA_act = np.array(
                [
                    computeInterfacialArea2D(
                        rollout_coords, rollout_act[i][:, vol_frac_col]
                    )
                    for i in range(len(rollout_pred))
                ]
            )
        return {
            "IA_error": np.mean(np.abs(IA_pred - IA_act) / IA_act),
            "IA_pred": IA_pred,
            "IA_act": IA_act,
        }

    @staticmethod
    def get_VF_data(rollout_pred, rollout_act, vol_frac_col, last_frame_only=True):
        VF_pred = rollout_pred[:, :, vol_frac_col]
        VF_act = rollout_act[:, :, vol_frac_col]
        if last_frame_only:
            VF_pred = VF_pred[-1:]
            VF_act = VF_act[-1:]
        return {
            "VF_error": np.mean(
                np.linalg.norm(VF_pred - VF_act, axis=1)
                / np.linalg.norm(VF_act, axis=1)
            ),
            "VF_pred": VF_pred,
            "VF_act": VF_act,
        }


class RolloutArgs:
    def __init__(
        self,
        config_path,
        cpu=False,
        test_set=True,
        rollout_num=0,
        GT=False,
        viz_rollout_num=None,
    ):
        self.config_file = config_path
        self.cpu = cpu
        self.test_set = test_set
        self.rollout_num = rollout_num
        self.GT = GT
        self.viz_rollout_num = viz_rollout_num


class AnalyzeRollouts:
    """
    Given a directory of existing rollouts or an MGN config, facilitates
    acquisition of metrics for specific rollouts.

    When they have not yet been run, runs rollouts.

    ...

    USAGE
    ----------
    there are two options for instantiating this class:
        option (1) with `dir` containing rollouts.
        option (2) with `rollout_args`, an object of type `RolloutArgs`. here,
            `__init__` runs a rollout if needed and `dir` will be set to the
            location of the rollouts (defined in the config of `rollout_args`).

    once instantiated, run `metrics = get_sim_metrics(k)`, where `k` is the
    index of the simulation that metrics are desired for... subsequently,
    `IA(metrics)` or `VF(metrics)` can be called.
    """

    def __init__(
        self,
        dir=None,  # e.g., '/data/ccsi/logs/test/pnnl_extrap_interp'
        rollout_args=None,
        vol_frac_col=2,
        last_frame_only=True,
        test_set=True,
        interpolation_sizes=None,
    ):
        """
        dir: directory of rollouts that have been run
        rollout_args: set of arguments for a specific rollout that
                        that will be run if needed
        vol_frac_col: the column of the num_node x num_features matrix
                            corresponding to the volume fraction
        last_frame_only: compute metrics using only the last timestep
        """
        assert not (
            rollout_args and dir
        ), "Give a dir with existing rollouts or a set of args to run rollouts, not both"
        self.dir = dir
        self.rollout_args = rollout_args
        if not dir:
            assert (
                self.rollout_args.test_set == test_set
            ), f"test_set value {test_set} and rollout_args test_set value {self.rollout_args.test_set} do not match"
            self.dir = self.run_rollout()
        self.vol_frac_col = vol_frac_col
        self.last_frame_only = last_frame_only
        self.test_set = test_set
        self.interpolation_sizes = interpolation_sizes

    def run_rollout(self):
        if type(self.rollout_args.rollout_num) is not list:
            return rollout_function(self.rollout_args)
        else:
            for num in self.rollout_args.rollout_num:
                args = copy(self.rollout_args)
                args.rollout_num = num
                dir = rollout_function(args)
                gc.collect()
                cuda.empty_cache()
            return dir

    def get_dir_metrics(self, IA=True):
        dataset_str = "test" if self.test_set else "train"
        dir_metrics = {}
        sim_nums = []
        for f in glob(self.dir + f"/rollout_{dataset_str}*"):
            sim_nums.append(int(re.findall(r"\d+", f)[0]))
        for sim in sim_nums:
            dir_metrics[sim] = (
                self.IA(self.get_sim_metrics(sim))
                if IA
                else self.VF(self.get_sim_metrics(sim))
            )
        return dir_metrics

    def get_sim_metrics(self, num):
        return Metrics(
            self.file_name(num),
            self.file_name(num, gt=True),
            self.file_name(num),
            self.vol_frac_col,
            self.last_frame_only,
            self.interpolation_sizes,
        )

    @staticmethod
    def IA(metrics):
        return metrics.IA_rel_error

    @staticmethod
    def VF(metrics):
        return metrics.VF_rel_error

    def file_name(self, num, gt=False):
        dataset_str = "test" if self.test_set else "train"
        return f'{self.dir}/rollout_{dataset_str}_{num}{"_GT" if gt else ""}.pk'


def sumLineSegments(p):
    d = p[1:] - p[:-1]
    d = np.sum(np.sqrt(np.sum(d**2, axis=1)))
    return d


def computeInterfacialArea2D(coordinates, values, level=0.5):
    """
    Computes interfacial area for 2D data.

    coordinates: Either (1) a 2-element list corresponding to the output of np.meshgrid/mgrid or
                (2) n x 2 numpy array of coordinates
    values: measurement values at grid/point locations
    level: contour level to compute IA
    """

    if type(coordinates) is list:  # output of np.meshgrid/mgrid
        contour = plt.contour(coordinates[0], coordinates[1], values, levels=[level])
    elif type(coordinates) is np.ndarray:  # point cloud
        contour = plt.tricontour(
            coordinates[:, 0], coordinates[:, 1], values, levels=[level]
        )
    else:
        raise Exception("Unsupported coordinate type")

    # contour = plt.contour(grid_x, grid_y, values, levels=[level])
    IA = np.sum(list(map(lambda x: sumLineSegments(x), contour.allsegs[0])))
    plt.close()

    return IA
