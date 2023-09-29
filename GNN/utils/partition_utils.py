from GNN.partitioning.grid_partitioner import GridPartitioner
from GNN.partitioning.null_partitioner import NullPartitioner
from GNN.partitioning.range_partitioner import RangePartitioner
from GNN.partitioning.modularity_partitioner import ModularityPartitioner


def get_partitioner(params_dict):
    """
    returns a Partitioner object based on the parameters specified in params_dict

    params_dict is a dictionary. It must contain the name of a partition method,
    keyed by "partition_method", and the relevant parameters for that method.

    See the partitioner classes in GNN.partitioning for expected params
    and the config files under config_files for examples
    """
    partitioning_method = params_dict["partitioning_method"]
    if partitioning_method == "grid":
        return GridPartitioner(
            nrows=params_dict["nrows"],
            ncols=params_dict["ncols"],
            padding=params_dict["padding"],
        )
    elif partitioning_method == "range":
        range_keys = [key for key in params_dict.keys() if key.startswith("range_")]
        range_keys = sorted(range_keys, lambda x: int(x.split("_")[1]))
        ranges = [params_dict[key] for key in range_keys]
        return RangePartitioner(ranges=ranges, padding=params_dict["padding"])
    elif partitioning_method == "modularity":
        return ModularityPartitioner(padding=params_dict["padding"])
    elif partitioning_method == "null":
        return NullPartitioner()
    else:
        raise ValueError("Unrecognized partitioning method: %s" % partitioning_method)
