import os

# Set World Size Key
if "WORLD_SIZE" in os.environ:
    WORLD_SIZE_KEY = "WORLD_SIZE"

elif "OMPI_COMM_WORLD_SIZE" in os.environ:
    WORLD_SIZE_KEY = "OMPI_COMM_WORLD_SIZE"

# Set World Rank Key
if "SLURM_PROCID" in os.environ:
    WORLD_RANK_KEY = "SLURM_PROCID"

elif "OMPI_COMM_WORLD_RANK" in os.environ:
    WORLD_RANK_KEY = "OMPI_COMM_WORLD_RANK"

# Set Keys for Master Address, Master Port, and Num Processes Per Node
MASTER_ADDR_KEY = "MASTER_ADDR"
MASTER_PORT_KEY = "MASTER_PORT"
NUM_PROCESSES_PER_NODE_KEY = "NUM_PROCESS_PER_NODE"

# ----- SLURM -----
# WORLD_SIZE_KEY = 'WORLD_SIZE'
# WORLD_RANK_KEY = 'SLURM_PROCID'
# MASTER_ADDR_KEY = 'MASTER_ADDR'
# MASTER_PORT_KEY = 'MASTER_PORT'
# NUM_PROCESSES_PER_NODE_KEY = 'NUM_PROCESS_PER_NODE'

# ----- LSF -----
# WORLD_SIZE_KEY = 'OMPI_COMM_WORLD_SIZE'
# WORLD_RANK_KEY = 'OMPI_COMM_WORLD_RANK'
# LOCAL_RANK_KEY = 'OMPI_COMM_LOCAL_RANK'
# MASTER_ADDR_KEY = 'MASTER_ADDR'
# MASTER_PORT_KEY = 'MASTER_PORT'
# NUM_PROCESSES_PER_NODE_KEY = 'NUM_PROCESS_PER_NODE'


class CustomLSFEnvironment:
    """
    An environment for running on clusters managed by the LSF resource manager.
    """

    def __init__(self):
        self.world_size = int(os.environ[WORLD_SIZE_KEY])
        self.rank = int(os.environ[WORLD_RANK_KEY])
        self.num_processes_per_node = int(os.environ[NUM_PROCESSES_PER_NODE_KEY])
        self.local_rank = self.rank % self.num_processes_per_node


class CustomManualEnvironment:
    """
    The classic style of setting up PyTorch's DDP (Instead of relying on LSF's jsrun or SLURM's srun to start multiple processes that call a python script, use pytorch's "mp.spawn" in Python directly!)
    This class exists to re-use code that depends on CustomLSFEnvironment
    """

    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = rank
