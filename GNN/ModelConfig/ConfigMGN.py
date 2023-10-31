from configparser import ConfigParser, ExtendedInterpolation
from ast import literal_eval
from os import makedirs
from os.path import join
import datetime as dt


class ModelConfig:
    def __init__(self, config_file):

        self.config_file = config_file

        # parse config file
        config = ConfigParser(
            inline_comment_prefixes="#",
            allow_no_value=True,
            interpolation=ExtendedInterpolation(),
        )
        config.optionxform = str  # preserve case of keys
        config.read(config_file)

        # --------------- data-related parameters ---------------
        c_data = config["DATA"]

        # dataset class
        self.class_name = c_data["class_name"]
        self.name = "untitled" if "name" not in c_data else c_data["name"]

        # graph construction parameters, if a graph is not provided for certain dataset classes
        self.normal_node_types = (
            [0]
            if "normal_node_types" not in c_data
            else literal_eval(c_data["normal_node_types"])
        )
        self.boundary_node_types = (
            [1]
            if "boundary_node_types" not in c_data
            else literal_eval(c_data["boundary_node_types"])
        )
        self.source_node_types = (
            [2]
            if "source_node_types" not in c_data
            else literal_eval(c_data["source_node_types"])
        )
        if "num_node_types" not in c_data:
            self.num_node_types = (
                len(self.normal_node_types)
                + len(self.boundary_node_types)
                + len(self.source_node_types)
            )
        else:
            self.num_node_types = c_data.getint("num_node_types")
        self.graph_type = (
            "radius" if "graph_type" not in c_data else c_data["graph_type"]
        )
        self.graph_types = (
            [self.graph_type]
            if "graph_types" not in c_data
            else literal_eval(c_data["graph_types"])
        )
        self.k = 10 if "k" not in c_data else c_data.getint("k")
        self.radius = 0.01 if "radius" not in c_data else c_data.getfloat("radius")

        # processing params
        self.output_type = (
            "state" if "output_type" not in c_data else c_data["output_type"]
        )
        self.window_length = (
            5 if "window_length" not in c_data else c_data.getint("window_length")
        )
        self.apply_onehot = (
            False if "apply_onehot" not in c_data else c_data.getboolean("apply_onehot")
        )
        self.apply_partitioning = (
            False
            if "apply_partitioning" not in c_data
            else c_data.getboolean("apply_partitioning")
        )
        self.noise = None if "noise" not in c_data else literal_eval(c_data["noise"])
        self.noise_gamma = (
            0.1 if "noise_gamma" not in c_data else c_data.getfloat("noise_gamma")
        )
        self.normalize = (
            True if "normalize" not in c_data else c_data.getboolean("normalize")
        )
        self.distributed_sampler_bounds = (
            None
            if "distributed_sampler_bounds" not in c_data
            else literal_eval(c_data["distributed_sampler_bounds"])
        )

        # --------------- MGN-related parameters ---------------

        m_data = config["MODEL"]
        self.model_arch = (
            "MeshGraphNets" if "model_arch" not in m_data else m_data["model_arch"]
        )
        self.graph_processor_type = (
            "Original"
            if "graph_processor_type" not in m_data
            else m_data["graph_processor_type"]
        )
        self.mgn_dim = 128 if "mgn_dim" not in m_data else m_data.getint("mgn_dim")
        self.mp_iterations = (
            15 if "mp_iterations" not in m_data else m_data.getint("mp_iterations")
        )
        self.mlp_norm_type = (
            "LayerNorm" if "mlp_norm_type" not in m_data else m_data["mlp_norm_type"]
        )
        self.mlp_norm_type = (
            None if self.mlp_norm_type == "None" else self.mlp_norm_type
        )
        self.normalize_output = (
            True
            if "normalize_output" not in m_data
            else m_data.getboolean("normalize_output")
        )

        self.connection_type = (
            "FullResidualConnection"
            if "connection_type" not in m_data
            else m_data["connection_type"]
        )
        self.connection_alpha = (
            0.5
            if "connection_alpha" not in m_data
            else m_data.getfloat("connection_alpha")
        )
        self.connection_aggregation = (
            "concat"
            if "connection_aggregation" not in m_data
            else m_data["connection_aggregation"]
        )
        self.integrator = (
            "euler" if "integrator" not in m_data else m_data["integrator"]
        )

        # --------------- training-related parameters ---------------

        t_data = config["TRAINING"]

        self.epochs = 100 if "epochs" not in t_data else int(t_data.getfloat("epochs"))
        self.steps = None if "steps" not in t_data else int(t_data.getfloat("steps"))
        self.batch_size = (
            8 if "batch_size" not in t_data else t_data.getint("batch_size")
        )
        self.grad_accum_steps = (
            1 if "grad_accum_steps" not in t_data else t_data.getint("grad_accum_steps")
        )
        self.lr = 5e-3 if "lr" not in t_data else t_data.getfloat("lr")
        self.weight_decay = (
            0 if "weight_decay" not in t_data else t_data.getfloat("weight_decay")
        )
        self.scheduler = "ExpLR" if "scheduler" not in t_data else t_data["scheduler"]
        self.use_parallel = (
            False if "use_parallel" not in t_data else t_data.getboolean("use_parallel")
        )
        self.use_amp = (
            False if "use_amp" not in t_data else t_data.getboolean("use_amp")
        )

        self.use_sbp = (
            False if "use_sbp" not in t_data else t_data.getboolean("use_sbp")
        )
        self.sbp_start_epoch = (
            1 if "sbp_start_epoch" not in t_data else t_data.getint("sbp_start_epoch")
        )
        self.sbp_rate = 3 if "sbp_rate" not in t_data else t_data.getint("sbp_rate")
        self.sbp_percent = (
            0.25 if "sbp_percent" not in t_data else t_data.getfloat("sbp_percent")
        )
        self.sbp_randomness = (
            0.05
            if "sbp_randomness" not in t_data
            else t_data.getfloat("sbp_randomness")
        )

        self.load_prefix = (
            "last_epoch" if "load_prefix" not in t_data else t_data["load_prefix"]
        )

        self.log_rate = 100 if "log_rate" not in t_data else t_data.getint("log_rate")
        self.use_tensorboard = (
            False
            if "use_tensorboard" not in t_data
            else t_data.getboolean("use_tensorboard")
        )
        self.tb_rate = 10 if "tb_rate" not in t_data else t_data.getint("tb_rate")
        self.train_time_limit = (
            1
            if "train_time_limit" not in t_data
            else t_data.getfloat("train_time_limit")
        )
        self.validation_time_limit = (
            1 / 20
            if "validation_time_limit" not in t_data
            else t_data.getfloat("validation_time_limit")
        )
        self.sample_limit = (
            1e12 if "sample_limit" not in t_data else t_data.getint("sample_limit")
        )

        self.expt_name = None if "expt_name" not in t_data else t_data["expt_name"]
        # train output dir
        self.train_output_dir = (
            "mgn_train_output_dir"
            if "train_output_dir" not in t_data
            else t_data["train_output_dir"]
        )
        # checkpoint
        self.checkpoint_dir = (
            join("checkpoints", self.name)
            if "checkpoint_dir" not in t_data
            else t_data["checkpoint_dir"]
        )
        # logs
        self.log_dir = (
            join("log", self.name) if "log_dir" not in t_data else t_data["log_dir"]
        )
        # tensorboard
        self.tensorboard_dir = (
            join("tensorboard", self.name)
            if "tensorboard_dir" not in t_data
            else t_data["tensorboard_dir"]
        )
        # how often to save a model and rollout during training
        self.save_interval = (
            None
            if "save_interval" not in t_data
            else int(t_data.getfloat("save_interval"))
        )
        self.run_rollout_at_interval = (
            False
            if "run_rollout_at_interval" not in t_data
            else t_data.getboolean("run_rollout_at_interval")
        )

        if self.use_tensorboard:
            makedirs(self.tensorboard_dir, exist_ok=True)
        makedirs(self.checkpoint_dir, exist_ok=True)

        # ddp
        self.ddp_hook = "mean" if "ddp_hook" not in t_data else t_data.get("ddp_hook")
        self.pin_memory = (
            False if "pin_memory" not in t_data else t_data.getboolean("pin_memory")
        )

        # distributed
        self.ddp_type = "srun" if "ddp_type" not in t_data else t_data.get("ddp_type")
        self.world_size = (
            1 if "world_size" not in t_data else t_data.getint("world_size")
        )
        self.data_num_workers = (
            0 if not "data_num_workers" in c_data else c_data.getint("data_num_workers")
        )

        # --------------- scheduler-related parameters ---------------
        if "SCHEDULER" not in config:
            self.scheduler_params = {}
        else:
            # all params under [SCHEDULER] must be numerical (int or float) for this dictionary comprehension to work
            self.scheduler_params = {
                k: config["SCHEDULER"].getfloat(k) for k in config["SCHEDULER"]
            }

        # --------------- partitioning-related parameters ---------------
        if "PARTITIONING" not in config:
            self.partitioning_params = {}
        else:
            self.partitioning_params = {
                k: literal_eval(config["PARTITIONING"][k])
                for k in config["PARTITIONING"]
            }

        # --------------- testing-related parameters ---------------
        c_test = config["TESTING"]

        self.test_batch_size = (
            self.batch_size
            if "batch_size" not in c_test
            else c_test.getint("batch_size")
        )
        self.do_rollout_test = (
            False
            if "do_rollout_test" not in c_test
            else c_test.getboolean("do_rollout_test")
        )
        self.rollout_start_idx = (
            100
            if "rollout_start_idx" not in c_test
            else c_test.getint("rollout_start_idx")
        )
        self.test_output_dir = (
            "mgn_prediction"
            if "test_output_dir" not in c_test
            else c_test["test_output_dir"]
        )
        makedirs(self.test_output_dir, exist_ok=True)
        self.rollout_dir = join(
            self.test_output_dir,
            "rollout" if "rollout_dir" not in c_test else c_test["rollout_dir"],
        )
        makedirs(self.rollout_dir, exist_ok=True)

        if "outfile" not in c_test:
            self.outfile = join(
                self.rollout_dir,
                ("rollout_%s.pk" % dt.datetime.now().strftime("%Y%m%dT%H%M%S")),
            )
        else:
            self.outfile = c_test["outfile"]

        # --------------- other data params, custom to specific tasks ---------------
        main_config_sections = ["DATA", "MODEL", "TRAINING", "SCHEDULER", "TESTING"]
        other_config_sections = [
            key for key in config.keys() if key not in main_config_sections
        ]

        self.other_data_params = dict()
        for section in other_config_sections:
            self.__dict__[section] = {}
            section_params = config[section]
            keys = set(section_params.keys())
            for key in keys:
                try:
                    val = literal_eval(section_params[key])
                except:
                    val = section_params[key]
                self.other_data_params[key] = val
                # this will set as class variable!
                self.__dict__[section][key] = val

    # --------------- get methods ---------------

    def get_config_file(self):
        return self.config_file

    def get_class_name(self):
        return self.class_name

    def get_name(self):
        return self.name

    def get_normal_node_types(self):
        return self.normal_node_types

    def get_boundary_node_types(self):
        return self.boundary_node_types

    def get_source_node_types(self):
        return self.source_node_types

    def get_num_node_types(self):
        return self.num_node_types

    def get_graph_type(self):
        return self.graph_type

    def get_graph_types(self):
        return self.graph_types

    def get_k(self):
        return self.k

    def get_radius(self):
        return self.radius

    def get_output_type(self):
        return self.output_type

    def get_window_length(self):
        return self.window_length

    def get_apply_onehot(self):
        return self.apply_onehot

    def get_apply_partitioning(self):
        return self.apply_partitioning

    def get_noise(self):
        return self.noise

    def get_noise_gamma(self):
        return self.noise_gamma

    def get_normalize(self):
        return self.normalize

    def get_model_arch(self):
        return self.model_arch

    def get_graph_processor_type(self):
        return self.graph_processor_type

    def get_mgn_dim(self):
        return self.mgn_dim

    def get_mp_iterations(self):
        return self.mp_iterations

    def get_mlp_norm_type(self):
        return self.mlp_norm_type

    def get_normalize_output(self):
        return self.normalize_output

    def get_connection_type(self):
        return self.connection_type

    def get_connection_alpha(self):
        return self.connection_alpha

    def get_connection_aggregation(self):
        return self.connection_aggregation

    def get_integrator(self):
        return self.integrator

    def get_epochs(self):
        return self.epochs

    def get_steps(self):
        return self.steps

    def get_batch_size(self):
        return self.batch_size

    def get_grad_accum_steps(self):
        return self.grad_accum_steps

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_lr(self):
        return self.lr

    def get_weight_decay(self):
        return self.weight_decay

    def get_scheduler(self):
        return self.scheduler

    def get_use_parallel(self):
        return self.use_parallel

    def get_use_amp(self):
        return self.use_amp

    def get_use_sbp(self):
        return self.use_sbp

    def get_sbp_start_epoch(self):
        return self.sbp_start_epoch

    def get_sbp_rate(self):
        return self.sbp_rate

    def get_sbp_percent(self):
        return self.sbp_percent

    def get_sbp_randomness(self):
        return self.sbp_randomness

    def get_load_prefix(self):
        return self.load_prefix

    def get_log_rate(self):
        return self.log_rate

    def get_use_tensorboard(self):
        return self.use_tensorboard

    def get_tb_rate(self):
        return self.tb_rate

    def get_train_time_limit(self):
        return self.train_time_limit

    def get_validation_time_limit(self):
        return self.validation_time_limit

    def get_sample_limit(self):
        return self.sample_limit

    def get_expt_name(self):
        return self.expt_name

    def get_train_output_dir(self):
        return self.train_output_dir

    def get_checkpoint_dir(self):
        return self.checkpoint_dir

    def get_log_dir(self):
        return self.log_dir

    def get_tensorboard_dir(self):
        return self.tensorboard_dir

    def get_save_interval(self):
        return self.save_interval

    def get_run_rollout_at_interval(self):
        return self.run_rollout_at_interval

    def get_ddp_hook(self):
        return self.ddp_hook

    def get_pin_memory(self):
        return self.pin_memory

    def get_ddp_type(self):
        return self.ddp_type

    def get_world_size(self):
        return self.world_size

    def get_data_num_workers(self):
        return self.data_num_workers

    def get_scheduler_params(self):
        return self.scheduler_params

    def get_partitioning_params(self):
        return self.partitioning_params

    def get_do_rollout_test(self):
        return self.do_rollout_test

    def get_rollout_start_idx(self):
        return self.rollout_start_idx

    def get_test_output_dir(self):
        return self.test_output_dir

    def get_rollout_dir(self):
        return self.rollout_dir

    def get_outfile(self):
        return self.outfile

    def get_other_data_params(self):
        return self.other_data_params

    def get_distributed_sampler_bounds(self):
        return self.distributed_sampler_bounds

    #######

    def get_train_data_params(self):
        train_data_params = {
            "root_dir": self.train_output_dir,
            "graph_type": self.graph_type,
            "k": self.k,
            "radius": self.radius,
            "output_type": self.output_type,
            "window_length": self.window_length,
            "apply_onehot": self.apply_onehot,
            "apply_partitioning": self.apply_partitioning,
            "boundary_nodes": self.boundary_node_types,
            "source_nodes": self.source_node_types,
            "noise": self.noise,
            "noise_gamma": self.noise_gamma,
            "normalize": self.normalize,
        }

        return train_data_params

    def get_test_data_params(self):

        test_data_params = {
            "root_dir": self.test_output_dir,
            "graph_type": self.graph_type,
            "k": self.k,
            "radius": self.radius,
            "output_type": self.output_type,
            "window_length": self.window_length,
            "apply_onehot": self.apply_onehot,
            "apply_partitioning": self.apply_partitioning,
            "boundary_nodes": self.boundary_node_types,
            "source_nodes": self.source_node_types,
            "normalize": self.normalize,
        }

        return test_data_params
