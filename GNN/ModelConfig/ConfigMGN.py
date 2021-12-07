from configparser import ConfigParser, ExtendedInterpolation
from ast import literal_eval
from os import makedirs
from os.path import join
import datetime as dt

class ModelConfig:
    def __init__(self, config_file):
        # parse config file
        config = ConfigParser(inline_comment_prefixes='#', allow_no_value=True,
                                           interpolation=ExtendedInterpolation())
        config.optionxform = str #preserve case of keys
        config.read(config_file)

        #######

        #data-related parameters

        c_data = config['DATA']

        #dataset class
        self.class_name = c_data['class_name']

        self.name = 'untitled' if 'name' not in c_data else c_data['name']
        self.train_dir = None if 'train_dir' not in c_data else c_data['train_dir']
        self.test_dir = None if 'test_dir' not in c_data else c_data['test_dir']
        # self.data_file = None if 'data_file' not in c_data else c_data['data_file']
        # self.mesh_file = None if 'mesh_file' not in c_data else c_data['mesh_file']

        #graph construction parameters, if a graph is not provided for certain dataset classes
        self.graph_type = 'radius' if 'graph_type' not in c_data else c_data['graph_type']
        self.k = 10 if 'k' not in c_data else c_data.getint('k')
        self.radius = 0.01 if 'radius' not in c_data else c_data.getfloat('radius')

        #processing parameters
        self.output_type = 'state' if 'output_type' not in c_data else c_data['output_type']
        self.window_length = 5 if 'window_length' not in c_data else c_data.getint('window_length')
        self.apply_onehot = False if 'apply_onehot' not in c_data else c_data.getboolean('apply_onehot')
        self.boundary_node_types = [1] if 'boundary_node_types' not in c_data else literal_eval(c_data['boundary_node_types'])
        self.source_node_types = [2] if 'source_node_types' not in c_data else literal_eval(c_data['source_node_types'])
        self.noise = None if 'noise' not in c_data else literal_eval(c_data['noise'])
        self.noise_gamma = 0.1 if 'noise_gamma' not in c_data else c_data.getfloat('noise_gamma')
        self.normalize = True if 'normalize' not in c_data else c_data.getboolean('normalize')

        #other params
        self.other_data_params = dict()
        keys = set(c_data.keys())
        keys = keys.difference({'class_name', 'name', 'train_dir', 'test_dir', 
        	'graph_type', 'k', 'radius', 
            'output_type', 'window_length', 'apply_onehot',
        	'boundary_node_types', 'source_node_types', 'noise', 'noise_gamma',
            'normalize'})

        for key in keys:
        	try:
        		self.other_data_params[key] = literal_eval(c_data[key])
        	except:
        		self.other_data_params[key] = c_data[key]

        #######

        #MGN-related parameters

        m_data = config['MODEL']

        self.mgn_dim = 128 if 'mgn_dim' not in m_data else m_data.getint('mgn_dim')
        self.mp_iterations = 15 if 'mp_iterations' not in m_data else m_data.getint('mp_iterations')
        self.mlp_norm_type = 'LayerNorm' if 'mlp_norm_type' not in m_data else m_data['mlp_norm_type']
        self.mlp_norm_type = None if self.mlp_norm_type == 'None' else self.mlp_norm_type
        #future work: allow different normalization types at each mp iteration

        #######

        #training-related parameters

        t_data = config['TRAINING']

        self.epochs = 100 if 'epochs' not in t_data else t_data.getint('epochs')
        self.batch_size = 8 if 'batch_size' not in t_data else t_data.getint('batch_size')
        self.scheduler = 'ExpLR' if 'scheduler' not in t_data else t_data['scheduler']
        self.use_parallel = False if 'use_parallel' not in t_data else t_data.getboolean('use_parallel')

        #checkpoint
        self.checkpoint_dir = join('checkpoints', self.name) if 'checkpoint_dir' not in t_data else t_data['checkpoint_dir']

        #tensorboard
        self.use_tensorboard = False if 'use_tensorboard' not in t_data else t_data.getboolean('use_tensorboard')
        self.tensorboard_dir = join('tensorboard', self.name) if 'tensorboard_dir' not in t_data else t_data['tensorboard_dir']
        self.tb_rate = 10 if 'tb_rate' not in t_data else t_data.getint('tb_rate')
        self.log_rate = 100 if 'log_rate' not in t_data else t_data.getint('log_rate')

        if self.use_tensorboard:
        	makedirs(self.tensorboard_dir, exist_ok=True)
        makedirs(self.checkpoint_dir, exist_ok=True)

    #######

        #testing-related parameters
        c_test = config['TESTING']
        self.do_rollout_test = False if 'do_rollout_test' not in c_test else c_test.getboolean('do_rollout_test')
        self.rollout_start_idx = 100 if 'rollout_start_idx' not in c_test else c_test.getint('rollout_start_idx')
        self.test_output_dir = 'mgn_prediction' if 'test_output_dir' not in c_test else c_test['test_output_dir']
        makedirs(self.test_output_dir, exist_ok=True)
        if 'outfile' not in c_test:
            self.outfile = join(self.test_output_dir, ('rollout_%s.pk' % dt.datetime.now().strftime("%Y%m%dT%H%M%S")))
        else:
            self.outfile = c_test['outfile']

    #get methods

    ###

    def get_class_name(self):
    	return self.class_name

    def get_name(self):
        return self.name

    def get_train_dir(self):
        return self.train_dir

    def get_test_dir(self):
        return self.test_dir

    # def get_data_file(self):
    #     return self.data_file

    # def get_mesh_file(self):
    #     return self.mesh_file

    def get_graph_type(self):
        return self.graph_type

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

    def get_boundary_node_types(self):
        return self.boundary_node_types

    def get_source_node_types(self):
        return self.source_node_types

    def get_noise(self):
        return self.noise

    def get_noise_gamma(self):
        return self.noise_gamma

    def get_normalize(self):
        return self.get_normalize

   ###

    def get_mgn_dim(self):
        return self.mgn_dim

    def get_mp_iterations(self):
        return self.mp_iterations

    def get_mlp_norm_type(self):
        return self.mlp_norm_type

   ###

    def get_epochs(self):
        return self.epochs

    def get_batch_size(self):
        return self.batch_size

    def get_scheduler(self):
        return self.scheduler

    def get_use_parallel(self):
        return self.use_parallel

    def get_checkpoint_dir(self):
        return self.checkpoint_dir

    def get_use_tensorboard(self):
        return self.use_tensorboard

    def get_tensorboard_dir(self):
        return self.tensorboard_dir

    def get_tb_rate(self):
        return self.tb_rate

    def get_log_rate(self):
        return self.log_rate

    #######

    def get_do_rollout_test(self):
        return self.do_rollout_test

    def get_rollout_start_idx(self):
        return self.rollout_start_idx

    def get_test_output_dir(self):
        return self.test_output_dir

    def get_outfile(self):
        return self.outfile

    #######

    def get_train_data_params(self):
        train_data_params = {
            'root_dir': self.train_dir,
            'graph_type': self.graph_type,
            'k': self.k,
            'radius': self.radius,
            'output_type': self.output_type,
            'window_length': self.window_length,
            'apply_onehot': self.apply_onehot,
            'boundary_nodes': self.boundary_node_types,
            'source_nodes': self.source_node_types,
            'noise': self.noise,
            'noise_gamma': self.noise_gamma,
            'normalize': self.normalize
        }
        train_data_params.update(self.other_data_params)

        return train_data_params

    def get_test_data_params(self):
        test_data_params = self.get_train_data_params()
        test_data_params['root_dir'] = self.test_dir
        del test_data_params['noise']
        del test_data_params['noise_gamma']

        return test_data_params

    # def get_model_params(self):
    #     pass
