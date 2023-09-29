# Training Scripts

*Note: This code is generalized to work with any dataset/config, but the READMEs will use the DM Cylinder Flow dataset/config as examples*

## Single-Node, Single-Process Training

`python train_mgn_generalized.py --config [CONFIG_FILE]`
- `[CONFIG_FILE]` - Must be an **absolute path** to a config file
- Trains a model on an interactive single node

## Single-Node, Multi-Process Distributed Data Parallel (DDP) Training

`python train_mgn_generalized_ddp.py --config [CONFIG_FILE]`
- `[CONFIG_FILE]` - Must be an **absolute path** to a config file
- Be sure to set `ddp_type` to `manual` in the configuration file
- Trains a model on an interactive single node, using DDP

## Multi-Node, Multi-Process Distributed Data Parallel (DDP) Training

- Refer to the `README.md`'s in `bash_scripts/train/SLURM/` and `bash_scripts/train/LSF/`