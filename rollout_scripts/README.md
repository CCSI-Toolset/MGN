# Rollout Scripts

*Note: This code is generalized to work with any dataset/config, but the READMEs will use the DM Cylinder Flow dataset/config as examples*

## Single-Node, Single-Process, Full Domain Rollouts

`python rollout_full_domains.py [CONFIG_FILE]`
- `[CONFIG_FILE]` - Must be an **absolute path** to a config file
- Predicts a full-domain rollout on an interactive node

## Single-Node, Multi-Process, Full Domain Rollouts

TBD

## Multiple Processes, Patch Rollouts.

- Refer to the `README.md`'s in `bash_scripts/rollout/SLURM/` and `bash_scripts/rollout/LSF/`