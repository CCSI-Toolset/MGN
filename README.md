# MeshGraphNets

This code base contains PyTorch implementations of graph neural networks for CFD simulation surrogate development. The plan is to apply this code to predict volume fraction fields associated with CFD simulations of a representative column model (Fu et al., 2020) for solvent-based carbon capture within the CCSI2 project.

## Training Models

We provide two main settings for training models:

1. **Training on an Interactive Node**
  - Single process.
  - Multiple processes. Gradients synced using distributed data parallel (DDP).
    - Each process will have it's own GPU.
  - See the `README.md` in `training_scripts/` for more information.
  
2. **Training with Multiple Nodes using Job Schedulers**
  - Multiple processes. Gradients synced using distributed data parallel (DDP) **across multiple nodes**.
  - For **SLURM** based systems (e.g. NERSC), see the `README.md` in `bash_scripts/training/SLURM/` for more information.
  - For **LSF** based systems (e.g. LC's Lassen), see the `README.md` in `bash_scripts/training/LSF/` for more information. 

## Rolling Out Simulations

We provide two main settings for rollouts:

1. **Rollouts on an Interactive Node**
  - Single process, full-domain rollouts.
  - See the `README.md` in `rollout_scripts/` for more information.
  
2. **Rollouts with Multiple Nodes using Job Schedulers**
  - Multiple processes, patch rollouts. Patch syncing between processes, uses distributed data parallel (DDP) **across multiple nodes**.
  - For **SLURM** based systems (e.g. NERSC), see the `README.md` in `bash_scripts/rollouts/SLURM/` for more information.
  - For **LSF** based systems (e.g. LC's Lassen), see the `README.md` in `bash_scripts/rollouts/LSF/` for more information. 

## Relevant Publications/Datasets

[[10.48550/arXiv.2304.00338] Scientific Computing Algorithms to Learn Enhanced Scalable Surrogates for Mesh Physics](https://arxiv.org/abs/2304.00338)

[[10.1016/j.ces.2020.115693] Investigation of countercurrent flow profile and liquid holdup in random packed column with local CFD data](https://www.sciencedirect.com/science/article/pii/S0009250920302256)

[[10.48550/arXiv.2010.03409] Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)

[[10.25584/1963908] Packed Column Simulations](https://data.pnnl.gov/group/nodes/dataset/33472)

## Requirements
See [environment.yml](environment.yml).

## Authors

### V1
    - Phan Nguyen
    - Brian Bartoldson
    - Sam Nguyen
    - Jose Cadena
    - Rui Wang
    - David Widemann
    - Brenda Ng

### V2
    - Phan Nguyen
    - Brian Bartoldson
    - Amar Saini
    - Jose Cadena
    - Yeping Hu

## Release

LLNL release number (V1): LLNL-CODE-829430

LLNL release number (V2): LLNL-CODE-855189

This work was produced under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.

This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award BES-ERCAP0024437. 

