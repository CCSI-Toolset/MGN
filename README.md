# MeshGraphNets

This code base contains PyTorch implementations of graph neural networks for CFD simulation surrogate development. The plan is to apply this code to predict volume fraction fields associated with CFD simulations of a representative column model (Fu et al., 2020) for solvent-based carbon capture within the CCSI2 project. 

Pfaff et al., 2021. "[Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)." *International Conference on Learning Representations (ICLR)* (2021)

Fu et al., 2020. "[Investigation of Countercurrent Flow Profile and Liquid Holdup in Random Packed Column with Local CFD Data](https://www.sciencedirect.com/science/article/pii/S0009250920302256?)." *Chemical Engineering Science* 221 (2020)


## Authors
    - Phan Nguyen
    - Brian Bartoldson
    - Sam Nguyen
    - Jose Cadena
    - Rui Wang
    - David Widemann
    - Brenda Ng

## Requirements
    - matplotlib 
    - networkx 
    - numpy 
    - pandas 
    - scipy 
    - PyTorch 
    - PyTorch Geometric 
    - PyTorch Scatter 
    - tqdm

## Sample usage

```
cd training_scripts
bash train_mgn_config_cylinderflow_np.sh
```

## Release

LLNL release number: LLNL-CODE-829430

