# Recurrent Conditional Query Learning (RCQL)
This repository contains the Pytorch implementation of

[One Model Packs Thousands of Items with Recurrent Conditional Query Learning](https://www.sciencedirect.com/science/article/pii/S095070512100945X)

Dongda Li, Zhaoquan Gu, Yuexuan Wang, Changwei Ren, Francis C.M. Lau

We propose a Recurrent Conditional Query Learning (RCQL) method to solve both 2D and 3D packing problems. We first embed states by a recurrent encoder, and then adopt attention with conditional queries from previous actions. The conditional query mechanism fills the information gap between learning steps, which shapes the problem as a Markov decision process. Benefiting from the recurrence, a single RCQL model is capable of handling different sizes of packing problems. Experiment results show that RCQL can effectively learn strong heuristics for offline and online strip packing problems (SPPs), out- performing a wide range of baselines in space utilization ratio. RCQL reduces the average bin gap ratio by 1.83% in offline 2D 40-box cases and 7.84% in 3D cases compared with state-of-the-art methods. Meanwhile, our method also achieves 5.64% higher space utilization ratio for SPPs with 1000 items than the state of the art.

## Usage

### Preparation

1. Install conda
2. Run `conda env create -f environment.yml`

### Train

1. Modify the config file in `config.py` as you need.
2. Run `python main.py`.
