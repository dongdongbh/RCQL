# Recurrent Conditional Query Learning (RCQL)

PyTorch implementation of [One Model Packs Thousands of Items with Recurrent Conditional Query Learning](https://www.sciencedirect.com/science/article/pii/S095070512100945X)

**Authors:** Dongda Li, Zhaoquan Gu, Yuexuan Wang, Changwei Ren, Francis C.M. Lau

We propose a Recurrent Conditional Query Learning (RCQL) method to solve both 2D and 3D packing problems. We first embed states by a recurrent encoder, and then adopt attention with conditional queries from previous actions. The conditional query mechanism fills the information gap between learning steps, which shapes the problem as a Markov decision process. Benefiting from the recurrence, a single RCQL model is capable of handling different sizes of packing problems. Experiment results show that RCQL can effectively learn strong heuristics for offline and online strip packing problems (SPPs), outperforming a wide range of baselines in space utilization ratio. RCQL reduces the average bin gap ratio by 1.83% in offline 2D 40-box cases and 7.84% in 3D cases compared with state-of-the-art methods. Meanwhile, our method also achieves 5.64% higher space utilization ratio for SPPs with 1000 items than the state of the art.

## Usage

### Installation

1. Install conda
2. Run `conda env create -f environment.yml`

### Training

**Quick Start:**
```bash
# 3D packing with 10 boxes (default)
python main.py

# 2D packing with 40 boxes
python main.py --problem-type pack2d --block-num 40

# Online packing mode
python main.py --online
```

**Important Parameters** (in `config.py` or command line):
- `--problem-type`: `pack2d` or `pack3d`
- `--block-num`: Number of boxes per instance (must be divisible by `block-size`)
- `--block-size`: Sequence length for recurrent processing (default: 20)
- `--online`: Enable online packing mode (boxes arrive sequentially)
- `--niter`: Number of training iterations (default: 100,000)
- `--batch-sz`: Batch size (default: 128)

**Note:** Total boxes per instance = `block-num`, and must be an integer multiple of `block-size`. For example:
- 30 boxes: use `--block-num 30 --block-size 10` (10, 15, or 30 also valid)
- 100 boxes: use `--block-num 100 --block-size 20`

### Monitoring Training

**Training Metrics:**
- `gap_ratio`: Wasted space ratio (lower is better) - **primary performance metric**
- `avg_rewards`: Includes entropy regularization bonus (decreases as model becomes confident)
- `entropy`: Policy randomness (decreases during training - expected behavior)

**Important:** `avg_rewards` may decrease during training due to entropy regularization, but this is normal. Monitor `gap_ratio` to track actual packing performance improvement.

### Outputs

Results are saved to `outputs/{problem_type}_{block_num}/{run_name}_{timestamp}/`:
- `checkpoint.pt`: Model checkpoint
- `args.json`: Full configuration
- `log/`: Training logs (CSV and human-readable)
- TensorBoard events for visualization

## Architecture Details

- **Encoder**: Multi-layer Transformer with recurrent cache (default: 3 layers, 8 attention heads)
- **Decoder**: Separate decoders for selection, rotation, and positioning sub-actions
- **Training**: On-policy actor-critic with GAE (Generalized Advantage Estimation)
- **Context Size**: FIFO queues maintain last 20 packed/unpacked boxes for memory efficiency

## Citation

```bibtex
@article{li2022one,
  title={One model packs thousands of items with recurrent conditional query learning},
  author={Li, Dongda and Gu, Zhaoquan and Wang, Yuexuan and Ren, Changwei and Lau, Francis CM},
  journal={Knowledge-Based Systems},
  volume={235},
  pages={107683},
  year={2022},
  publisher={Elsevier}
}
```
