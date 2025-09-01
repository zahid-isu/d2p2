# [Balancing Utility and Privacy: Dynamically Private SGD with Random Projection (TMLR 2025)](https://openreview.net/pdf?id=u6OSRdkAwl)

This README file describes how to set up the environment, run training scripts for different models/datasets and store training outputs (JSON/plots).


## Models and Datasets
| Models | Datasets | Script file(s) |
|---|---|---|
| `LinearModel` `MultilayerPerceptron` `LogisticRegression` | MNIST-family (`MNIST` `FMNIST` `KMNIST` `EMNIST`) | `mnist.py`, `fmnist.py`, `kmnist.py`, `emnist.py` |
| `ConvolutionalNeuralNet` | CIFAR-10 (adaptable to MNIST) | `cifar10.py` |
| `DenseNet3` | CIFAR-10 | `cifar10.py` |
| `ResNet20`, `ResNet20Small`, `ResNet20Medium` | CIFAR-10 | `cifar10.py` |

Notes:
- The `models.py` contains different model classes indicating it can be adapted for MNIST-like/other datasets by adjusting input channels and fully-connected layers.
- If you add or modify models, update `models.py` and the corresponding call site in the script you intend to run.



## 1) Create the environment
The repository contains a conda environment file `env.yml`. It pins Python and PyTorch versions. Key versions from the YAML:
- Python: 3.9+
- PyTorch: 2.2.0 (with compatible `torchvision` and `torchaudio`)

To create and activate the conda environment run:

```bash
conda env create -f env.yml
conda activate d2p2
```

## 2) Start training using `.sh` scripts
There are convenience shell scripts in the repo to launch training. The scripts expose common hyper-parameters such as `batch_sizes` (training batch size), `red_rate` (reduction rate for random prjection), `sigma` (noise multiplier), `seeds` (random seed), device choice (CUDA index) and worker count. Example scripts include:

- `run_cnn_cifar.sh` — launches the CNN training flow (the script calls `cifar10.py`).

Typical run command:

```bash
# make sure conda env is activated
# from the repository root make the script executable
chmod +x run_cnn_cifar.sh
# run .sh file to start training
./run_cnn_cifar.sh
```

How to change parameters:
- The `.sh` script typically set variables like `BATCH_SIZE`, `SIGMA`, `RED_RATE`, `CUDA_DEVICE`, and `WORKERS`.

Example snippet:
```bash
CUDA_DEVICE=0        # switch to 1 or other index as needed
BATCH_SIZE=512
SIGMA=3.0
RED_RATE=0.3
WORKERS=4
python cifar2.py --batch-size $BATCH_SIZE --sigma $SIGMA --red_rate $RED_RATE --workers $WORKERS --device cuda:$CUDA_DEVICE
```

### DP modes (dp_mode)

For example, `cifar10.py` iterates over a list of privacy modes in the code:

```python
for dp_mode in [ None, 'static', 'dynamic', 'RP', 'd2p2']:  # [SGD, DP-SGD, D2P-SGD, DP2-SGD]
```

Short explanation of each value:
- `None` — vanilla SGD (no differential privacy). `args.disable_dp` will be True for this mode.
- `'static'` — standard DP-SGD with a fixed noise multiplier (`--sigma`) for all epochs.
- `'dynamic'` — DP-SGD with a dynamic noise multiplier: the script reduces sigma each epoch (it sets
  `optimizer.noise_multiplier = args.sigma / (epoch ** 0.25)` inside the training loop).
- `'RP'` — DP-SGD with Random Projection. Use `--red_rate` to control the projection reduction rate; the optimizer
  is configured with `optimizer.red_rate = args.red_rate` for this mode.
- `'d2p2'` — a D2P2-style variant that combines dynamic sigma scheduling with random projection (also uses `--red_rate`).

Notes
- The simplest way to run only one mode is to edit that single line in `cifar10.py` and leave only the mode you want, for example:

```python
for dp_mode in ['dynamic']:
    # runs only dynamic DP-SGD
```

- Alternatively change the list to a multiple modes if you want to run multiple training loops.
Flags to remember
- For `RP` and `d2p2` modes include `--red_rate` on the command line or export it as an env var used by the `.sh` wrapper.
- For DP modes, set `--sigma` (noise multiplier) and `--max-per-sample-grad_norm` appropriately.


## 3) Output files and format
Training runs save plots and JSON logs under `final/`, `log/`, or dataset-specific subfolders (for example `final/CNN_svhn1/`). Typical artifacts:

- PNG plots — training curves and result visuals. Example path: `final/CNN_svhn1/2024-09-17 01:13:04_sigma_3.0_batch_1024_seed_42.png`
- JSON files — per-run structured logs storing losses, accuracies, epsilons, and sigma per epoch. Example path: `final/CNN_svhn1/2024-09-17 01:13:04_sigma_3.0_batch_1024_seed_42.json`
- Checkpoints — `checkpoint.tar` or `model_best.pth.tar` in repo root or experiment folders.


The training scripts have a helper like `plot_combined_results(train_results, sigma, batch_size, red_rate, seed)`.  Key points:

- `train_results` is a dictionary keyed by the optimizer/DP label (for example `"SGD"`, `"DP-SGD(static)"`, `"DP-SGD(dynamic)"`). Each value is itself a dict with arrays for the keys: `loss`, `acc`, `ep`, and `sigma`.
- The plotting function creates a figure with two subplots: the top shows training loss vs epoch and the bottom shows test accuracy vs epoch. It iterates over `train_results.items()` and plots `result['loss']` and `result['acc']` for each label.
- Filenames use a timestamp plus key hyper-parameters. For example in `fmnist.py` the filename is built as:

```
filename = f'final/CNN_Fmnist_tmlr/{current_time}_sigma_{sigma}_batch_{batch_size}_seed_{seed}'
```

This produces both `filename.png` (the plot) and `filename.json` (the structured log). The plot includes a `suptitle` with the sigma, batch, and red_rate for quick identification.
- The PNG is created with matplotlib's `plt.savefig()`; the JSON is written using `json.dump(train_results, file, indent=4)` so it is human-readable and easy to parse programmatically.
- The saved JSON follows the same structure used to draw the plots. 

JSON structure example:

```json
{
  "SGD": {
    "loss": [2.26, 2.23, ...],
    "acc": [0.1958, 0.1958, ...],
    "ep": [0, 0, ...],
    "sigma": [3.0, 3.0, ...]
  },
  "DP-SGD(static)": {
    "loss": [2.25, 2.24, ...],
    "acc": [0.1958, 0.1973, ...],
    "ep": [0.145, 0.204, ...],
    "sigma": [3.0, 3.0, ...]
  },
  "DP-SGD(dynamic)": { ... }
}
```

The real JSON files contain full arrays for each epoch and each DP/mode tested. Use these JSON files to reproduce plots.

Filepath edit (argparse + plotting change)
----------------------------------------

1. Add an `--output-dir` argument in the script's `parse_args()`:

```python
parser.add_argument(
    "--output-dir",
    type=str,
    default="final/CNN_fmnist",
    help="Directory where PNG and JSON outputs will be saved",
)
```

2. Update `plot_combined_results` to accept `output_dir` and create it if missing. Replace the `filename` logic with a safe timestamp and join to `output_dir`:

```python
import os
from datetime import datetime

def plot_combined_results(train_results, sigma, batch_size, red_rate, seed, output_dir="final/CNN_fmnist"):
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(2, figsize=(10, 10), dpi=400)
    # ... plotting code unchanged ...

    # safer timestamp (no colons)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_dir, f"{current_time}_sigma_{sigma}_batch_{batch_size}_seed_{seed}")
    fig.suptitle(f"...")
    plt.savefig(f"{filename}.png")
    with open(f"{filename}.json", "w") as file:
        json.dump(train_results, file, indent=4)

```

3. Call the plotting function passing the parsed argument from `main()`:

```python
plot_combined_results(train_results, args.sigma, args.batch_size, args.red_rate, args.seed, args.output_dir)
```



## Tips & notes
- If a script uses `opacus` / differential privacy, some models require minor layer adjustments — `cifar2.py` runs `ModuleValidator` to detect unsupported layers.
- If you change model code (`models.py`) and get validator errors, either adjust the model or switch to the simple `convnet` in `cifar10.py` for quick experiments.
- For multi-GPU or Slurm clusters, see the `setup()` function in the scripts (`cifar2.py`/`cifar10.py`) — they detect Slurm environment variables and call `torch.distributed.init_process_group` accordingly.



## Contributing
### How to contribute
- Add a model class:
  1. Implement the new model in `models.py` following existing style (use `nn.Module`, provide a clear constructor and `forward`).
  2. Keep GroupNorm/BatchNorm choices compatible with `opacus` where possible; if your model uses an unsupported layer, add a small note in the PR describing why and how to make it compatible.

- Add a dataset/training script:
  1. Create a dataset-specific training script (copy an existing one like `fmnist.py` or `cifar10.py` and adapt). Keep parsing of command-line args consistent using `argparse`.
  2. Wire the model into the script (import the model from `models.py`), add a short validation run and use `plot_combined_results` or the shared plotting helper to write outputs.

- Validation and compatibility:
  - If the script supports DP (`opacus`), run `ModuleValidator.validate()` (see `cifar2.py`) and document any required fixes.
  - Prefer `ModuleValidator.fix(model)` before training so contributors can spot unsupported layers early.

- Outputs and reproducibility:
  - Use the `--output-dir` argument or `D2P2_OUTPUT_DIR` env var to place PNG/JSON outputs in a predictable location.
  - Add a small example run command in the new script's top-level docstring or the repo `README.md` table.

 ### Minimal smoke test
- Add a smoke test that:
  - Instantiates the model, runs a single forward pass with dummy input, and ensures no exceptions are raised.
  - Optionally runs a single training step (forward + backward) to catch shape/grad issues.

## Acknowledgements
This project builds on and is inspired by the [Opacus library](https://github.com/pytorch/opacus) for differential privacy in PyTorch. Thanks to the Opacus authors and the PyTorch community for their open-source work, which made developing the privacy tooling in this repo much easier.

## Citation
If you use this code in academic work, please cite the Opacus paper as well as this repository:

<div align="right">📋 Copy</div>

```bibtex
@article{d2p2Zhanhong,
  title={Balancing Utility and Privacy: Dynamically Private SGD with Random Projection},
  author={Zhanhong Jiang and Md Zahid Hasan and Nastaran Saadati and Aditya Balu and Chao Liu and Soumik Sarkar},
  journal={TMLR},
  url = {https://openreview.net/pdf?id=u6OSRdkAwl},
  year={2025}
}
```

## License
This code is released under Apache 2.0, as found in the original Opacus repo [LICENSE](https://github.com/pytorch/opacus/tree/main/LICENSE) file.
