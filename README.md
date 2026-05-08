# Torch Model Template

A minimal, config-driven training harness for PyTorch models. 

Simply edit `config.yaml`, run `python main.py`, get a trained model and a loss curve. 

Adding a new model, optimizer, or loss is just a registry entry

## Quickstart

```bash
pip install -r requirements.txt
python main.py --config config.yaml
```

Each run writes a loss curve to `outputs/<run_id>.png` where `<run_id>` is the start timestamp (e.g. `20260507_232029`).

## Project structure

```
config.yaml      # all knobs (data, model, training, evaluation, output)
main.py          # CLI entry + run() orchestrator
data.py          # make_toy_data, load_dataframe, TabularDataset
preprocess.py    # build_preprocessor, preprocess_data
model.py         # MODELS registry + build_model
loss.py          # LOSSES registry + build_loss_fn
optim.py         # OPTIMIZERS registry + build_optimizer
training.py      # train_one_epoch, train_model, plot_loss_curves
evaluation.py    # evaluate, predict_proba
tests/           # pytest tests
```

## Writing the config

Each component (model, optimizer, loss) is selected by a `name` key that maps to a registry entry. The runner shallow-copies the section, pops `name`, and passes the rest as keyword arguments to the constructor. So if a model takes `dropout=0.1`, you write `dropout: 0.1` next to its `name`. No code changes needed to swap components.

### Full example

```yaml
data:
  use_toy_data: true
  toy_data:
    n: 5000
    seed: 42
  input_path: null         # used when use_toy_data is false
  target_col: label

split:
  test_size: 0.2
  random_state: 42
  stratify: true

preprocessing:
  numeric:
    imputer:
      strategy: median     # mean | median | most_frequent
    scaler:
      enabled: true
  categorical:
    imputer:
      strategy: constant
      fill_value: missing
    onehot:
      handle_unknown: ignore
      sparse_output: false

model:
  name: BinaryClassifier   # key in MODELS (model.py)
  hidden_dims: [128, 64]
  dropout: 0.1

training:
  batch_size: 512
  epochs: 10
  shuffle_train: true
  device: auto             # auto | cpu | cuda
  optimizer:
    name: AdamW            # key in OPTIMIZERS (optim.py)
    lr: 0.001
    weight_decay: 0.0001
  loss:
    name: BCEWithLogitsLoss  # key in LOSSES (loss.py)

evaluation:
  threshold: 0.5

inference:
  drop_target_col_if_present: true

output:
  dir: outputs
```

### Switching to a CSV dataset

```yaml
data:
  use_toy_data: false
  input_path: path/to/data.csv
  target_col: my_target
```

The preprocessing step infers numeric vs categorical columns from dtype: `object` and `category` columns go through impute → one-hot, everything else through impute (and optionally `StandardScaler`).

### Built-in registry entries

| Section                   | Available names                                      |
|---------------------------|------------------------------------------------------|
| `model.name`              | `BinaryClassifier`                                   |
| `training.optimizer.name` | `AdamW`, `Adam`, `SGD`                               |
| `training.loss.name`      | `BCEWithLogitsLoss`, `CrossEntropyLoss`, `MSELoss`   |

## Adding a new component

To plug in a new model:

1. Define the class in `model.py` (or import it there):
   ```python
   class MyModel(nn.Module):
       def __init__(self, input_dim: int, depth: int = 4):
           ...
   ```
2. Register it:
   ```python
   MODELS = {
       "BinaryClassifier": BinaryClassifier,
       "MyModel": MyModel,
   }
   ```
3. Use it in the config:
   ```yaml
   model:
     name: MyModel
     depth: 6
   ```

The same pattern applies to `OPTIMIZERS` in `optim.py` and `LOSSES` in `loss.py`.

## Development

```bash
pip install -r requirements-dev.txt
pre-commit install
pytest
```

`pre-commit install` installs both `pre-commit` and `pre-push` hooks (the framework is configured via `default_install_hook_types` in `.pre-commit-config.yaml`):

- **pre-commit**: `ruff check --fix` and `ruff format` on staged files.
- **pre-push**: full `pytest` suite must pass before the push goes through.

Lint runs on every commit, tests run on every push. Configuration lives in `pyproject.toml` (ruff, pytest) and `.pre-commit-config.yaml` (hook wiring).
