# Improving information gain for robustness in node-based Bayesian neural networks

This repository contains a PyTorch implementation of our project for the MVA Bayesian Machine learning class.

Please cite our work if you find it useful:

```bibtex
@misc{info_bayesian_vts,
  title = {Improving information gain for robustness in node-based Bayesian neural networks},
  authors = {Vaillant-Tenzer, Constantin and Scheffler, Mathis}
  year = {2025}
}
```

## Installation

Get into this repository and run:
```bash
conda env create
conda activate nodes_env
```



## Downloading the datasets
To run the experiments, one needs to run the following commands to download the necessary datasets and store them in the `data` folder:
```bash
bash download_scripts/download_cifar10_c.sh
bash download_scripts/download_cifar100_c.sh
bash download_scripts/download_tinyimagenet.sh
bash download_scripts/download_tinyimagenet_c.sh
```

## File Structure

```
+-- utils.py (utility functions and modules)
+-- datasets.py (containing functions to load data)
+-- train.py (script for training our model)
+-- loss.py (script to compute the loss)
+-- dataset.py (script to use the dataset)
+-- train_sgd.py (script for training deterministic models)
+-- train_vanila.py (to train our own model)
+-- main_vanila.py (script to lauch the model implemented in trinh22a)
+-- main.py (script to run our implementation of the model)
+-- batch_train.sh (if you need to run our script in background)
+-- model.py (script that contains the basic model)
+-- plot.py (to plot models comparisons)
+-- tiny_imagenet (contains useful parts of the model)
```

## Command to replicate the result

```bash
python main.py #for our model
python main_vanila.py #for the model proposed in trinh22a
```

Alternatively, if you use slurm, you may after having personalized your file, use :

```bash
sbatch batch_train.py
```

## References

We mainly started from the code developped here by Trung Trinh, Markus Heinonen, Luigi Acerbi and Samuel Kaski:

[Tackling covariate shift with node-based Bayesian neural networks](https://proceedings.mlr.press/v162/trinh22a.html)

For more information about their paper, please visit the [website](https://aaltopml.github.io/node-BNN-covariate-shift).
