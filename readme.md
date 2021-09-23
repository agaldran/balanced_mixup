# Balanced-MixUp for Highly Imbalanced Medical Image Classification
## Summary
This repository contains the code accompanying the paper:
```
Balanced-MixUp for Highly Imbalanced Medical Image Classification
Adrian Galdran, Gustavo Carneiro, Miguel A. González Ballester
MICCAI 2021
```
Link: [https://arxiv.org/abs/2109.09850](https://arxiv.org/abs/2109.09850)

Balanced MixUp is a relatively simple approach to perform classification on imbalanced data scenarios. It combines MixUp with conventional data sampling techniques. Briefly speaking, the idea is to sample a training data batch with minority class oversampling, another one without it, and then mix them up, normally giving more weight to the non-oversampled batch to avoid overfitting. In the paper we show that this approach improves performance for retinal image grading and endoscopic image classification.

The above idea has been implemented in this repository in Pytorch; the logic for data loading in the way described above can be found in `utils/get_loaders.py`, lines 90-178, and if you want to check how I mix up those two batches you can look into `train_lt_mxp.py`, lines 128-138.

You should be able to download the Eyepacs dataset from Kaggle. A pre-processed version ready to use can be found [here](https://www.kaggle.com/agaldran/eyepacs). In theory, you could also download the endoscopic dataset (Kvasir) by simply running `sh get_endo_data.sh`. Once you have the data ready, check the `run_lt_experiments.sh` file to see how to reproduce the experiments in the paper (for the mobilenet architecture). Note that the hyperparameter $alpha$ in the paper corresponds to the input parameter `do_mixup` in the code, *e.g.* you could call:
```
python train_lt_mxp.py --do_mixup 0.1 --save_path eyepacs/mxp_1e-1/mobilenet --model_name mobilenetV2 --n_epochs 30 --metric kappa
```

Please let me know if there is something that does not work as expected by opening an issue. Good luck!
