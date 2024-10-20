# Dissertation

## Setup

Create and setup virtualenv
```shell
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`TRAIN.md` file contains information about setting up machine with GPU and run train there.

## Structure

1. `classifier` directory contains high-level functions and classes for classification model.
2. `gan` directory contains high-level functions and classes for conditional GAN model.
3. `output` directory contains artifacts of training and testing models.
   1. `classifier` directory contains metrics with accuracy and losses, its plot and model for each of three: trained on real, synthetic and mixed dataset.
   2. `gan` directory contains metrics of losses with its plot.
      1. `gan-train` directory contains examples of image for each of GAN generation. 
      2. Model weights saved in [Google Drive](https://drive.google.com/file/d/1dHjoBR7ve9oqz_u85-qwfMVWlM5oMkkP/view?usp=sharing).
4. `classes.py` contains human readable names of classes from dataset.
5. `*_main.py` scripts are entrypoints for executing scripts.
6. `data.py` contains low-level functions for loading datasets.

## GAN

Train new gan model
```shell
python gan_main.py --action="train" --data-dir="./data" --cp-dir="./gan-cp/cp" --out-dir="./gan-train" --report-file="./losses.csv"
```

Generate images for each class from checkpoint
```shell
python gan_main.py --action="generate" --cp-path="./gan-cp/cp-1" --out-dir="./gan-generate" --count=1000
```

Plot loss metrics from file
```shell
python gan_main.py --action="plot_metrics" --report-file="./losses.csv" --output-file="./losses.png"
```

Merge natural and synthetic datasets into one
```shell
python gan_main.py --action="mix_datasets" --source-dir-1="./data/Train/" --source-dir-2="./gan-generate" --target-dir="./mixed"
```

## Classifier

Train model on train dataset
```shell
python classifier_main.py --action="train" --train-dir="./data/Train" --model-path="./model.h5" --report-file="./metrics.csv"
```

Test model on test dataset
```shell
python classifier_main.py --action="test" --model-path="./model.h5" --test-dir="./data"
```

Make test for single image
```shell
python classifier_main.py --action="test_once" --model-path="./model.h5" --image-path="./data/Test/1.png"
```

Plot accuracy and loss metrics from file
```shell
python classifier_main.py --action="plot_metrics" --report-file="./metrics.csv" --output-file="./metrics.png"
```
