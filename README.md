# ArcGen: Generalizing Neural Backdoor Detection Across Diverse Architectures

This repository provides the official implementation of **ArcGen**, a method proposed in our paper:

> **ArcGen: Generalizing Neural Backdoor Detection Across Diverse Architectures**

ArcGen introduces a novel black-box backdoor detection framework that learns architecture-invariant alignment features. It enables effective generalization to unseen model architectures, which is crucial in real-world settings where the defender has no knowledge of the model internals.

## Overview

ArcGen detects whether a given model is backdoored without requiring access to its architecture or parameters. The detector is trained using both benign and backdoored *proxy models* constructed on *known* architectures. At test time, it generalizes to target models from *unseen* architectures.

Key features:

* Architecture-invariant feature learning
* Black-box model access (query-based only)
* Generalization to unseen DNN architectures
* Extensive support for CNNs and ViTs

---

## Project Structure

* `classifier_models/`: Backbone architectures for target and proxy models
* `generate_target_benign.py`: Train benign target models
* `generate_trojaned.py`: Train backdoored target models
* `generate_given_benign.py`: Train benign proxy models
* `generate_proxy_trojaned.py`: Train backdoored proxy models
* `ArcGen_detection.py`: Train and evaluate the ArcGen detector
* `defence/ArcGen/`: Core implementation of the ArcGen framework

---

## Preliminaries

**Before running the code:**

* Set the `num_classes` for each model architecture in `classifier_models/` based on the dataset.
* If the model is trained using the ImageNet dataset, the model architecture needs to be changed to use the model architecture in `classifier_models_imagenet/`.
* Install required dependencies (e.g., `torch`, `torchvision`, etc.)
* Unzip the ImageNet dataset: `unzip -o -d ./raw_data ./raw_data/imagenet_resized.zip`

**Environment Setting**

The environment setup for ArcGen is listed in requirements.txt. To install, run:

```shell
conda create -n ArcGen python=3.9
source activate ArcGen
pip install -r requirements.txt
```

---

## Training Target Models (CIFAR-10)

These are the models you want to test for backdoors.

```bash
# Benign target models
python generate_target_benign.py --epoch 150 --batch_size 100 --dataset cifar10 --model resnet18 --target_prop 0.55 --proxy_prop 0.45 --target_num 256

# Backdoored target models
python generate_trojaned.py --dataset cifar10 --epoch 150 --batch_size 100 --model resnet18 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --attack_type badnets --target_num 256
```

---

## Training Proxy Models for ArcGen (CIFAR-10)

These are models used to train the ArcGen detector.

```bash
cd ./defence/ArcGen

# Benign proxy models
python generate_given_benign.py --epoch 150 --batch_size 100 --dataset cifar10 --model mobilnetv2 --target_prop 0.55 --proxy_prop 0.45 --target_num 256
python generate_given_benign.py --epoch 150 --batch_size 100 --dataset cifar10 --model senet18 --target_prop 0.55 --proxy_prop 0.45 --target_num 256

# Backdoored proxy models
python generate_proxy_trojaned.py --epoch 150 --dataset cifar10 --batch_size 100 --model mobilnetv2 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --target_num 256 
python generate_proxy_trojaned.py --epoch 150 --dataset cifar10 --batch_size 100 --model senet18 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --target_num 256 
```

---

## Detection with ArcGen (CIFAR-10)

```bash
python ArcGen_detection.py --batch_size 30 --epoch 300 --num_workers 0 --dataset cifar10 --mask 0.02 --query_num 20
```

---

## Model Configuration for Different Datasets

When training models on different datasets, it is necessary to adjust the model architecture to suit the specific dataset. For example, you may need to modify the number of output classes by setting the `num_classes` parameter accordingly.

The model definitions under `classifier_models/` are dataset-specific. You can switch between configurations as follows:

* **Switching from CIFAR-10 to ImageNet:**

  ```bash
  mv classifier_models classifier_models_normal
  mv classifier_models_imagenet classifier_models
  ```
  
* **Switching from ImageNet back to CIFAR-10:**

  ```bash
  mv classifier_models classifier_models_imagenet
  mv classifier_models_normal classifier_models
  ```

In addition, for experiments on **GTSRB** and **MNTD**, please modify the model architecture in the `classifier_models/` folder (originally designed for CIFAR-10) according to the **inline code comments** to ensure compatibility with the respective dataset.

Make sure to perform the appropriate switch and code adjustment before training or evaluating models on a new dataset.

## Generalization to Vision Transformers (ImageNet Subset)

ArcGen supports detecting backdoors in ViTs trained on subsets of ImageNet.

### Switching from CIFAR-10 to ImageNet
  
```bash
mv classifier_models classifier_models_normal
mv classifier_models_imagenet classifier_models
```

### Train Target Models (ViTs)

```bash
python generate_target_benign.py --epoch 10 --batch_size 60 --dataset imagenet --model vit_b_16 --target_prop 0.55 --proxy_prop 0.45 --target_num 64
python generate_target_benign.py --epoch 10 --batch_size 60 --dataset imagenet --model vit_l_32 --target_prop 0.55 --proxy_prop 0.45 --target_num 64
python generate_target_benign.py --epoch 10 --batch_size 60 --dataset imagenet --model vit_b_32 --target_prop 0.55 --proxy_prop 0.45 --target_num 64

python generate_trojaned.py --dataset imagenet --epoch 10 --batch_size 60 --model vit_b_16 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --attack_type badnets --target_num 64
python generate_trojaned.py --dataset imagenet --epoch 10 --batch_size 60 --model vit_l_32 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --attack_type badnets --target_num 64
python generate_trojaned.py --dataset imagenet --epoch 10 --batch_size 60 --model vit_b_32 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --attack_type badnets --target_num 64
```

### Train Proxy Models (CNNs)

```bash
cd ./defence/ArcGen

# Benign proxy models
python generate_given_benign.py --epoch 150 --batch_size 100 --dataset imagenet --model mobilnetv2 --target_prop 0.55 --proxy_prop 0.45 --target_num 64
python generate_given_benign.py --epoch 150 --batch_size 100 --dataset imagenet --model resnet18 --target_prop 0.55 --proxy_prop 0.45 --target_num 64
python generate_given_benign.py --epoch 150 --batch_size 100 --dataset imagenet --model efficientnetb0 --target_prop 0.55 --proxy_prop 0.45 --target_num 64

# Backdoored proxy models
python generate_proxy_trojaned.py --dataset imagenet --epoch 150 --batch_size 100 --model mobilnetv2 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --target_num 64
python generate_proxy_trojaned.py --dataset imagenet --epoch 150 --batch_size 100 --model resnet18 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --target_num 64
python generate_proxy_trojaned.py --dataset imagenet --epoch 150 --batch_size 100 --model efficientnetb0 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --target_num 64
```

### Detection on ViTs

```bash
python ArcGen_detection.py --batch_size 30 --epoch 300 --num_workers 0 --dataset imagenet --mask 0.02 --query_num 20
```

---

## Contact

For questions or feedback, feel free to open an issue or contact the authors.
