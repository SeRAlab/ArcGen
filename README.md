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

Before running the code:

* Set the `num_classes` for each model architecture in `classifier_models/` based on the dataset.
* Install required dependencies (e.g., `torch`, `torchvision`, etc.)

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

## Generalization to Vision Transformers (ImageNet Subset)

ArcGen supports detecting backdoors in ViTs trained on subsets of ImageNet.

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
