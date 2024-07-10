# ArcGen
ArcGen: Toward Architecture-Generalized Neural Backdoor Detection via Alignment

## Generate the target models to be detected

```bash
python generate_target_benign.py --epoch 150 --batch_size 100 --dataset cifar10 --model resnet18 --target_prop 0.55 --proxy_prop 0.45 --target_num 256
python generate_trojaned.py --dataset cifar10 --epoch 150 --batch_size 100 --model resnet18 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --attack_type badnets --target_num 256
```

## Generate the proxy models for ArcGen

```bash
cd ./defence/ArcGen
python generate_given_benign.py  --epoch 150 --batch_size 100 --dataset cifar10 --model mobilnetv2 --target_prop 0.55 --proxy_prop 0.45 --target_num 256
python generate_given_benign.py  --epoch 150 --batch_size 100 --dataset cifar10 --model senet18 --target_prop 0.55 --proxy_prop 0.45 --target_num 256
python generate_proxy_trojaned.py  --epoch 150 --dataset cifar10 --batch_size 100  --model mobilnetv2 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --target_num 256 
python generate_proxy_trojaned.py  --epoch 150 --dataset cifar10 --batch_size 100  --model senet18 --target_prop 0.55 --proxy_prop 0.45 --attack_mode alltoone --target_num 256 
```

## Test

```bash
python ArcGen_detection.py --batch_size 30 --epoch 300 --num_workers 0 --dataset cifar10   --mask 0.02 --query_num 20
```
