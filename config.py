import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./raw_data")
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--tensorboard_dir", type=str, default="./result-logs")
    parser.add_argument("--document", type=str, default="./results_and_analysis")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default=None)


    parser.add_argument("--target_prop", type=float, default=0.95)
    parser.add_argument("--proxy_prop", type=float, default=0.05)

    parser.add_argument("--target_num", type=int, default=100)    

    parser.add_argument("--attack_type", type=str, default='badnets')
    parser.add_argument("--attack_mode", type=str, default="alltoone")
    parser.add_argument("--attack_target", type=int, default=-1)
    parser.add_argument("--fixed_target", type=int, default=0)
    parser.add_argument("--poisoning_ratio", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--pc", type=float, default=0.1)
    parser.add_argument("--cross_ratio", type=float, default=2)  # rho_a = pc, rho_n = pc * cross_ratio

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)

    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98


    parser.add_argument("--query_num", type=int, default=10)  
    
    parser.add_argument("--mask", type=float, default=0.0)
    parser.add_argument("--save_detection_dir", type=str, default='./result-models')

    

    return parser
