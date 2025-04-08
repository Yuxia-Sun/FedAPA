import argparse
import copy
import time

import numpy as np
import torch
import yaml
from torch import nn
from models.FedAvgCNN import FedAvgCNN
from dataset.generate_dataset import generate_dataset
from models.models import Net, BaseHeadMerge
from utils import save_result
from src.FedAPA.server_FedAPA import ServerFedAPA
from models.ResNet import ResNet18
from models.ResNet_T import resnet4
# from src.pFedLA import server_pFedLA
from utils import readable_size





def prepare_models(args):
    dim=0
    in_channels=0
    if args.dataset == "mnist":
        args.num_classes = 10  # 类别数
        args.feature_dim = 84
        dim = 16 * 4 * 4  # 具体数值需要根据样本尺寸变化，mnist 图片尺寸为 1*28*28，黑白
        in_channels=1

    elif args.dataset == "fmnist":
        args.num_classes = 10  # 类别数
        dim = 16 * 4 * 4  # fmnist 图片尺寸为 1*28*28，黑白
        in_channels=1
        # args.model = Net(in_channels=1, num_classes=10, dim=dim)
        args.feature_dim = 84

    elif args.dataset == "cifar10":
        args.num_classes = 10  # 类别数
        dim = 16 * 5 * 5  # cifar10 图片尺寸为 3*32*32，彩色
        in_channels=3
        # args.model = Net(in_channels=3, num_classes=10, dim=dim)
        args.feature_dim = 84

    elif args.dataset == "cifar100":
        args.num_classes = 100  # 类别数
        dim = 16 * 5 * 5  # cifar100 图片尺寸为 3*32*32，彩色
        in_channels=3
        # args.model = Net(in_channels=3, num_classes=100, dim=dim)
        args.feature_dim = 84

    elif args.dataset == "kronoDroid":
        args.num_classes = 13  # 类别数
        dim = 16 * 1 * 1  # cifar100 图片尺寸为 3*18*18，彩色
        in_channels=3
        args.feature_dim = 84

    else:
        raise NotImplementedError



    if args.modelName =='LeNet5':
        args.model = Net(in_channels=in_channels, num_classes=args.num_classes, dim=dim)  # 基础分类模型，对这个模型后续训练过程不能有任何改动
    elif args.modelName=="ResNet18":
        args.model = ResNet18()
        args.model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        args.model.fc = torch.nn.Linear(512, args.num_classes)  # 将最后的全连
    elif args.modelName=="resnet4":
        args.model = resnet4(num_classes=args.num_classes)
    elif args.modelName=="FedAvgCNN":
        args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

    print(f"Model Framework: {args.model}")
    return













def run(args):
    results = {"experiment": [], "best_acc": [], "upload": [], "download": [], "allTime": [], "iterationTime": []}
    for repeat_time in range(args.num_repeat_times):
        print(f"================= Repeat Time: {repeat_time + 1} =================")
        args.train_config, args.statistic = generate_dataset(args)
        prepare_models(args)

        
      
        if args.algorithm == "FedAPA":  # Federated Adaptive Parameter Aggregation
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
            server = ServerFedAPA(args)

        
        else:
            raise NotImplementedError

        all_start_time = time.time()
        acc_record, uploadCom, downloadCom = server.train()
        all_end_time = time.time()
        time_all = all_end_time - all_start_time

        result = {"statistic": args.statistic, "acc_record": acc_record}

        results["experiment"].append(result)
        results["best_acc"].append(max(acc_record.values()))
        results["upload"].append(uploadCom)
        results["download"].append(downloadCom)
        results["allTime"].append(time_all)
        results["iterationTime"].append(time_all / args.server['global_rounds'])

    print("============ Saving Result ============")
    results["config"] = args.train_config  # 只在最后保存一下训练的配置和超参
    results["mean"] = np.mean(results["best_acc"]).item()
    results["stdvar"] = np.var(results["best_acc"]).item()
    results["uploadMean"] = readable_size(np.mean(results["upload"]).item())
    results["downloadMean"] = readable_size(np.mean(results["download"]).item())
    results["allTimeMean"] = np.mean(results["allTime"]).item()
    results["iterationTimeMean"] = np.mean(results["iterationTime"]).item()
    results["model"]= args.modelName
    save_result(dataset=args.dataset,
                distribution=args.distribution,
                algorithm=args.algorithm,
                num_clients=args.client["num_clients"],
                result=results,
                prefix=args.prefix)
    print("============     DONE!     ============")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
    parser.add_argument('--algorithm', type=str, default='FedAPA', help='name of algorithm')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--prefix', type=str, default='test1', help='prefix of result filename')
    parser.add_argument('--modelName', type=str, default='LeNet5', help='training model')
    parser.add_argument('--benignAlone',action='store_true', help='if every client has benignAlone')

    args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cuda")  # 直接使用 CPU 更快

    # set_random_seed(42)  # 设置随机种子

    config_file_name = f'./configs/{args.algorithm}.yaml'
    with open(config_file_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("=====================      Config      =====================")
    print(f"Algorithm: {args.algorithm}, Dataset: {args.dataset}, Model: {args.modelName}, Device:{args.device}")
    if args.debug:
        print("debug")
    if args.benignAlone:
        print("benign alone")
    for key, value in config.items():
        print(key + ": " + str(value))
    print("===================== Experiment Begin =====================")

    if args.debug:
        args.num_repeat_times = 1
        # args.num_repeat_times = config["num_repeat_times"]
    else:
        args.num_repeat_times = config["num_repeat_times"]
    args.distribution = config["distribution"]
    args.alpha = config["alpha"]

    args.server = config["server"]
    args.client = config["client"]

    run(args)
