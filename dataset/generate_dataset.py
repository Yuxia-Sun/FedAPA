import os.path
from collections import defaultdict

import numpy as np
import ujson

import torchvision
from torch.utils.data import Subset
from torchvision.transforms import transforms

from dataset.data_utils import check, save_data
from torch.utils.data.dataset import Dataset
import torch
from PIL import Image
class KronoDroidDataset(Dataset):
    def __init__(self, root="/root/autodl-fs/kronoDroid", dataType='train'):
        # load Data
        self.data_source = root + '/{}_time_junyihua.npy'.format(dataType)  # 每个样本大小是32*32
        print(self.data_source)
        self.data = np.load(self.data_source, allow_pickle=True).item()
        self.x = self.data['X']
        self.targets = self.data['Y'].astype('int64')
        self.hash = self.data['sha256']
        self.time = self.data['time']
        self.malware = self.data['malware']
        self.family = self.data['familyName']
        self.transform = transforms.Compose(
            [
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img = np.stack((self.x[index],) * 3, axis=-1)
        label = self.targets[index]
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        return img,label

    def __len__(self):
        return len(self.x)
#
# def generate_partition(num_clients, num_classes,
#                        is_dirichlet=False, is_balanced=False,
#                        alpha=0.1, num_classes_per_client=2):
#     """ 生成数据集分布 """
#     if is_dirichlet:
#         partition = np.random.dirichlet(alpha=[alpha] * num_clients, size=num_classes)
#     else:
#         count_classes_each_client = [num_classes_per_client for _ in range(num_clients)]  # 每个用户剩余的样本类数
#
#         num_clients_each_class = int(np.ceil(num_clients * num_classes_per_client / num_classes))  # 一类样本被分配到的用户数
#
#         partition = np.zeros(shape=(num_classes, num_clients))
#         for cls in range(num_classes):
#             """ 添加共享同一类的用户 """
#             selected_clients = []
#             for client_id in range(num_clients):
#                 if count_classes_each_client[client_id] > 0 and len(selected_clients) < num_clients_each_class:
#                     selected_clients.append(client_id)
#                     count_classes_each_client[client_id] -= 1
#
#             if is_balanced:
#                 proportions = np.full(shape=num_clients_each_class, fill_value=1)
#             else:
#                 proportions = np.random.uniform(low=0.3, high=0.7, size=num_clients_each_class)  # 生成对应用户数个随机数
#             proportions = (proportions / np.sum(proportions)).tolist()
#
#             """ 给到对应的用户 """
#             for client_id in selected_clients:
#                 prop = proportions.pop()
#                 partition[cls][client_id] = prop
#
#     return partition
#
#
# def generate_subsets(dataset, num_classes, num_clients, partitions, is_return_stats=False):
#     """ 生成每个用户的数据集 """
#     samples, labels = dataset.data, dataset.targets  # 原始数据集的样本和标签
#
#     idx_each_class = []  # 每个类的所有样本索引
#     num_each_class = []  # 每个类的样本数
#     """ 获得每个类的样本索引和样本数 """
#     for cls in range(num_classes):
#         indices = np.where(np.array(labels) == cls)[0]  # 标签为 cls 的所有样本的索引
#         idx_each_class.append(indices)
#         num_each_class.append(len(indices))
#
#     idx_sample_client = defaultdict(list)  # 每个用户的数据集的索引
#     statistics = [{} for _ in range(num_clients)]  # 数据记录
#
#     """ 分配数据集序号 """
#     for cls in range(num_classes):
#         cls_array = partitions[cls]
#         selected_clients = cls_array.nonzero()[0]  # 共享该标签数据样本的用户
#         for client in selected_clients:
#             num_samples_client = int(cls_array[client] * num_each_class[cls])  # 分配给这个用户的样本量
#             if num_samples_client < 1:
#                 continue
#             idx_sample_client[client].extend(idx_each_class[cls][:num_samples_client])
#             idx_each_class[cls] = idx_each_class[cls][num_samples_client:]
#             # 字典的 key 可以是 int 或者 string，但是 json 中 key 只能是 string，因此从 json 文件中读到的 key 都是 string
#             statistics[client][str(cls)] = num_samples_client
#
#     """ 返回每个用户的数据集 """
#     return ([Subset(dataset, idx_sample_client[client]) for client in range(num_clients)],
#             statistics if is_return_stats else None)
#
#
# def generate_paths(dataset, distribution):
#     """ 生成数据集及其信息的本地保存路径 """
#     rawdata_path = os.path.join("dataset/rawdata/", dataset)
#     data_path = os.path.join(rawdata_path, distribution)
#     config_path = os.path.join(data_path, "config.json")
#     train_path = os.path.join(data_path, "trainsets.pth")
#     test_path = os.path.join(data_path, "testsets.pth")
#
#     return {
#         "rawdata": rawdata_path,
#         "data": data_path,
#         "config": config_path,
#         "train": train_path,
#         "test": test_path,
#     }
#
#
# def generate_rawdata(paths, dataset):
#     """ 下载原始数据集 """
#     if dataset == "mnist":
#         """ 样本预处理 """
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])
#
#         trainset = torchvision.datasets.MNIST(
#             root=paths["rawdata"], train=True, download=True, transform=transform)
#         testset = torchvision.datasets.MNIST(
#             root=paths["rawdata"], train=False, download=True, transform=transform)
#
#     elif dataset == "fmnist":
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])
#
#         trainset = torchvision.datasets.FashionMNIST(
#             root=paths["rawdata"], train=True, download=True, transform=transform)
#         testset = torchvision.datasets.FashionMNIST(
#             root=paths["rawdata"], train=False, download=True, transform=transform)
#
#     elif dataset == "cifar10":
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#
#         trainset = torchvision.datasets.CIFAR10(
#             root=paths["rawdata"], train=True, download=True, transform=transform)
#         testset = torchvision.datasets.CIFAR10(
#             root=paths["rawdata"], train=False, download=True, transform=transform)
#
#     elif dataset == "cifar100":
#         """ 样本预处理 """
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#
#         trainset = torchvision.datasets.CIFAR100(
#             root=paths["rawdata"], train=True, download=True, transform=transform)
#         testset = torchvision.datasets.CIFAR100(
#             root=paths["rawdata"], train=False, download=True, transform=transform)
#
#     elif dataset == "kronoDroid":
#         """ 样本预处理 """
#         transform = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#         ])
#         trainset = KronoDroidDataset(dataType='train')
#         testset = KronoDroidDataset(dataType='test')
#
#         # trainset = torchvision.datasets.CIFAR100(
#         #     root=paths["rawdata"], train=True, download=True, transform=transform)
#         # testset = torchvision.datasets.CIFAR100(
#         #     root=paths["rawdata"], train=False, download=True, transform=transform)
#
#     else:
#         raise NotImplementedError
#
#     return trainset, testset
#
#
# def generate_dataset(args):
#     args.paths = generate_paths(args.dataset, args.distribution)
#
#     if not os.path.exists(args.paths["data"]):
#         os.makedirs(args.paths["data"])
#
#     """ 检查是否需要重新生成数据集 """
#     if check(args, args.paths["config"]):
#         with open(args.paths["config"], "r") as f:
#             config = ujson.load(f)
#
#         train_config = config["train_config"]
#         statistic = config["statistic"]
#         return train_config, statistic
#
#     print("Generating Dataset...")
#
#     """ 生成原始数据集 """
#     trainset, testset = generate_rawdata(args.paths, args.dataset)
#
#     num_clients = args.client["num_clients"]  # 用户数
#     num_classes = len(np.unique(trainset.targets))  # 数据集的类数
#
#     while True:
#         """ 获得数据集配置，生成比例分布 """
#         if args.distribution == "iid":
#             num_classes_per_client = 10  # 每个用户的拥有的数据类数
#             partition = generate_partition(num_clients, num_classes,
#                                            num_classes_per_client=num_classes_per_client,
#                                            is_balanced=True)
#
#         elif args.distribution == "non-balanced":
#             num_classes_per_client = args.client["num_classes_per_client"]  # 每个用户的拥有的数据类数
#             partition = generate_partition(num_clients, num_classes,
#                                            num_classes_per_client=num_classes_per_client,
#                                            is_balanced=False)
#
#         elif args.distribution == "dirichlet":
#             alpha = args.alpha
#             partition = generate_partition(num_clients, num_classes,
#                                            is_dirichlet=True, alpha=alpha)
#         else:
#             raise NotImplementedError
#
#         """ 生成数据 subset 以及统计 """
#         train_subsets, statistic = generate_subsets(
#             trainset, num_classes, num_clients, partition, is_return_stats=True)
#         test_subsets, _ = generate_subsets(
#             testset, num_classes, num_clients, partition, is_return_stats=False)
#
#         if all([len(subset) > 128 for subset in train_subsets]):
#             break
#         print("partition is not good, regenerate...")
#
#     """ 保存数据集 """
#     train_config = {
#         "dataset": args.dataset,
#         "distribution": args.distribution,
#         "alpha": args.alpha,
#         "server": args.server,
#         "client": args.client,
#     }
#     save_data(train_config, statistic, train_subsets, test_subsets, args.paths["config"], args.paths["train"], args.paths["test"])
#
#     return train_config, statistic

def generate_partition(num_clients, num_classes,
                       is_dirichlet=False, is_balanced=False,
                       alpha=0.1, num_classes_per_client=2):
    """ 生成数据集分布 """
    if is_dirichlet:
        partition = np.random.dirichlet(alpha=[alpha] * num_clients, size=num_classes)
    else:
        count_classes_each_client = [num_classes_per_client for _ in range(num_clients)]  # 每个用户剩余的样本类数

        num_clients_each_class = int(np.floor(num_clients * num_classes_per_client / num_classes))  # 一类样本被分配到的用户数

        partition = np.zeros(shape=(num_classes, num_clients))
        for cls in range(num_classes):
            """ 添加共享同一类的用户 """
            selected_clients = []
            for client_id in range(num_clients):
                if count_classes_each_client[client_id] > 0 and len(selected_clients) < num_clients_each_class:
                    selected_clients.append(client_id)
                    count_classes_each_client[client_id] -= 1

            if is_balanced:
                proportions = np.full(shape=num_clients_each_class, fill_value=1)
            else:
                proportions = np.random.uniform(low=0.3, high=0.7, size=num_clients_each_class)  # 生成对应用户数个随机数
            proportions = (proportions / np.sum(proportions)).tolist()

            """ 给到对应的用户 """
            for client_id in selected_clients:
                prop = proportions.pop()
                partition[cls][client_id] = prop

    return partition

def generate_subsets(dataset, num_classes, num_clients, partitions, is_return_stats=False):
    """ 生成每个用户的数据集 """
    samples, labels = dataset.data, dataset.targets  # 原始数据集的样本和标签

    idx_each_class = []  # 每个类的所有样本索引
    num_each_class = []  # 每个类的样本数
    """ 获得每个类的样本索引和样本数 """
    for cls in range(num_classes):
        indices = np.where(np.array(labels) == cls)[0]  # 标签为 cls 的所有样本的索引
        idx_each_class.append(indices)
        num_each_class.append(len(indices))

    idx_sample_client = defaultdict(list)  # 每个用户的数据集的索引
    statistics = [{} for _ in range(num_clients)]  # 数据记录

    """ 分配数据集序号 """
    for cls in range(num_classes):
        cls_array = partitions[cls]
        selected_clients = cls_array.nonzero()[0]  # 共享该标签数据样本的用户
        for client in selected_clients:
            num_samples_client = int(cls_array[client] * num_each_class[cls])  # 分配给这个用户的样本量
            if num_samples_client < 1:
                continue
            idx_sample_client[client].extend(idx_each_class[cls][:num_samples_client])
            idx_each_class[cls] = idx_each_class[cls][num_samples_client:]
            # 字典的 key 可以是 int 或者 string，但是 json 中 key 只能是 string，因此从 json 文件中读到的 key 都是 string
            statistics[client][str(cls)] = num_samples_client

    """ 返回每个用户的数据集 """
    return ([Subset(dataset, idx_sample_client[client]) for client in range(num_clients)],
            statistics if is_return_stats else None)


def generate_subsets_Android(dataset, num_classes, num_clients, partitions, is_return_stats=False):
    print("每个用户都有良性！！！！")
    """ 生成每个用户的数据集 """
    samples, labels = dataset.data, dataset.targets  # 原始数据集的样本和标签

    idx_each_class = []  # 每个类的所有样本索引
    num_each_class = []  # 每个类的样本数
    """ 获得每个类的样本索引和样本数 """
    for cls in range(num_classes):
        indices = np.where(np.array(labels) == cls)[0]  # 标签为 cls 的所有样本的索引
        idx_each_class.append(indices)
        num_each_class.append(len(indices))

    idx_sample_client = defaultdict(list)  # 每个用户的数据集的索引
    statistics = [{} for _ in range(num_clients)]  # 数据记录

    """ 分配数据集序号 """
    for cls in range(num_classes-1):
        cls_array = partitions[cls]
        selected_clients = cls_array.nonzero()[0]  # 共享该标签数据样本的用户
        for client in selected_clients:
            num_samples_client = int(cls_array[client] * num_each_class[cls+1])  # 分配给这个用户的样本量
            if num_samples_client < 1:
                continue
            idx_sample_client[client].extend(idx_each_class[cls+1][:num_samples_client])
            idx_each_class[cls+1] = idx_each_class[cls+1][num_samples_client:]
            # 字典的 key 可以是 int 或者 string，但是 json 中 key 只能是 string，因此从 json 文件中读到的 key 都是 string
            statistics[client][str(cls+1)] = num_samples_client

    # 添加良性的
    #     求和每个客户端所有恶意的总数
    malware_sum = len([x for x in labels if x != 0])
    for client in range(num_clients):
        malware_clent_sum = len(idx_sample_client[client])
        benign_partition = malware_clent_sum/malware_sum
        num_samples_client = int(benign_partition * num_each_class[0])
        idx_sample_client[client].extend(np.random.choice(idx_each_class[0],num_samples_client,replace=False))
        # idx_sample_client[client].extend(random.sample(idx_each_class[0],num_samples_client))
        statistics[client][str(0)] = num_samples_client
        #     求每个客户端恶意总数展全部恶意的比例

    #       比例*良性


    """ 返回每个用户的数据集 """
    return ([Subset(dataset, idx_sample_client[client]) for client in range(num_clients)],
            statistics if is_return_stats else None)


def generate_paths(dataset, distribution):
    """ 生成数据集及其信息的本地保存路径 """
    rawdata_path = os.path.join("dataset/rawdata/", dataset)
    data_path = os.path.join(rawdata_path, distribution)
    config_path = os.path.join(data_path, "config.json")
    train_path = os.path.join(data_path, "trainsets.pth")
    test_path = os.path.join(data_path, "testsets.pth")

    return {
        "rawdata": rawdata_path,
        "data": data_path,
        "config": config_path,
        "train": train_path,
        "test": test_path,
    }


def generate_rawdata(paths, dataset):
    """ 下载原始数据集 """
    if dataset == "mnist":
        """ 样本预处理 """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        trainset = torchvision.datasets.MNIST(
            root=paths["rawdata"], train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root=paths["rawdata"], train=False, download=True, transform=transform)

    elif dataset == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        trainset = torchvision.datasets.FashionMNIST(
            root=paths["rawdata"], train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(
            root=paths["rawdata"], train=False, download=True, transform=transform)

    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=paths["rawdata"], train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=paths["rawdata"], train=False, download=True, transform=transform)

    elif dataset == "cifar100":
        """ 样本预处理 """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR100(
            root=paths["rawdata"], train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(
            root=paths["rawdata"], train=False, download=True, transform=transform)

    elif dataset == "kronoDroid":
        """ 样本预处理 """
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        trainset = KronoDroidDataset(dataType='train')
        testset = KronoDroidDataset(dataType='test')

        # trainset = torchvision.datasets.CIFAR100(
        #     root=paths["rawdata"], train=True, download=True, transform=transform)
        # testset = torchvision.datasets.CIFAR100(
        #     root=paths["rawdata"], train=False, download=True, transform=transform)

    else:
        raise NotImplementedError

    return trainset, testset


def generate_dataset(args):
    args.paths = generate_paths(args.dataset, args.distribution)

    if not os.path.exists(args.paths["data"]):
        os.makedirs(args.paths["data"])

    """ 检查是否需要重新生成数据集 """
    if check(args, args.paths["config"]):
        with open(args.paths["config"], "r") as f:
            config = ujson.load(f)

        train_config = config["train_config"]
        statistic = config["statistic"]
        return train_config, statistic

    print("Generating Dataset...")

    """ 生成原始数据集 """
    trainset, testset = generate_rawdata(args.paths, args.dataset)

    num_clients = args.client["num_clients"]  # 用户数
    num_classes = len(np.unique(trainset.targets))  # 数据集的类数
    if (args.benignAlone):
        partition_num_classes =num_classes-1
    else:
        partition_num_classes=num_classes
    while True:
        """ 获得数据集配置，生成比例分布 """
        if args.distribution == "iid":
            num_classes_per_client = 10  # 每个用户的拥有的数据类数
            partition = generate_partition(num_clients, num_classes,
                                           num_classes_per_client=num_classes_per_client,
                                           is_balanced=True)

        elif args.distribution == "non-balanced":
            num_classes_per_client = args.client["num_classes_per_client"]  # 每个用户的拥有的数据类数
            partition = generate_partition(num_clients, num_classes,
                                           num_classes_per_client=num_classes_per_client,
                                           is_balanced=False)

        elif args.distribution == "dirichlet":
            alpha = args.alpha
            partition = generate_partition(num_clients, partition_num_classes,
                                           is_dirichlet=True, alpha=alpha)
        else:
            raise NotImplementedError

        """ 生成数据 subset 以及统计 """
        if (args.benignAlone):
            train_subsets, statistic = generate_subsets_Android(
                trainset, num_classes, num_clients, partition, is_return_stats=True)
            test_subsets, _ = generate_subsets_Android(
                testset, num_classes, num_clients, partition, is_return_stats=False)
        else:
            train_subsets, statistic = generate_subsets(
                trainset, num_classes, num_clients, partition, is_return_stats=True)
            test_subsets, _ = generate_subsets(
                testset, num_classes, num_clients, partition, is_return_stats=False)

        if all([len(subset) > 128 for subset in train_subsets]) and all([len(subset) > 2 for subset in test_subsets]):
        # if all([len(subset) > 0 for subset in train_subsets]) and all([len(subset) > 0 for subset in test_subsets]):
            break
        print("partition is not good, regenerate...")


    """ 保存数据集 """
    train_config = {
        "dataset": args.dataset,
        "distribution": args.distribution,
        "alpha": args.alpha,
        "server": args.server,
        "client": args.client,
    }
    save_data(train_config, statistic, train_subsets, test_subsets, args.paths["config"], args.paths["train"], args.paths["test"])

    return train_config, statistic
