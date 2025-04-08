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

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    num_clients = args.client["num_clients"]  # 用户数
    num_classes = len(np.unique(trainset.targets))  # 数据集的类数

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, niid, balance, partition)

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
        X = [[] for _ in range(num_clients)]
        y = [[] for _ in range(num_clients)]
        statistic = [[] for _ in range(num_clients)]

        dataset_content, dataset_label = data
        # guarantee that each client must have at least one batch of data for testing.
        least_samples = int(min(batch_size / (1 - train_ratio), len(dataset_label) / num_clients / 2))

        dataidx_map = {}

        if not niid:
            partition = 'pat'
            class_per_client = num_classes

        if partition == 'pat':
            idxs = np.array(range(len(dataset_label)))
            idx_for_each_class = []
            for i in range(num_classes):
                idx_for_each_class.append(idxs[dataset_label == i])

            class_num_per_client = [class_per_client for _ in range(num_clients)]
            for i in range(num_classes):
                selected_clients = []
                for client in range(num_clients):
                    if class_num_per_client[client] > 0:
                        selected_clients.append(client)
                if len(selected_clients) == 0:
                    break
                selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

                num_all_samples = len(idx_for_each_class[i])
                num_selected_clients = len(selected_clients)
                num_per = num_all_samples / num_selected_clients
                if balance:
                    num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
                else:
                    num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                    num_selected_clients - 1).tolist()
                num_samples.append(num_all_samples - sum(num_samples))

                idx = 0
                for client, num_sample in zip(selected_clients, num_samples):
                    if client not in dataidx_map.keys():
                        dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                    else:
                        dataidx_map[client] = np.append(dataidx_map[client],
                                                        idx_for_each_class[i][idx:idx + num_sample], axis=0)
                    idx += num_sample
                    class_num_per_client[client] -= 1

        elif partition == "dir":
            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
            min_size = 0
            K = num_classes
            N = len(dataset_label)

            try_cnt = 1
            while min_size < least_samples:
                if try_cnt > 1:
                    print(
                        f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

                idx_batch = [[] for _ in range(num_clients)]
                for k in range(K):
                    idx_k = np.where(dataset_label == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                    proportions = np.array(
                        [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                try_cnt += 1

            for j in range(num_clients):
                dataidx_map[j] = idx_batch[j]

        elif partition == 'exdir':
            r'''This strategy comes from https://arxiv.org/abs/2311.03154
            See details in https://github.com/TsingZ0/PFLlib/issues/139

            This version in PFLlib is slightly different from the original version 
            Some changes are as follows:
            n_nets -> num_clients, n_class -> num_classes
            '''
            C = class_per_client

            '''The first level: allocate labels to clients
            clientidx_map (dict, {label: clientidx}), e.g., C=2, num_clients=5, num_classes=10
                {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}
            '''
            min_size_per_label = 0
            # You can adjust the `min_require_size_per_label` to meet you requirements
            min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
            if min_require_size_per_label < 1:
                raise ValueError
            clientidx_map = {}
            while min_size_per_label < min_require_size_per_label:
                # initialize
                for k in range(num_classes):
                    clientidx_map[k] = []
                # allocate
                for i in range(num_clients):
                    labelidx = np.random.choice(range(num_classes), C, replace=False)
                    for k in labelidx:
                        clientidx_map[k].append(i)
                min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])

            '''The second level: allocate data idx'''
            dataidx_map = {}
            y_train = dataset_label
            min_size = 0
            min_require_size = 10
            K = num_classes
            N = len(y_train)
            print("\n*****clientidx_map*****")
            print(clientidx_map)
            print("\n*****Number of clients per label*****")
            print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

            # ensure per client' sampling size >= min_require_size (is set to 10 originally in [3])
            while min_size < min_require_size:
                idx_batch = [[] for _ in range(num_clients)]
                # for each class in the dataset
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                    # Balance
                    # Case 1 (original case in Dir): Balance the number of sample per client
                    proportions = np.array(
                        [p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in
                         enumerate(zip(proportions, idx_batch))])
                    # Case 2: Don't balance
                    # proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    # process the remainder samples
                    '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                    There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                    In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                    '''
                    if proportions[-1] != len(idx_k):
                        for w in range(clientidx_map[k][-1], num_clients - 1):
                            proportions[w] = len(idx_k)
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(num_clients):
                np.random.shuffle(idx_batch[j])
                dataidx_map[j] = idx_batch[j]

        else:
            raise NotImplementedError

        # assign data
        for client in range(num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]

            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client] == i))))

        del data
        # gc.collect()

        for client in range(num_clients):
            print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
            print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            print("-" * 50)

        return X, y, statistic

    # num_clients = args.client["num_clients"]  # 用户数
    # num_classes = len(np.unique(trainset.targets))  # 数据集的类数
    # if (args.benignAlone):
    #     partition_num_classes =num_classes-1
    # else:
    #     partition_num_classes=num_classes
    # while True:
    #     """ 获得数据集配置，生成比例分布 """
    #     if args.distribution == "iid":
    #         num_classes_per_client = 10  # 每个用户的拥有的数据类数
    #         partition = generate_partition(num_clients, num_classes,
    #                                        num_classes_per_client=num_classes_per_client,
    #                                        is_balanced=True)
    #
    #     elif args.distribution == "non-balanced":
    #         num_classes_per_client = args.client["num_classes_per_client"]  # 每个用户的拥有的数据类数
    #         partition = generate_partition(num_clients, num_classes,
    #                                        num_classes_per_client=num_classes_per_client,
    #                                        is_balanced=False)
    #
    #     elif args.distribution == "dirichlet":
    #         alpha = args.alpha
    #         partition = generate_partition(num_clients, partition_num_classes,
    #                                        is_dirichlet=True, alpha=alpha)
    #     else:
    #         raise NotImplementedError
    #
    #     """ 生成数据 subset 以及统计 """
    #     if (args.benignAlone):
    #         train_subsets, statistic = generate_subsets_Android(
    #             trainset, num_classes, num_clients, partition, is_return_stats=True)
    #         test_subsets, _ = generate_subsets_Android(
    #             testset, num_classes, num_clients, partition, is_return_stats=False)
    #     else:
    #         # train_subsets, statistic = generate_subsets(
    #         #     trainset, num_classes, num_clients, partition, is_return_stats=True)
    #         # test_subsets, _ = generate_subsets(
    #         #     testset, num_classes, num_clients, partition, is_return_stats=False)
    #
    #     if all([len(subset) > 128 for subset in train_subsets]) and all([len(subset) > 2 for subset in test_subsets]):
    #     # if all([len(subset) > 0 for subset in train_subsets]) and all([len(subset) > 0 for subset in test_subsets]):
    #         break
    #     print("partition is not good, regenerate...")
    #
    #
    # """ 保存数据集 """
    # train_config = {
    #     "dataset": args.dataset,
    #     "distribution": args.distribution,
    #     "alpha": args.alpha,
    #     "server": args.server,
    #     "client": args.client,
    # }
    # save_data(train_config, statistic, train_subsets, test_subsets, args.paths["config"], args.paths["train"], args.paths["test"])
    #
    # return train_config, statistic
