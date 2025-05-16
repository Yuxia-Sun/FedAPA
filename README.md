# FedAPA: Server-side Gradient-Based Adaptive Personalized Aggregation for Federated Learning on Heterogeneous Data

## Description
Personalized federated learning (PFL) tailors models to clients' unique data distributions while preserving privacy. However, existing aggregation-weight-based PFL methods often struggle with heterogeneous data, facing challenges in accuracy, computational efficiency, and communication overhead. We propose FedAPA, a novel PFL method featuring a server-side, gradient-based adaptive aggregation strategy to generate personalized models, by updating aggregation weights based on gradients of client-parameter changes with respect to the aggregation weights in a centralized manner. FedAPA guarantees theoretical convergence and achieves superior accuracy and computational efficiency compared to 10 PFL competitors across three datasets, with competitive communication overhead.

> Yuxia Sun, Aoxiang Sun, Siyi Pan, Zhixiao Fu, and Jingcai Guo.2025. FedAPA: Server-side Gradient-Based Adaptive Personalized Aggregation for Federated Learning on Heterogeneous Data. In Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025), 2025. IJCAI.(Accepted)

## Supplementary
To access our proof of convergence, visit: [link](https://github.com/Yuxia-Sun/FedAPA/blob/main/FedAPA_pf_cvg.pdf)
