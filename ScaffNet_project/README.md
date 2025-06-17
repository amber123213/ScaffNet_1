# ScaffNet

ScaffNet 是一个用于交通流量预测的深度学习模型，基于图卷积网络和循环神经网络。

## 项目结构

```
ScaffNet_project/
├── configs/             # 配置文件
├── data/                # 数据目录
├── lib/                 # 工具库
├── model/               # 模型实现
│   ├── AGCN.py          # 自适应图卷积网络
│   ├── AGCRNCell.py     # AGCRN单元
│   ├── BasicTrainer.py  # 训练器
│   └── scaffnet.py      # ScaffNet模型
└── scripts/             # 脚本
    └── run_experiment.py # 主运行脚本
```

## 使用方法

1. 准备数据：将PEMS04或PEMS08数据放入data目录
2. 配置参数：修改configs/PEMSD4_ScaffNet.conf文件
3. 运行训练：

```bash
cd ScaffNet_project
python scripts/run_experiment.py
```

## 模型介绍

ScaffNet模型结合了自适应图卷积网络和门控循环单元，能够同时捕捉交通数据的空间和时间依赖关系。模型的主要特点包括：

- 自适应邻接矩阵：通过节点嵌入学习最优的图结构
- 门控机制：使用类似GRU的门控机制处理时间序列
- 多步预测：支持多步时间序列预测 