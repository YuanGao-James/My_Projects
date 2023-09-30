MAS-Signature
==========

### 安装
`pip install -r requirements`

### 目标
- 通过改进MASignature.signature，使数据信息分布在整个曲线上
  - 参考：[Deep Signature Transforms](https://papers.nips.cc/paper/2019/file/d2cdf047a6674cef251d56544a3cf029-Paper.pdf) Fig5
- 通过设置不同的超参数使得Signature可以衡量待拟合参数的差异程度
  - 通过改进评价方式（目前是L1，L2），使得signature可以衡量参数的优劣
  - 通过改进signature的数据预处理方法，signature的超参数或者其他的路径签名方法使得可以衡量不同超参数的优劣

***详细可参考`./experiment/signature analysis.ipynb`***

### 超参数解释
```python
class Args:
    batch_size = 100            ## 模型训练batch_size 大小
    epochs = 500                ## 模型训练epoch
    input_shape = (10, 1)       ## 模型输入序列长度
    output_shape = (10, 1)      ## 模型输出序列长度
    lr = 0.001                  ## 学习率
    mu1 = 100.0                 ## -----------
    mu2 = 0.01                  #｜
    nu1 = 0.1                   #｜
    nu2 = 0.1                   #｜MAS 超参数
    rho = 0.15                  #｜参考./reference/MAS_GAN_note.pdf
    sigma_i = 0                 #｜
    sigma_p = 1                 #｜
    num_samples = [7, 5, 3]     #｜-----------
    param_b = 1.0               ## 待拟合的参数b
    param_c = 1.0               ## 待拟合的参数c
    param_d = 0.5               ## 待拟合的参数d
    param_init = True           ## 训练时是否初始化超参数或者使用随机超参数
    param_b_init = 0.6          ## 初始化超参数b，在param_init = True时生效
    param_c_init = 0.6          ## 初始化超参数c，在param_init = True时生效
    param_d_init = 0.1          ## 初始化超参数d，在param_init = True时生效
    project = "MAS"             ## 工程名称，用于指定clearml的log保存地址
    name = "test"               ## 实验名称，用于指定clearml的log保存地址
    logs_dir = "logs"           ## 实验log本地保存地址
    device = 'cuda'             ## 训练时使用的设备
```

### 各部分文件功能解释
```bash
--mas-gan
|--experiment                               ## 实验文件夹
|  |-- signature analysis.ipynb             ## 路径签名分析实验
|--model                                    ## 模型文件夹
|  |-- dataset.py                           ## 数据集生成程序
|  |-- maslayer.py                          ## MAS 生成器
|  |-- massignature.py                      ## MAS 生成器 + Signature 判别器训练过程
|--tests                                    ## 单元测试，测试各部分工作是否征程
|--main.py                                  ## 主函数入口
```