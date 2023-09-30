import configparser
import random
import sys
import numpy as np
import logging

import matplotlib.pyplot as plt
import torch

# pylint: disable=wrong-import-position, invalid-name，logging-fstring-interpolation
##!!! 这里把路径换成自己工作目录
sys.path.append("/home/ai/project/mas-gan")
from model.dataset import get_dataset
from model.maslayer import Generator
from model.massignature import MASignature
from hyperparameter import Args

## 设置随机种子
manualSeed = 0
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

logger = logging.getLogger(__name__)

def diffScore(model: MASignature, in_price: torch.Tensor, in_price_star: torch.Tensor, \
        in_sigma: torch.Tensor, target_sig: torch.Tensor, b: float, c: float, d: float):
    """计算在给定的b, c, d参数下模型的输出与b=1, c=1, d=0.5下的结果的差距

    Args:
        model (MASignature): 模型
        in_price (torch.Tensor): 输入price
        in_price_star (torch.Tensor): 输入price-star
        target_sig (torch.Tensor): 在b=1, c=1, d=0.5时目标signature结果
        b (float): 目前的b
        c (float): 目前的c
        d (float): 目前的d
    """
    for name, param in model.generator.named_parameters():
        if name == "mas_cell.noise_traders.param_b":
            torch.nn.init.constant_(param, b)
        if name == "mas_cell.noise_traders.param_c":
            torch.nn.init.constant_(param, c)
        if name == "mas_cell.technical_traders.param_d":
            torch.nn.init.constant_(param, d)
    model.generator.eval()
    plt.figure(figsize=(15, 5))
    gen_price, _, _ = model.generator(in_price, in_price_star, in_sigma)
    plt.subplot(1, 3, 1)
    plt.plot(in_price[0])
    plt.subplot(1, 3, 2)
    plt.plot(gen_price.detach().numpy()[0])
    gen_price_sig = model.signature(gen_price)
    gen_price_sig = gen_price_sig.detach().numpy()
    target_sig = target_sig.detach().numpy()
    plt.subplot(1, 3, 3)
    plt.plot(gen_price_sig[0])
    plt.show()
    logger.info(f"[*] b={b}, c={c}, d={d}")
    logger.info(f"L1 diff: {np.mean(np.sum((gen_price_sig - target_sig), axis=1), axis=0)}")
    logger.info(f"L2 diff: {np.mean(np.sum((gen_price_sig - target_sig) ** 2, axis=1), axis=0)}")

def main():
    ## 超参数
    args=Args()
    ## 根据args获取数据
    in_price, in_price_star, in_sigma, out_price = get_dataset(args)
    logger.info(f"输入price, shape: {in_price.shape}") # 输入price
    logger.info(f"输入price_star, shape: {in_price_star.shape}") # 输入price_star
    logger.info(f"输入sigma, shape: {in_sigma.shape}") # 输入sigma
    logger.info(f"输出价格: {out_price.shape}") # 输出价格

    model = MASignature(args)
    sig_in_p = model.signature(in_price) ## 对输入价格进行路径签名
    logger.info(f"输入价格路径签名后的shape: {sig_in_p.shape}")
    # plt.subplot
    target_sig = model.get_lr_sigs_pred(in_price, out_price) ## b=1, c=1, d=0.5 时目标signature
    logger.info(target_sig.shape)
    ## 计算b=0.6, c=0.6, d=0.1时模型的输出结果与b=1，c=1, d=0.5的差距
    diffScore(model, in_price, in_price_star, in_sigma, target_sig, b=0.6, c=0.6, d=0.1)
    ## 计算b=0.9, c=0.9, d=0.4时模型的输出结果与b=1，c=1, d=0.5的差距
    diffScore(model, in_price, in_price_star, in_sigma, target_sig, b=0.9, c=0.9, d=0.4)
    ## 计算b=1.0, c=1.0, d=0.5时模型的输出结果与b=1，c=1, d=0.5的差距
    diffScore(model, in_price, in_price_star, in_sigma, target_sig, b=1.0, c=1.0, d=0.5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
