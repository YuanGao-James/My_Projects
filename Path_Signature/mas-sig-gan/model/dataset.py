"""
@Author     dizs@using.ai
@Date       2022-04-27
@Describe   利用MAS制作数据集
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
# pylint: disable = wrong-import-position
from model.maslayer import Generator

class MASDataset(Dataset):
    """MAS dataset"""
    def __init__(self, in_price: Tensor, in_price_star: Tensor, in_sigma: Tensor, out_price: Tensor) -> None:
        super().__init__()
        self.in_price = in_price
        self.in_price_star = in_price_star
        self.in_sigma = in_sigma
        self.out_price = out_price

    def __len__(self):
        return len(self.in_price)

    def __getitem__(self, index):
        return (self.in_price[index], self.in_price_star[index], self.in_sigma[index]), self.out_price[index]


def get_dataset(args: dict, batch_size: int=100, gen_length: int=50, stable_len: int=50):
    """生成训练数据集

    Args:
        args (dict): generator 初始化参数
        batch_size (int, optional): 同时生成数据量的大小. Defaults to 50.
        gen_length (int, optional): 生成数据的长度. Defaults to 100.
        stable_len (int, optional): 丢弃的一段时间，由于technical trader的存在，刚开始的数据会不稳定. 
            Defaults to 50.
    Returns:
        MASDataset: 生成的数据集
    """
    generator = Generator(args.num_samples, gen_length + stable_len, args.rho, args.sigma_i, args.sigma_p, args.mu1, args.mu2, args.nu1, args.nu2)
    
    ## 初始price, price_star, sigma，参考./reference/MAS_GAN_note.pdf
    price = torch.ones((batch_size, 1, 1), dtype=torch.float32) * 100.
    p_star = torch.ones((batch_size, 1, 1), dtype=torch.float32) * 100.
    sigma = torch.ones((batch_size, 1, 1), dtype=torch.float32) * 0.01

    ## 设置生成数据时的超参数，即待拟合的目标
    for name, param in generator.named_parameters():
        if name == "mas_cell.noise_traders.param_b":
            torch.nn.init.constant_(param, args.param_b)
        if name == "mas_cell.noise_traders.param_c":
            torch.nn.init.constant_(param, args.param_c)
        if name == "mas_cell.technical_traders.param_d":
            torch.nn.init.constant_(param, args.param_d)

    ## 生成的数据，目前数据比较长，是（batch_size， gen_length， 1）
    price_t, sigma_t, p_star_t = generator(price, p_star, sigma)
    price_g = price_t.detach().numpy()
    # price_star_g = sigma_t.detach().numpy()
    # sigma_t_g = p_star_t.detach().numpy()
    price_star_g = p_star_t.detach().numpy()
    sigma_t_g = sigma_t.detach().numpy()

    # Z-Score 中心化处理
    # price_g = (price_g - np.mean(price_g, axis=0, keepdims=True)) / \
    #     np.std(price_g, axis=0, keepdims=True)
    # price_star_g = (price_star_g - np.mean(price_star_g, axis=0, keepdims=True)) / \
    #     np.std(price_star_g, axis=0, keepdims=True)
    # sigma_t_g = (sigma_t_g - np.mean(sigma_t_g, axis=0, keepdims=True)) / \
    #     np.std(sigma_t_g, axis=0, keepdims=True)

    # 根据生成的数据划分为输入，输出对；
    inputs_price = []
    inputs_price_star = []
    inputs_sigma = []
    outputs = []
    for i in range(batch_size):
        for j in range(gen_length-20):
            inputs_price.append(price_g[i, j:j+10, :])
            inputs_price_star.append(price_star_g[i, j:j+10, :])
            inputs_sigma.append(sigma_t_g[i, j:j+10, :])
            outputs.append(price_g[i, j+10:j+20, :])

    in_price = np.concatenate(np.expand_dims(inputs_price, 0), axis=0)
    in_price_star = np.concatenate(np.expand_dims(inputs_price_star, 0), axis=0)
    in_sigma = np.concatenate(np.expand_dims(inputs_sigma, 0), axis=0)
    out_price = np.concatenate(np.expand_dims(outputs, 0), axis=0)

    return torch.tensor(in_price), torch.tensor(in_price_star), torch.tensor(in_sigma), torch.tensor(out_price)
