"""
@Author     dizs@using.ai
@Date       2022-04-27
@Describe   MAS 实现
"""
from typing import List
import random

import torch
from torch import nn, Tensor


class Discriminator(nn.Module):
    """原生GAN 判别器"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_model = nn.Sequential(
            nn.Linear(10, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_input):
        """前向过程"""
        x_input = self.fc_model(x_input)
        return x_input

class Generator(nn.Module):
    """MAS 生成器"""
    def __init__(self, num_samples: list, output_length: int, rho: int=0.15, \
            sigma_I: float=9., sigma_P: float=1., mu1: float=100., mu2: float=0.01, \
            nu1: float=0.1, nu2: float=0.1):
        """init
        """
        super().__init__()
        self.samples = num_samples
        self.output_length = output_length
        self.mas_cell = MasLayer(n_samples=self.samples, move_average_length=output_length,\
            rho=rho, sigma_I=sigma_I, sigma_P=sigma_P, mu1=mu1, mu2=mu2, nu1=nu1, nu2=nu2)

    def forward(self, price: Tensor, p_start_t: Tensor, sigma_t: Tensor):
        """
        Args:
            price (Tensor): 股票价格 shape: (batch_size, input_len, 1)
            p_star (Tensor): 股票价值 shape: (batch_size, input_len, 1)
            sigma (Tensor): shape: (batch_size, input_len, 1)
        Returns:
            Tensor: (batch_size, output_len, 1)
        """
        # (batch_size, 1), (batch_size, 1), (batch_size, 1)
        output_len = self.output_length
        prices = torch.zeros((price.shape[0], price.shape[1]+output_len, 1), device=price.device, requires_grad=False)
        prices[:, :price.shape[1], :] = price
        p_starts = torch.zeros((price.shape[0], price.shape[1]+output_len, 1), device=price.device, requires_grad=False)
        p_starts[:, :price.shape[1], :] = p_start_t
        sigmas = torch.zeros((price.shape[0], price.shape[1]+output_len, 1), device=price.device, requires_grad=False)
        sigmas[:, :price.shape[1], :] = sigma_t
        for i in range(price.shape[1], prices.shape[1]):
            _price_t, _price_star_t, _sigma_t = self.mas_cell(prices[:, :i, :], p_starts[:, :i, :],\
                sigmas[:, :i, :])
            prices[:, i:i+1, :] = _price_t
            p_starts[:, i:i+1, :] = _price_star_t
            sigmas[:, i:i+1, :] = _sigma_t

        return prices[:, -output_len:, :], sigmas[:, -output_len:, :], p_starts[:, -output_len:, :]


class MasLayer(nn.Module):
    """MAS Layer"""
    def __init__(
        self, n_samples: List[int], move_average_length: int=10, rho: int=0.15, \
            sigma_I: float=9., sigma_P: float=1., mu1: float=100., mu2: float=0.01, \
            nu1: float=0.1, nu2: float=0.1
    ):
        super().__init__()
        self.noise_traders = NoiseLayer(n_samples=n_samples[0], sigma_I=sigma_I, sigma_P=sigma_P)
        self.technical_traders = TechnicalLayer(n_samples=n_samples[1])
        self.fundamental_traders = FundamentalLayer(n_samples=n_samples[2], mu1=mu1, mu2=mu2, \
            nu1=nu1, nu2=nu2)
        self.move_average_length = move_average_length
        self.rho = rho
        self.n_samples = n_samples

    def forward(self, prices: Tensor, p_stars: Tensor, sigmas: Tensor):
        """
        Args:
            price (Tensor): 股票价格 shape: (batch_size, input_len, 1)
            p_star (Tensor): 股票价值 shape: (batch_size, input_len, 1)
            sigma (Tensor): shape: (batch_size, input_len, 1)

        Returns:
            Tensor: shape (batch_size, 1, 1)
        """
        # (batch_size, n_feature)
        demand1 = self.noise_traders(prices)
        if prices.shape[1] >= self.move_average_length:
            demand2 = self.technical_traders(prices)
        else:
            demand2 = torch.zeros((prices.shape[0], 1, 1), device=prices.device)
        demand3, sigma_t, p_star_t = self.fundamental_traders(prices, p_stars, sigmas)
        # demand = torch.cat([demand1, demand2, demand3], dim=1)
        # demand = torch.nan_to_num(demand, nan=1.)
        # demand_all = torch.sum(demand, dim=1, keepdim=True)
        price_t = prices[:, -1:, :].clone().detach().to(prices.device) * (1 + self.rho * (demand1 + demand2 + demand3))

        return price_t, p_star_t, sigma_t

class NoiseLayer(nn.Module):
    """Noise Trader"""
    def __init__(self, n_samples: int=1, sigma_I: float=9., sigma_P: float=1):
        """
        Args:
            n_samples (int, optional): Noise Trader 数量. Defaults to 1.
            sigma_I (float, optional): Noise Trader 接收的 public signal 超参数.
                    Defaults to 9..
        """
        super().__init__()
        self.param_b = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.param_c = nn.Parameter(torch.randn(1, 1), requires_grad=True)

        self.n_samples = n_samples
        self.sigma_i = sigma_I
        self.sigma_p = sigma_P

    def forward(self, price: Tensor):
        """

        Args:
            price (Tensor): 历史价格 shape: (batch_size, input_len, 1)

        Returns:
            Tensor: (batch_size, 1, 1)
        """
        # 1. 生成public signal
        # (batch_size, n_samples, 1)
        n_pub_sign = torch.empty((price.shape[0],)).normal_(mean=0., std=self.sigma_p)
        # (batch_size, n_samples, 1)
        pub_sign = torch.ones((price.shape[0], self.n_samples, 1))
        for i in range(price.shape[0]):
            pub_sign[i] = torch.full((self.n_samples, 1), n_pub_sign[i].item())

        # 2. 生成individual signal
        # (batch_size, n_samples, 1)
        priv_sign = torch.empty((price.shape[0], self.n_samples, 1)).normal_(mean=0., std=self.sigma_i)

        # 3. 计算所有noise trader 的 demand signal
        pub_sign = pub_sign.to(price.device)
        priv_sign = priv_sign.to(price.device)
        signal = pub_sign * self.param_b + priv_sign * self.param_c
        signal[signal > 20] = 20.
        signal[signal < -20] = -20.
        signal = 2 * torch.exp(signal) / (1 + torch.exp(signal)) - 1
        signal = torch.mean(signal, dim=1, keepdim=True)
        return signal

class TechnicalLayer(nn.Module):
    """Technical Trader"""
    def __init__(self, n_samples: int=5):
        """
        Args:
            n_samples (int, optional): technical trader 数量. Defaults to 5.
        """
        super().__init__()
        self.param_d = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.n_samples = n_samples


    def forward(self, price: Tensor):
        """

        Args:
            price (Tensor): 历史价格 (batch_size, input_len, 1)

        Returns:
            Tensor: shape (batch_size, 1, 1)
        """
        # (batch_size, n_samples, input_len)
        len1 = torch.zeros((price.shape[0], self.n_samples, price.shape[1]))
        len2 = torch.zeros((price.shape[0], self.n_samples, price.shape[1]))
        for j in range(self.n_samples):
            values = sorted(random.sample(range(2,11), 2))
            len1[:, j, -values[0]:] = 1 / values[0]
            len2[:, j, -values[1]:] = 1 / values[1]
        len1 = len1.to(price.device)
        len2 = len2.to(price.device)
        price = torch.unsqueeze(price, 1).expand((price.shape[0], \
            self.n_samples, price.shape[1], 1)).squeeze()
        signal = torch.sum(price * len1, axis=-1, keepdim=True) - \
            torch.sum(price * len2, axis=-1, keepdim=True)
        signal[signal > 20] = 20
        signal[signal < -20] = -20
        signal = 2 * torch.exp(self.param_d * signal) / (1 + torch.exp(self.param_d * signal)) - 1
        signal = torch.mean(signal, dim=1, keepdim=True)
        return signal

class FundamentalLayer(nn.Module):
    """Fundamental Trader"""
    def __init__(self, n_samples: int=3, mu1: float=100., mu2: float=0.01, \
            nu1: float=0.01, nu2: float=0.1):
        """
        Args:
            n_samples (int, optional): fundamental trader 数量. Defaults to 3.
            mu1 (float, optional): Heston model 超参数. Defaults to 100..
            mu2 (float, optional): Heston model 超参数. Defaults to 0.01.
            nu1 (float, optional): Heston model 超参数. Defaults to 0.011.
            nu2 (float, optional): Heston model 超参数. Defaults to 0.1.
        """
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.nu1 = nu1
        self.nu2 = nu2
        self.n_samples = n_samples

    def forward(self, price: Tensor, p_star: Tensor, sigma: Tensor):
        """
        Args:
            price (Tensor): 股票价格 shape: (batch_size, input_len, 1)
            p_star (Tensor): 股票价值 shape: (batch_size, input_len, 1)
            sigma (Tensor): shape: (batch_size, input_len, 1)
        Returns:
            Tensor: (batch_size, 1, 1)
        """
        epsilon1 = torch.randn((price.shape[0], 1, 1), device=price.device)
        epsilon2 = torch.randn((price.shape[0], 1, 1), device=price.device)
        sigma_f = torch.randn((price.shape[0], self.n_samples, 1), device=price.device)
        price = torch.unsqueeze(price[:, -1, :], 1).expand((price.shape[0], self.n_samples, 1))
        ## !!! 这里可以研究下sigma的非负问题
        sigma_t = sigma[:, -1:, :] + self.nu2 * (self.mu2 - sigma[:, -1:, :]) + \
            torch.abs(sigma[:, -1:, :]) ** 0.5 * epsilon2
        p_star_t = p_star[:, -1:, :] + self.nu1 * (self.mu1 - p_star[:, -1:, :]) + \
            torch.abs(sigma_t) ** 0.5 * epsilon1

        p_star = p_star_t.expand((p_star_t.shape[0], self.n_samples, p_star_t.shape[2]))
        signal = p_star - price + sigma_f
        signal[signal > 20] = 20
        signal[signal < -20] = -20
        signal = 2 * torch.exp(signal) / (1 + torch.exp(signal)) - 1
        signal = torch.mean(signal, dim=1, keepdim=True)
        return signal, sigma_t, p_star_t
