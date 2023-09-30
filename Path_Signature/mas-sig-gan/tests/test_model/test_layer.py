"""测试MAS layer"""
import configparser
import unittest
import torch
from model.maslayer import NoiseLayer, TechnicalLayer, FundamentalLayer, \
    MasLayer, Generator

class TestNoiseLayer(unittest.TestCase):
    """测试NoiseLayer"""
    def test_init(self):
        """测试初始化"""
        noise_layer = NoiseLayer(n_samples=7, sigma_I=9.0, sigma_P=1.0)
        self.assertTrue(isinstance(noise_layer, torch.nn.Module))

    def test_forward(self):
        """测试前向过程"""
        noise_layer = NoiseLayer(n_samples=7, sigma_I=9.0, sigma_P=1.0)
        price = torch.randn((1000, 10, 1))
        output = noise_layer(price)
        self.assertEqual(output.shape, (1000, 1, 1))

class TestTechnicalLayer(unittest.TestCase):
    """测试TechnicalLayer"""
    def test_init(self):
        """测试初始化"""
        noise_layer = TechnicalLayer(n_samples=5)
        self.assertTrue(isinstance(noise_layer, torch.nn.Module))

    def test_forward(self):
        """测试前向过程"""
        technical_layer = TechnicalLayer(n_samples=5)
        price = torch.randn((1000, 10, 1))
        output = technical_layer(price)
        self.assertEqual(output.shape, (1000, 1, 1))

class TestFundamentalLayer(unittest.TestCase):
    """测试FundamentalLayer"""
    def test_init(self):
        """测试初始化"""
        fundamental_layer = FundamentalLayer(n_samples=5)
        self.assertTrue(isinstance(fundamental_layer, torch.nn.Module))

    def test_forward(self):
        """测试前向过程"""
        fundamental_layer = FundamentalLayer(n_samples=5)
        price = torch.randn((1000, 10, 1))
        p_star = torch.randn((1000, 10, 1))
        sigma = torch.randn((1000, 10, 1))
        output = fundamental_layer(price, p_star, sigma)
        self.assertEqual(output[0].shape, (1000, 1, 1))
        self.assertEqual(output[1].shape, (1000, 1, 1))
        self.assertEqual(output[2].shape, (1000, 1, 1))

class TestMasLayer(unittest.TestCase):
    """测试MasLayer"""
    def test_init(self):
        """测试初始化"""
        mas_layer = MasLayer(n_samples=[7, 5, 3])
        self.assertTrue(isinstance(mas_layer, torch.nn.Module))

    def test_forward(self):
        """测试前向过程"""
        mas_layer = MasLayer(n_samples=[7, 5, 3])
        price = torch.randn((1000, 10, 1))
        p_star = torch.randn((1000, 10, 1))
        sigma = torch.randn((1000, 10, 1))
        output = mas_layer(price, p_star, sigma)
        self.assertEqual(output[0].shape, (1000, 1, 1))
        self.assertEqual(output[1].shape, (1000, 1, 1))
        self.assertEqual(output[2].shape, (1000, 1, 1))

class TestGenerator(unittest.TestCase):
    """测试Generator"""
    def test_init(self):
        """测试初始化"""
        generator = Generator([7, 5, 3], 10)
        self.assertTrue(isinstance(generator, torch.nn.Module))

    def test_forward(self):
        """测试前向过程"""
        generator = Generator([7, 5, 3], 10)
        price = torch.randn((1000, 10, 1))
        p_star = torch.randn((1000, 10, 1))
        sigma = torch.randn((1000, 10, 1))
        output = generator(price, p_star, sigma)
        self.assertEqual(output[0].shape, (1000, 10, 1))
        self.assertEqual(output[1].shape, (1000, 10, 1))
        self.assertEqual(output[2].shape, (1000, 10, 1))


if __name__ == "__main__":
    unittest.main()
