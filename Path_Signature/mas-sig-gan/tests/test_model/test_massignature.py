"""测试MAS layer"""
# import sys
# sys.path.append("/home/ai/project/mas-gan")
import unittest
import torch
from model.massignature import MASignature

class TestMASignature(unittest.TestCase):
    """测试MASignature"""
    def test_signature(self):
        model = MASignature()
        data = torch.randn((1000, 10, 1))
        result = model.signature(data)
        self.assertEqual(result.shape, (1000, 584))

if __name__ == "__main__":
    unittest.main()
