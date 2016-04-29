import unittest
from utils import TextLoader
import numpy as np

class TestUtilsMethods(unittest.TestCase):
    def setUp(self):
        self.data_loader = TextLoader("tests/test_data", batch_size=2, seq_length=5)

    def test_init(self):
        print ("Work in progress.")


if __name__ == '__main__':
    unittest.main()