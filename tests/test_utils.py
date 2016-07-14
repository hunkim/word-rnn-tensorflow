import unittest
from utils import TextLoader
import numpy as np
from collections import Counter

class TestUtilsMethods(unittest.TestCase):
    def setUp(self):
        self.data_loader = TextLoader("tests/test_data", batch_size=2, seq_length=5)

    def test_init(self):
      print (self.data_loader.vocab)
      print (self.data_loader.tensor)
      print (self.data_loader.vocab_size)

    def test_build_vocab(self):
        sentences = ["I", "love", "cat", "cat"]
        vocab, vocab_inv = self.data_loader.build_vocab(sentences)
        print (vocab, vocab_inv)

        # Must include I, love, and cat
        self.assertEqual(Counter(list(vocab)), Counter(list(["I", "love", "cat"])))
        self.assertDictEqual(vocab, {'I': 0, 'love': 2, 'cat': 1})

        self.assertEqual(Counter(list(vocab_inv)), Counter(list(["I", "love", "cat"])))

    def test_batch_vocab(self):
        print (np.array(self.data_loader.x_batches).shape)
        self.assertEqual(Counter(list(self.data_loader.x_batches[0][0][1:])),
                              Counter(list(self.data_loader.y_batches[0][0][:-1])))
        self.assertEqual(Counter(list(self.data_loader.x_batches[0][1][1:])),
                              Counter(list(self.data_loader.y_batches[0][1][:-1])))


if __name__ == '__main__':
    unittest.main()
