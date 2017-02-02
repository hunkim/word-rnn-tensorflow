import unittest
import numpy as np

from beam import BeamSearch


def naive_predict(sample, state):
    """Fake predict function.

    For our model, let's assume a vocabulary of size 5. Furthermore, let's say
    that the `state` is exactly the probability that each vocabulary occurs,
    and these probabilities never change.
    """

    return np.array(state)[None, :], state


class TestBeamMethods(unittest.TestCase):
    def setUp(self):
        self.prime_labels = [0, 1]
        self.initial_state = [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_single_beam(self):
        bs = BeamSearch(naive_predict, self.initial_state, self.prime_labels)
        samples, scores = bs.search(None, None, k=1, maxsample=5)
        self.assertEqual(samples, [[0, 1, 4, 4, 4]])

    def test_multiple_beams(self):
        bs = BeamSearch(naive_predict, self.initial_state, self.prime_labels)
        samples, scores = bs.search(None, None, k=4, maxsample=5)

        self.assertIn([0, 1, 4, 4, 4], samples)

        # All permutations of this form must be in the results.
        self.assertIn([0, 1, 4, 4, 3], samples)
        self.assertIn([0, 1, 4, 3, 4], samples)
        self.assertIn([0, 1, 3, 4, 4], samples)

        # Make sure that the best beam has the lowest score.
        self.assertEqual(samples[np.argmin(scores)], [0, 1, 4, 4, 4])

if __name__ == '__main__':
    unittest.main()
