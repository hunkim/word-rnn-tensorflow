import tensorflow as tf
import numpy as np


class BeamSearch():
    def __init__(self, probs):
        self.probs = probs

    def beamsearch(self, oov, empty, eos, k=1, maxsample=4000, use_unk=False):
        """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
        all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
        You need to supply `predict` which returns the label probability of each sample.
        `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
        """

        dead_k = 0  # samples that reached eos
        dead_samples = []
        dead_scores = []
        live_k = 1  # samples that did not yet reached eos
        live_samples = [[empty]]
        live_scores = [0]

        while live_k and dead_k < k:

            # total score for every sample is sum of -log of word prb
            cand_scores = np.array(live_scores)[:, None] - np.log(self.probs)
            if not use_unk and oov is not None:
                cand_scores[:, oov] = 1e20
            cand_flat = cand_scores.flatten()

            # find the best (lowest) scores we have from all possible samples and new words
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]
            live_scores = cand_flat[ranks_flat]

            # append the new words to their appropriate live sample
            voc_size = self.probs.shape[1]
            live_samples = [live_samples[r // voc_size] + [r % voc_size] for r in ranks_flat]

            # live samples that should be dead are...
            zombie = [s[-1] == eos or len(s) >= maxsample for s in live_samples]

            # add zombies to the dead
            dead_samples += [s for s, z in zip(live_samples, zombie) if z]  # remove first label == empty
            dead_scores += [s for s, z in zip(live_scores, zombie) if z]
            dead_k = len(dead_samples)
            # remove zombies from the living
            live_samples = [s for s, z in zip(live_samples, zombie) if not z]
            live_scores = [s for s, z in zip(live_scores, zombie) if not z]
            live_k = len(live_samples)

        return dead_samples + live_samples, dead_scores + live_scores