# unigram word-frequency prior from wordfreq (independent of all study data)
import numpy as np


def unigram_logprior(words):
    """Natural-log unigram probability, normalised over the lexicon (zipf = log10 per-billion frequency)."""
    from wordfreq import zipf_frequency
    lp = np.array([zipf_frequency(w, "en") for w in words]) * np.log(10)
    return lp - np.logaddexp.reduce(lp)
