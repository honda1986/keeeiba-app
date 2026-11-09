"""Plackett-Luce(Harville)で組み合わせの確率を出す小道具(実験用)"""
import os, sys
from itertools import permutations
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import keiba_final as F


def fit(p3):
    """3着内の確率(合計3)に合う強さ"""
    t = np.clip(np.array(p3, float), .005, .995); t = np.minimum(t * 3 / t.sum(), .995)
    w = t / t.sum()
    for _ in range(200):
        q = F.pl_top3(w); w = w * (t / np.maximum(q, 1e-9)) ** .8; w /= w.sum()
        if np.abs(q - t).max() < 1e-4:
            break
    return w


def combo_p(w, us, k):
    """番号 us の k 頭が全部3着内に入る確率(k=3 は3連複、k=2 はワイド)"""
    S = w.sum(); p = 0.0
    if k == 3:
        for a, b, c in permutations(us):
            p += w[a] / S * w[b] / (S - w[a]) * w[c] / (S - w[a] - w[b])
        return p
    a, b = us
    return sum(combo_p(w, [a, b, c], 3) for c in range(len(w)) if c not in (a, b))
