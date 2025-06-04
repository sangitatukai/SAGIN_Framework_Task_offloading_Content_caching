import numpy as np

def compute_rate_general(sender_pos, receiver_pos, 
                         bandwidth=1e6,
                         P_tx=10.0,
                         noise=1e-13,          # Ultra-low noise
                         fc=12e9,
                         fading=1.0,
                         G_tx=10000,           # 40 dB
                         G_rx=10000,           # 40 dB
                         xi=2.0):
    d = np.linalg.norm(np.array(sender_pos) - np.array(receiver_pos))
    c = 3e8
    if d == 0:
        d = 1.0  # Prevent divide by zero
    L = (4 * np.pi * fc * d / c) ** xi   # Free-space path loss (linear)
    G = fading * G_tx * G_rx / L
    SINR = (P_tx * G) / noise
    rate = bandwidth * np.log2(1 + SINR)
    return rate, rate >= 1e5  # True if at least 100 kbps
