from functools import lru_cache

@lru_cache
def compute_padding(window_size, t, h, w):
    pad_t0 = pad_h0 = pad_w0 = 0
    pad_t1 = (window_size[0] - t % window_size[0]) % window_size[0]
    pad_h1 = (window_size[1] - h % window_size[1]) % window_size[1]
    pad_w1 = (window_size[2] - w % window_size[2]) % window_size[2]
    return pad_w0, pad_w1, pad_h0, pad_h1, pad_t0, pad_t1, 
