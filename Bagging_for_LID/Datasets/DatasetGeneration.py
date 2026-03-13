import skdim
import numpy as np
###############################################################################################################################DATASET GENERATION###############################################################################################################################
def lollipop_dataset(bs, seed=0, categorical=0.95):
    np.random.seed(seed)
    cs = int(categorical * bs)
    x = np.zeros((bs, 2))
    intrinsic_dims = np.zeros(bs, dtype=int)
    r = np.random.uniform(size=cs)
    fi = np.random.uniform(0, 2 * np.pi, size=cs)
    x[:cs, 0] = r ** 0.5 * np.sin(fi)
    x[:cs, 1] = r ** 0.5 * np.cos(fi)
    x[:cs] += 2
    intrinsic_dims[:cs] = 2
    stick = np.random.uniform(high=2 - 1 / np.sqrt(2), size=(bs - cs))
    x[cs:, 0] = stick
    x[cs:, 1] = stick
    intrinsic_dims[cs:] = 1
    return x, intrinsic_dims

def lollipop_dataset_0(bs, seed=0):
    np.random.seed(seed)
    cs = int(0.94 * bs)
    cp = int(0.99 * bs)
    x = np.zeros((bs, 2))
    intrinsic_dims = np.zeros(bs, dtype=int)
    r = np.random.uniform(size=cs)
    fi = np.random.uniform(0, 2 * np.pi, size=cs)
    x[:cs, 0] = r ** 0.5 * np.sin(fi)
    x[:cs, 1] = r ** 0.5 * np.cos(fi)
    x[:cs] += 2
    intrinsic_dims[:cs] = 2
    stick = np.random.uniform(high=2 - 1 / np.sqrt(2), size=(cp - cs))
    x[cs:cp, 0] = stick
    x[cs:cp, 1] = stick
    intrinsic_dims[cs:cp] = 1
    x[cp:] = np.random.normal(loc=(-.5, -.5), scale=1e-3, size=(bs - cp, 2))
    intrinsic_dims[cp:] = 2
    x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    return x, intrinsic_dims


def lollipop_dataset_0_dense_head(bs, seed=0):
    np.random.seed(seed)
    cs = int(0.5 * bs)
    cp = int(0.7 * bs)
    x = np.zeros((bs, 2))
    intrinsic_dims = np.zeros(bs, dtype=int)
    r = np.random.uniform(size=cs)
    fi = np.random.uniform(0, 2 * np.pi, size=cs)
    x[:cs, 0] = r ** 0.5 * np.sin(fi)
    x[:cs, 1] = r ** 0.5 * np.cos(fi)
    x[:cs] += 2
    intrinsic_dims[:cs] = 2
    stick = np.random.uniform(high=2 - 1 / np.sqrt(2), size=(cp - cs))
    x[cs:cp, 0] = stick
    x[cs:cp, 1] = stick
    intrinsic_dims[cs:cp] = 1
    x[cp:] = np.random.normal(loc=(-.5, -.5), scale=1e-3, size=(bs - cp, 2))
    intrinsic_dims[cp:] = 2
    x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    return x, intrinsic_dims

def ribbon_multi_dim_equal_density(n, dim=2, d_loc=2, d_glob=1, ratio=0.05):
    U_glob = np.random.uniform(low=0.0, high=1.0, size=(n, d_glob))
    U_loc = np.random.uniform(low=0.0, high=ratio, size=(n, d_loc-d_glob))
    if dim-d_loc > 0:
        zeros = np.zeros((n, dim-d_loc))
        data = np.hstack([U_glob, U_loc, zeros])
    data = np.hstack([U_glob, U_loc])
    lids = np.ones((n, 1))*d_loc
    return data, lids

def sparse(n, lid=2, dim=2, w=10, l=5, center=(0.0, 0.0), rng=None):
    if w <= 0:
        raise ValueError("w must be > 0")
    if l <= 0:
        raise ValueError("l must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = rng.uniform(0.0, 1.0, size=n)
    r = w * u ** (1.0 / l)
    cx, cy = center
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    data2d = np.column_stack([x, y])
    if dim > 2:
        zeros = np.zeros((n, dim - 2))
        data = np.hstack([data2d, zeros])
    else:
        data = data2d
    lids = np.ones((n, 1)) * lid
    return data, lids

def data_defaults():
    data_gen = skdim.datasets.BenchmarkManifolds()
    all_keys = [key for key in data_gen.dict_gen]
    keys = all_keys[0:4] + all_keys[5:13] + all_keys[14:17] + all_keys[19:21]
    keys.append("lollipop_")
    keys.append("uniform")
    keys.append("ribbon")
    keys.append("custom")
    keys.append("sparse")
    d_vals = [10, 3, 4, 4, 2, 6, 2, 12, 20, 10, 17, 24, 2, 20, 2, 18, 24, 2, 30, 2, 1, 2]
    m_vals = [11, 5, 6, 8, 3, 36, 3, 72, 20, 11, 18, 25, 3, 20, 3, 72, 96, 2, 100, 2, 1, 2]
    params = [(keys[i], [d_vals[i], m_vals[i]]) for i in range(len(keys))]
    used_params = dict(params)
    return used_params

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, _ = lollipop_dataset(10000)
    plt.scatter(x[:,0], x[:,1], marker=".")
    plt.show()