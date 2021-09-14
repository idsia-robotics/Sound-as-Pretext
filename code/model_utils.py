import torch


def _closest_log2(x, shift=0):
    return 1 if x < 2 else (x - 1).bit_length() + shift


def _pairs(l):
    return [l[i:i + 2] for i in range(len(l) - 1)]


# note: max_size is the size of the embedding space produced by the encoder
# note: min_size is the minimum size of a fc layer (used to avoid too many layers 1->2->4->8->16)

# note: min_length is the the lenght in one dimension (not channels) of the embedding space produced by the encoder
# note: min_channels is the minimum number of channels in the conv kernel (used to avoid too few features)

# note: for data that needs convolution (ndim > 1), make sure lenght of all dimensions (except channels) is one of
# 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
# 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640,
# 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960,
# 976, 992, 1008, 1024, 1040, 1056, 1072, 1088, 1104, 1120, 1136, 1152, 1168, 1184, 1200, 1216, 1232, 1248, 1264, 1280,
# 1296, 1312, 1328, 1344, 1360, 1376, 1392, 1408, 1424, 1440, 1456, 1472, 1488, 1504, 1520, 1536, 1552, 1568, 1584, 1600,
# 1616, 1632, 1648, 1664, 1680, 1696, 1712, 1728, 1744, 1760, 1776, 1792, 1808, 1824, 1840, 1856, 1872, 1888, 1904, 1920,
# 1936, 1952, 1968, 1984, 2000, 2016, 2032, 2048, ...


def _hidden_seq(in_size, max_size, min_size=16):
    compress = in_size > max_size
    l = max(min_size, 2 ** _closest_log2(in_size, -1 * compress))

    if compress:
        def get_next(x): return x >> 1
        def has_finished(x): return x < max_size
    else:
        def get_next(x): return x << 1
        def has_finished(x): return x > max_size

    res = [in_size]
    while not has_finished(l):
        res.append(l)
        l = get_next(l)

    return res


def _channel_seq(in_shape, min_length, max_channels, min_channels=16):
    c = max(min_channels, 2 ** _closest_log2(in_shape[0]))
    l = 2 ** _closest_log2(min(in_shape[1:]), -1)

    res = [in_shape[0]]
    while l >= min_length and c <= max_channels:
        res.append(c)
        c <<= 1
        l >>= 1

    if res[-1] < max_channels:
        res.append(max_channels)

    return res


def _compute_seq(in_shape, min_length, max_size):
    ndim = len(in_shape)

    if ndim < 1 or 4 < ndim:
        raise ValueError(
            'Supported shapes consist of tuples of length 1 through 4')

    if ndim == 1:
        return _hidden_seq(in_shape[0], max_size)
    else:
        return _channel_seq(in_shape, min_length, max_size)


def _build(in_shape, seq, encoder=True, nonlinear=torch.nn.ReLU6):
    layer_mapping = {
        True: {
            1: torch.nn.Linear,
            2: torch.nn.Conv1d,
            3: torch.nn.Conv2d,
            4: torch.nn.Conv3d,
        },
        False: {
            1: torch.nn.Linear,
            2: torch.nn.ConvTranspose1d,
            3: torch.nn.ConvTranspose2d,
            4: torch.nn.ConvTranspose3d,
        }
    }

    kwargs_mapping = {
        True: {'kernel_size': 5, 'stride': 2, 'padding': 2},
        False: {'kernel_size': 4, 'stride': 2, 'padding': 1}
    }

    pool_mapping = {
        2: torch.nn.AdaptiveAvgPool1d,
        3: torch.nn.AdaptiveAvgPool2d,
        4: torch.nn.AdaptiveAvgPool3d,
    }

    layers = []
    ndim = len(in_shape)
    layer = layer_mapping[encoder][ndim]
    kwargs = {} if ndim == 1 else kwargs_mapping[encoder]

    if not encoder:
        seq = list(reversed(seq))

    for cin, cout in _pairs(seq):
        layers.append(layer(cin, cout, **kwargs))
        layers.append(nonlinear())

    if ndim > 1:
        pool_layer = pool_mapping[ndim]

        if encoder:
            layers.append(pool_layer(1))
            layers.append(torch.nn.Flatten())
        else:
            encoded_shape = (torch.tensor(
                in_shape[1:], dtype=int) >> len(seq) - 1).tolist()
            layers.insert(0, pool_layer(encoded_shape))
            shape = tuple(seq[:1] + [1] * (ndim - 1))
            # note: for new versions of pytorch
            # layers.insert(0, torch.nn.Unflatten(
            #     dim=1, unflattened_size=shape))

            # note: workaround for old versions of pytorch
            class Unflatten(torch.nn.Module):
                def __init__(self, dim, unflattened_size):
                    super(Unflatten, self).__init__()
                    self.dim = dim
                    self.unflattened_size = unflattened_size

                def forward(self, input):
                    shape = input.shape[:self.dim]
                    return input.view(shape + self.unflattened_size)

            layers.insert(0, Unflatten(
                dim=1, unflattened_size=shape))

    return layers
