import torch
import torchvision
from torchsummary import summary
from model_utils import _build, _compute_seq


class NN(torchvision.models.MobileNetV2):
    def __init__(self, out_channels=1, settings=None):
        self.out_channels = out_channels

        if settings is None:
            settings = [
                ## t, c, n, s
                [1, 16, 1, 1],
                [6, 32, 2, 2],
                [6, 64, 3, 2],
                [6, 128, 2, 2],
                [6, 256, 1, 1]
            ]

        super(NN, self).__init__(out_channels, round_nearest=8,
                                 inverted_residual_setting=settings)

    def forward(self, img):
        return self._forward_impl(img)


class NN_autoenc(torch.nn.Module):

    def __init__(self, in_shape, min_length=4, max_size=128):
        super(NN_autoenc, self).__init__()

        seq = _compute_seq(in_shape, min_length, max_size)

        self.encoder = torch.nn.Sequential(
            *_build(in_shape, seq, encoder=True))
        self.decoder = torch.nn.Sequential(
            *_build(in_shape, seq, encoder=False))

        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU6(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU6(),
            torch.nn.Linear(32, 3)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def predict(self, x):
        return

    def forward(self, x, predict=False):
        enc = self.encode(x)
        dec = self.decode(enc)

        if predict:
            return dec, self.head(enc)

        return dec


class NN_fusion(torchvision.models.MobileNetV2):
    def __init__(self, out_channels=1, settings=None):
        self.out_channels = out_channels

        if settings is None:
            settings = [
                ## t, c, n, s
                [1, 16, 1, 1],
                [6, 32, 2, 2],
                [6, 64, 3, 2],
                [6, 128, 2, 2],
                [6, 256, 1, 1]
            ]

        super(NN_fusion, self).__init__(64, round_nearest=8,
                                        inverted_residual_setting=settings)

        self.audio_net = torch.nn.Sequential(
            torch.nn.Linear(9, 128),
            torch.nn.ReLU6(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU6(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU6(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU6(),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU6(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU6(),
            torch.nn.Linear(64, out_channels)
        )

    def forward(self, img, audio):
        img_feat = self._forward_impl(img)
        audio_feat = self.audio_net(audio)
        result = self.head(torch.cat([
            img_feat, audio_feat
        ], dim=-1))
        return result


class NN_audio(torch.nn.Module):
    def __init__(self, out_channels=1):
        super(NN_audio, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(9, 256),
            torch.nn.ReLU6(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU6(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU6(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU6(),
            torch.nn.Linear(64, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    model = NN(3)
    summary(model, (3, 80, 128), device='cpu')
