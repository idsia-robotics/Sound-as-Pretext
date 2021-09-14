import os
import torch
import argparse
import numpy as np
from model import NN
from dataset import get_dataset
from sklearn.metrics import mean_absolute_error, r2_score


def test(model, test_dataset, batch_size):
    # Test
    ys = []
    preds = []
    generator = test_dataset.batches(batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for data in generator:
            y_test = data['optitrack_tello_proj'][:, :3]
            im_test = data['rm_s1_camera_image_h264']
            y = y_test.cpu().numpy()
            pred = model(im_test).cpu().numpy()[:, :3]

            ys.append(y)
            preds.append(pred)

    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    loss = mean_absolute_error(y_pred=pred, y_true=y)
    r2 = [r2_score(y_pred=pred[:, i], y_true=y[:, i]) for i in range(3)]

    return (loss,) + tuple(r2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_viz')
    parser.add_argument('-f', '--filename', type=str, help='name of the dataset (.h5 file)',
                        default='./data/processed/d89jul_aaa')
    parser.add_argument('-s', '--split', type=str, help='dataset split, one of train, validation or test',
                        default='validation')
    parser.add_argument('-bs', '--batch-size', type=int, help='size of the batches of the training data',
                        default=64)
    parser.add_argument('-d', '--device', type=str, help=argparse.SUPPRESS,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    checkpoint_path = './model/' + args.name + '/checkpoints/'

    if not os.path.exists(checkpoint_path):
        raise ValueError(checkpoint_path + ' does not exist')

    model = NN(out_channels=3).to(args.device)
    model.load_state_dict(
        torch.load(checkpoint_path + 'best.pth',
                   map_location=args.device))

    dataset = get_dataset(args.filename, args.split,
                          augment=False, device=args.device)

    metrics = test(model, dataset, args.batch_size)

    print(args.split + ': L=%.4f, R2x=%.4f, R2y=%.4f, R2z=%.4f' % metrics)
