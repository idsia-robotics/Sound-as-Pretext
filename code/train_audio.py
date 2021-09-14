import os
import torch
import argparse
import numpy as np
import pandas as pd
from model import NN_audio
from test_audio import test
from tqdm import tqdm, trange
from datetime import datetime
from dataset import get_dataset


def train(name, filename, epochs, batch_size, learning_rate, device):
    # Create model folder
    model_path = './model/' + name + '_audio'
    checkpoint_path = model_path + '/checkpoints'

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Load dataset
    train_dataset = get_dataset(filename, 't2',
                                augment=True, device=device)
    val_dataset = get_dataset(filename, 'val',
                              augment=False, device=device)

    # Model, optimizer & loss
    model = NN_audio(out_channels=3).to(device)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)

    # Train
    history = pd.DataFrame()
    training_steps = len(train_dataset) // batch_size
    epochs_logger = trange(1, epochs + 1, desc='epoch')
    best_val_loss = 1e9

    for epoch in epochs_logger:
        generator = train_dataset.batches(batch_size, shuffle=True)
        steps_logger = tqdm(generator, desc='step', total=training_steps)

        val_metrics = [np.nan] * 4
        for step, data in enumerate(steps_logger, start=1):
            y_train = data['optitrack_tello_proj'][:, :3]
            x_train = data['mp3_features']

            model.train()
            optimizer.zero_grad()

            pred = model(x_train)
            loss = loss_function(pred, y_train)

            loss.backward()
            optimizer.step()

            if step == training_steps:
                val_metrics = test(model, val_dataset, batch_size)
                scheduler.step()

            history = history.append({
                'epoch': epoch,
                'step': step,
                'loss': loss.item(),
                'val_loss': val_metrics[0],
                'val_r2x': val_metrics[1],
                'val_r2y': val_metrics[2],
                'val_r2z': val_metrics[3],
            }, ignore_index=True)

            mean_values = history.query('epoch == ' + str(epoch)
                                        ).mean(axis=0, skipna=True)
            mean_metrics = mean_values[['loss', 'val_loss',
                                        'val_r2x', 'val_r2y', 'val_r2z']].tolist()

            log_str = 'L: %.4f' % (mean_metrics[0])
            steps_logger.set_postfix_str(log_str)

            if step == training_steps:
                log_str += ', VL: %.4f, R2x=%.4f, R2y=%.4f, R2z=%.4f' % tuple(
                    mean_metrics[1:])
                print
                epochs_logger.set_postfix_str(log_str)

        checkpoint_name = '%d_%.4f_state_dict.pth' % (epoch, mean_metrics[1])
        torch.save(model.state_dict(), checkpoint_path + '/' + checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path + '/last.pth')

        if best_val_loss >= mean_metrics[1]:
            torch.save(model.state_dict(), checkpoint_path + '/best.pth')
            best_val_loss = mean_metrics[1]

    history.to_csv(model_path + '/history.csv')

    import matplotlib.pyplot as plt
    history.groupby('epoch').mean().plot(
        y=['loss', 'val_loss'], use_index=True)
    plt.ylim(0, 0.5)
    plt.title('audio')
    plt.savefig(model_path + '/losses.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default='model_' + str(datetime.now()))
    parser.add_argument('-f', '--filename', type=str, help='name of the dataset (.h5 file)',
                        default='./data/processed/d89jul_aaa')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs of the training phase',
                        default=60)
    parser.add_argument('-bs', '--batch-size', type=int, help='size of the batches of the training data',
                        default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate used for the training phase',
                        default=1e-3)
    parser.add_argument('-d', '--device', type=str, help=argparse.SUPPRESS,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    train(**vars(args))
