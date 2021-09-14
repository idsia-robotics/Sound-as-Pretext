import os
import torch
import argparse
import numpy as np
import pandas as pd
from model import NN
from test_pose import test
from itertools import islice
from tqdm import tqdm, trange
from datetime import datetime
from dataset import get_dataset


def train(name, filename, epochs, batch_size, learning_rate, percent, device):
    # Create model folder
    model_path = './model/' + name + '_pretext_' + str(int(percent * 100))
    checkpoint_path = model_path + '/checkpoints'

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Load dataset
    train_dataset = get_dataset(filename, 't1',
                                augment=True, device=device)
    train_2_dataset = get_dataset(filename, 't2',
                                  augment=True, device=device)
    val_dataset = get_dataset(filename, 'val',
                              augment=False, device=device)

    # Model, optimizer & loss
    model = NN(out_channels=12).to(device)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)

    # Train
    history = pd.DataFrame()
    t1_length = float(len(train_dataset))
    t2_length = float(len(train_2_dataset) * percent)
    t2_steps = int(t2_length) // batch_size
    training_steps = int(t1_length + t2_length) // batch_size
    epochs_logger = trange(1, epochs + 1, desc='epoch')
    best_val_loss = 1e9

    for epoch in epochs_logger:
        t1_generator = train_dataset.batches(batch_size, shuffle=True)
        t2_generator = islice(train_2_dataset.batches(batch_size, shuffle=False),
                              t2_steps)
        steps_logger = trange(training_steps, desc='step')

        val_metrics = [np.nan] * 4
        step = 0
        for t1_data in t1_generator:
            im_train = t1_data['rm_s1_camera_image_h264']
            f_train = t1_data['mp3_features']

            model.train()
            optimizer.zero_grad()

            pred = model(im_train)
            loss = loss_function(pred[:, 3:], f_train)

            loss.backward()
            optimizer.step()

            history = history.append({
                'epoch': epoch,
                'step': step,
                'loss': np.nan,
                'loss_pre': loss.item(),
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

            step += 1
            steps_logger.set_postfix_str(log_str)
            steps_logger.update()

        for t2_data in t2_generator:
            y_train = t2_data['optitrack_tello_proj'][:, :3]
            im_train = t2_data['rm_s1_camera_image_h264']
            f_train = t2_data['mp3_features']

            model.train()
            optimizer.zero_grad()

            pred = model(im_train)
            loss = loss_function(pred[:, 3:], f_train)
            loss_pos = loss_function(pred[:, :3], y_train)

            (loss + (t1_length / t2_length) * loss_pos).backward()
            optimizer.step()

            if step == training_steps:
                val_metrics = test(model, val_dataset, batch_size)
                scheduler.step()

            history = history.append({
                'epoch': epoch,
                'step': step,
                'loss': loss_pos.item(),
                'loss_pre': loss.item(),
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

            if step == training_steps:
                log_str += ', VL: %.4f, R2x=%.4f, R2y=%.4f, R2z=%.4f' % tuple(
                    mean_metrics[1:])
                print
                epochs_logger.set_postfix_str(log_str)

            step += 1
            steps_logger.set_postfix_str(log_str)
            steps_logger.update()

        steps_logger.close()
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
    plt.title('pretext')
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
                        default=1e-4)
    parser.add_argument('-p', '--percent', type=float, help='percentage of labeled data to be used',
                        default=1.0)
    parser.add_argument('-d', '--device', type=str, help=argparse.SUPPRESS,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    train(**vars(args))
