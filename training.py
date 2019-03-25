"""
=================================================
                TRAINING PHASE
TRAIN MODEL ON TRAIN VIDEOS USING DEEP RL ALGORITHM
=================================================
"""


import utils
import torch
import os
import random
import torch.nn as nn
import math
import valid


TRAIN_INFO_STRING = "Title {video_title} | " \
                    "Mean Coord diff: {mean_coord} |"


def train(training_set_videos, net, optimizer, save_every,
          T, epochs, use_gpu, validating_set_videos):

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    save_path = os.path.join(os.getcwd(), 'weights', 'last.pt')

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    net.train()

    for epoch in range(1, epochs + 1):

        print(f"Epoch: {epoch}")

        criter = torch.nn.SmoothL1Loss()

        ep_rew = 0

        random.shuffle(training_set_videos)

        scheduler.step()

        net.train()

        for i, video in enumerate(training_set_videos):

            rew = 0

            for gt, images in video.get_dataloader(T):

                if use_gpu:
                    gt = gt.cuda()
                    images = images.cuda()

                # Clear gradients
                optimizer.zero_grad()

                outputs = torch.zeros(images.size(0), 5)

                for i, (single_gt, single_frame) in enumerate(zip(gt, images)):
                    net.refresh(single_frame, single_gt)
                    outputs[i] = net.forward()
                    net.pull_gts(single_gt.unsqueeze(0))

                # compute loss
                loss = criter(outputs, gt)
                rew = loss.item() * images.size(0)
                
                loss.backward()

                # Updating parameters
                optimizer.step()

            # clear data before next video
            net.clear()

            ep_rew += rew / video.len
            # print info for ep for this video
            iteration_info_format = {
                'video_title': video.title,
                'mean_coord': round(math.sqrt(rew / video.len / 5), 6)
            }; print(TRAIN_INFO_STRING.format(**iteration_info_format))

        if save_every is not None:
            if not epoch % save_every:
                torch.save(net, save_path)

        print("Mean epoch coordinate diff:", round(math.sqrt(ep_rew / len(training_set_videos) / 5), 6))

        # for divide info on ep blocks
        print("===============================================")

        if not epoch % 5:
            net = valid.compute_validation(validating_set_videos=validating_set_videos, net=net, use_gpu=use_gpu)

    # final save
    if save_every is not None:
        torch.save(net, save_path)

    return net
