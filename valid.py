"""
=================================================
                VALIDATION PHASE
    COMPUTE FORWARD PASS ON VALIDATION VIDEOS
=================================================
"""

import utils
import numpy as np


REINITIALIZE_GAP = 4
ACCURACY_CALC_GAP = 9
INFO_STRING = "Title: {video_title} | " \
              "Mean Reward: {mean_reward} | " \
              "Number of frames: {n_frames} | " \
              "Number of fails: {n_fails} |"
OVERALL_INFO_STRING = "Total Mean overlap: " \
                      "{overlap} |" \
                      "Total Robustness: " \
                      "{robustness}"


def compute_validation(validating_set_videos, net, use_gpu):

    print("Validation:")

    net.eval()

    all_overlaps, all_fails, all_frames = [], 0, 0

    for video in validating_set_videos:

        reinitialize_wait = False
        wait_frame = 0
        accuracy_wait_frames = 0

        for gt, frame in video.get_dataloader(1):

            if use_gpu:
                gt = gt.cuda().squeeze()
                frame = frame.cuda().squeeze()

            # check if it is still waiting time
            if reinitialize_wait:
                wait_frame -= 1
                accuracy_wait_frames -= 1
                if wait_frame == 0:
                    reinitialize_wait = False
                continue

            # processing
            net.refresh(frame, gt)
            output = net.forward()
            net.pull_gts(output.unsqueeze(0))

            # calculate overlap
            overlap = utils.reward2(output.view(-1), gt.view(-1))

            # check if overlap is 0, else add to the answer
            if overlap == 0:
                net.clear()
                video.n_fails += 1
                reinitialize_wait = True
                wait_frame = REINITIALIZE_GAP
                accuracy_wait_frames = ACCURACY_CALC_GAP
            else:
                if accuracy_wait_frames != 0:
                    accuracy_wait_frames -= 1
                    continue
                video.predictions.append(overlap)

        all_fails += video.n_fails
        all_frames += video.len
        all_overlaps += video.predictions
        ep_video_overlap = np.mean(video.predictions).item() if video.predictions != [] else 0

        # print info
        info_format = {
            'video_title': video.title,
            'mean_reward': round(ep_video_overlap, 6),
            'n_frames': video.len,
            'n_fails': video.n_fails
        }; print(INFO_STRING.format(**info_format))

        # clear for next episode
        video.n_fails = 0
        video.predictions = []

    ep_overlap = np.mean(all_overlaps).item() if all_overlaps != [] else 0
    ep_robustness = 100.0 * all_fails / all_frames

    # print overall info
    overall_info_format = {
        'overlap': round(ep_overlap, 6),
        'robustness': round(ep_robustness, 6)
    }; print(OVERALL_INFO_STRING.format(**overall_info_format))

    print("***********************************************\n\n")

    return net





