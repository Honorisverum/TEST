import utils
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import loader


LOAD_FILENAME = 'last.pt'
N_LEFT = 108
N_RIGHT = 371
CWD = os.getcwd()
REINITIALIZE_GAP = 4
ACCURACY_CALC_GAP = 9
NUM_WORKERS = 0
PATH = './sets/test_set.txt'



"""
in loader.py
self.n_fails = 0
        self.predictions = []
        self.first_fail = None
"""



with open(PATH) as f:
    testing_set_titles = f.read().splitlines()
use_gpu = torch.cuda.is_available()
testing_set_videos = loader.load_videos(testing_set_titles, use_gpu, "test set", NUM_WORKERS)




load_path = os.path.join(os.getcwd(), 'weights', LOAD_FILENAME)
full_net = torch.load(load_path)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
full_net = full_net.cuda()





def compute_overlap_threshold_curve(pred, num_frames):
    pt = list(np.linspace(0, 1, 50))
    ans = []
    for threshold in pt:
        val = len([x for x in pred if x > threshold])
        ans.append(val / num_frames)
    return ans, pt


def compute_tests(testing_set_videos, net, use_gpu):
    
    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net.eval()

    for video in testing_set_videos:

        print("testing {}".format(video.title))

        is_initialize = True
        reinitialize_wait = False
        wait_frame = 0
        accuracy_wait_frames = 0

        for i, (gt, frame) in enumerate(video.get_dataloader(1)):

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

            # calculate output
            net.refresh(frame, gt)
            output = net.forward()
            net.pull_gts(output.unsqueeze(0))

            # calculate overlap
            overlap = utils.reward2(output.view(-1), gt.view(-1))

            # check if overlap is 0, else add to the answer
            if overlap == 0:
                if video.first_fail is not None:
                    video.first_fail = i + 1
                video.n_fails += 1
                net.clear()
                reinitialize_wait = True
                wait_frame = REINITIALIZE_GAP
                accuracy_wait_frames = ACCURACY_CALC_GAP
            else:
                if accuracy_wait_frames != 0:
                    accuracy_wait_frames -= 1
                    continue
                video.predictions.append(overlap)
        net.clear()


def compute_metrics(test_set_videos):
    print("RESULTS:")

    all_predictions = []
    all_fails = 0
    all_frames = 0
    eao_dict, eao_nums = {}, {}

    with open(CWD + "/results/report.txt", "w+") as f:

        for video in test_set_videos:

            all_predictions += video.predictions
            all_fails += video.n_fails
            all_frames += video.len

            for j in range(1, video.len + 1):
                if j not in eao_nums.keys():
                    eao_nums[j] = 0
                    eao_dict[j] = 0
                if video.first_fail == None:
                    eao_nums[j] += 1
                    eao_dict[j] += sum(video.predictions[:j]) / j
                elif j < video.first_fail:
                    eao_nums[j] += 1
                    eao_dict[j] += sum(video.predictions[:j]) / j
                else:
                    eao_nums[j] += 1
                    eao_dict[j] += sum(video.predictions) / j

            video_report_format = {
                'video_title': video.title,
                'frames_num': video.len,
                'mean_overlap': round(np.mean(video.predictions).item(), 3) if video.predictions != [] else 0.0,
                'fails': video.n_fails
            }

            f.write("Test for {video_title} with "
                    "{frames_num} frames |"
                    " Mean overlap: {mean_overlap} |"
                    " Number of fails: {fails} \n".format(**video_report_format))

            print("Test for {video_title} with "
                  "{frames_num} frames |"
                  " Mean overlap: {mean_overlap} |"
                  " Number of fails: {fails}".format(**video_report_format))

        f.write("\n\n")

        eao = 0
        for k in range(N_LEFT, N_RIGHT + 1):
            eao += eao_dict[k] / eao_nums[k]
        eao /= N_RIGHT - N_LEFT

        final_report_format = {
            'average_accuracy': round(np.mean(all_predictions).item(), 4),
            'total_fails': all_fails,
            'robustness': round(100.0 * all_fails / all_frames, 4),
            'eao': round(eao, 4)
        }

        f.write("AVERAGE ACCURACY: {average_accuracy}\n"
                "TOTAL FAILS: {total_fails}\n"
                "ROBUSTNESS(s=100): {robustness} \n"
                "EAO: {eao}\n".format(**final_report_format))

        print("AVERAGE ACCURACY: {average_accuracy}\n"
              "TOTAL FAILS: {total_fails}\n"
              "ROBUSTNESS(s=100): {robustness} \n"
              "EAO: {eao}\n".format(**final_report_format))

    # plot overlap curve and save it values
    overlap_threshold_curve, points = compute_overlap_threshold_curve(all_predictions, all_frames)
    eao_curve = [0] * max(eao_dict.keys())
    for k in eao_dict.keys():
        eao_curve[k - 1] = eao_dict[k] / eao_nums[k]

    np.savetxt(CWD + "/results/overlap_threshold.txt", overlap_threshold_curve)
    np.savetxt(CWD + "/results/eao.txt", eao_curve)


compute_tests(testing_set_videos, full_net, use_gpu)
compute_metrics(testing_set_videos)