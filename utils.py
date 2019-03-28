import math
import torch
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geos import TopologicalError
import os
import psutil


def cpuStats(info=None):
    if info is not None:
        print(info, end="  |  ")
    print("cpu percent:", psutil.cpu_percent(), end="  |  ")
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', round(memoryUse, 2))


def gpuStats():
    """
    bash: sudo pip install gpustat
    python: import os
    """
    _ = os.system('gpustat -cp')



"""
=================================================
    SAMPLE PREDICTIONS AND CALCULATE REWARDS
=================================================
"""


def rotate_anticlockwise(vec, cos, sin):
    ans = np.zeros(2)
    ans[0] = vec[0] * cos - vec[1] * sin
    ans[1] = vec[0] * sin + vec[1] * cos
    return ans


def rotate_clockwise(vec, cos, sin):
    ans = np.zeros(2)
    ans[0] = vec[0] * cos + vec[1] * sin
    ans[1] = - vec[0] * sin + vec[1] * cos
    return ans


def extract_coord(tens):
    tens = tens.detach().cpu().numpy()
    v1 = tens[0:2]
    v3 = tens[2:4]
    a4 = (tens[4] + 1/2) * math.pi
    cos_a = math.cos(a4)
    sin_a = math.sin(a4)
    vc = (v1 + v3) / 2
    vc3 = v3 - vc
    v2 = vc + rotate_anticlockwise(vc3, -cos_a, sin_a)
    v4 = vc + rotate_clockwise(vc3, cos_a, sin_a)
    return [tuple(v1), tuple(v2), tuple(v3), tuple(v4)]


def reward2(pred, gt):
    if abs(pred[4].item()) >= 0.5:
        return 0
    pred_poly = Polygon(extract_coord(pred))
    gt_poly = Polygon(extract_coord(gt))
    try:
        inter_area = pred_poly.intersection(gt_poly).area
    except TopologicalError:
        inter_area = 0
    except ValueError:
        inter_area = 0

    if not inter_area:
        return 0
    else:
        return inter_area / (pred_poly.area + gt_poly.area - inter_area)


def compute_rewards2(gt, pred):
    """
    compute rewards at late training stage
    :param predictions: torch(T, 5)
    :param ground_truth: torch(T, N, 5)
    :return: torch(T, N)
    """
    out_rewards = torch.zeros(gt.size(0), pred.size(1))
    for i in range(gt.size(0)):
        for j in range(pred.size(1)):
            out_rewards[i][j] = reward2(pred[i][j], gt[i])
    return out_rewards



"""
=================================================
        BASELINES, LOSS AND SIGMA
=================================================
"""


# weights for weight change curve
def compute_weights(net):
    ans = 0
    for param in net.parameters():
        if param.requires_grad:
            ans += math.sqrt(torch.sum(param ** 2).item())
    return ans


# weights grad for weight grad change curve
def compute_weights_grad(net):
    ans = 0
    for param in net.parameters():
        if param.requires_grad:
            ans += math.sqrt(torch.sum(param.grad ** 2).item())
    return ans


def custom_loss(pred, gt, A, B, I):
    """
    compute rewards
    :param gt: torch(T, 5)
    :param pred: torch(T, 5)
    :return: torch(1)
    """
    distances = compute_distances(gt, pred)
    angle = compute_angle_reward(gt, pred)
    beta = compute_beta_reward(gt, pred)
    # print(f"dist: {distances.mean()}, angle: {angle.mean()}, beta: {beta.mean()}")
    return (A * angle + B * beta + I * distances).mean()


def compute_distances(gt, pred):
    gt_c = (gt[:, 0:2] + gt[:, 2:4]) / 2
    pred_c = (pred[:, 0:2] + pred[:, 2:4]) / 2
    return torch.sum((gt_c - pred_c) ** 2, dim=1)


def compute_angle_reward(gt, pred):
    v13_gt = gt[:, 2:4] - gt[:, 0:2]
    v13_pred = pred[:, 2:4] - pred[:, 0:2]
    return (1 - torch.sum(v13_gt*v13_pred, dim=1)
            / torch.sqrt(torch.sum(v13_gt ** 2, dim=1))
            / torch.sqrt(torch.sum(v13_pred ** 2, dim=1)))


def compute_beta_reward(gt, pred):
    return (gt[:, 4] - pred[:, 4]) ** 2



"""
=================================================
                        RL
=================================================
"""


def sample_predictions(out, sigma, N):
    """
    sample predictions
    :param out: torch(batch_size, 5)
    :param sigma: float
    :param N: int
    :return: torch(batch_size, N, 5)
    """
    ans = sigma * torch.randn(out.size(0), N, 5)
    return ans + out.unsqueeze(1).expand_as(ans)


def compute_rewards1(gt, pred):
    """
    compute rewards at early training stage
    :param gt: torch(batch_size, 5)
    :param pred: torch(batch_size, N, 5)
    :return: torch(batch_size, N)
    """
    gt = gt.unsqueeze(1).expand_as(pred)
    ans = torch.abs(pred - gt)
    return - (ans.mean(dim=2) + ans.max(dim=2)[0]) / 2


def compute_baseline(rew):
    """
    compute baseline
    :param rew: torch(batch_size, N)
    :return: torch(1)
    """
    rew = torch.sum(rew, dim=0)
    return torch.mean(rew, dim=0)


def calculate_diff(out, pred, sig):
    """
    calculate differences
    :param out: torch(batch_size, 5)
    :param pred: torch(batch_size, N, 5)
    :param sig: float
    :return: torch(batch_size, N, 5)
    """
    out = out.unsqueeze(1).expand_as(pred)
    df = (out - pred) / sig
    return df


def compute_loss(rew, bs, out, diff):
    """
    compute loss
    :param rew: torch(batch_size, N)
    :param bs: torch(1)
    :param out: torch(batch_size, 5)
    :param diff: torch(batch_size, N, 5)
    :return:
    """
    rew = torch.sum(rew, dim=0) - bs
    rew = rew.unsqueeze(0).unsqueeze(2)
    rew = rew.expand_as(diff)
    out = out.unsqueeze(1).expand_as(diff)
    return torch.sum(diff * out * rew)


def rl_loss(out, gt, epoch_ratio):
    """
    reinforsment learning loss
    :param out: torch(batch_size, 5)
    :param gt: torch(batch_size, 5)
    :return: torch(1)
    """
    sigma = 0.01
    N = 10
    predictions = sample_predictions(out, sigma, N).detach()
    rewards = compute_rewards1(gt, predictions) if epoch_ratio <= 0.5 else compute_rewards2(gt, predictions)
    baseline = compute_baseline(rewards)
    differences = calculate_diff(out, predictions, sigma).detach()
    return compute_loss(rewards, baseline, out, differences), baseline.item()


if __name__ == '__main__':
    " part for testing "

    T = 10
    N = 15
    gt = torch.randn(T, 5)
    pred = torch.randn(T, 5)
    print(gt.size())
    print(pred.size())
    ans = custom_loss(gt, pred, 1.0, 1.0, 1.0)
    print(ans)
    print(ans.size())





