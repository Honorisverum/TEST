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
    out_rewards = torch.zeros(gt.size(0))
    for i in range(gt.size(0)):
        out_rewards[i] = reward2(pred[i], gt[i])
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


if __name__ == '__main__':
    """
    part for testing functions above
    """
    T = 10
    N = 15
    gt = torch.randn(T, 5)
    pred = torch.randn(T, 5)
    print(gt.size())
    print(pred.size())
    ans = custom_loss(gt, pred, 1.0, 1.0, 1.0)
    print(ans)
    print(ans.size())

    """
    def compute_reward(pred, gt, A, B, I):
        #intersect = compute_rewards2(gt, pred)
        intersect = compute_intersect(gt, pred)
        angle = compute_angle_reward(gt, pred)
        beta = compute_beta_reward(gt, pred)
        print(angle.mean(), beta.mean(), intersect.mean())
        #return (A * angle + B * beta + I * intersect).mean()
        return (intersect).mean()
    """

    """
    def compute_intersect(gt, pred, eps=1e-6):
        r = torch.sqrt(torch.sum((gt[:, 0:2] - gt[:, 2:4]) ** 2, dim=1)) / 2
        R = torch.sqrt(torch.sum((pred[:, 0:2] - pred[:, 2:4]) ** 2, dim=1)) / 2
        d = torch.sqrt(torch.sum(((gt[:, 0:2] + gt[:, 2:4] - pred[:, 0:2] - pred[:, 2:4]) / 2) ** 2, dim=1))
        s = d + R + r
        squares = math.pi * (r ** 2 + R ** 2)
        #inter = (r ** 2) * torch.acos((d ** 2 + r ** 2 - R ** 2) / (2 * d * r + eps)) + \
                #(R ** 2) * torch.acos((d ** 2 + R ** 2 - r ** 2) / (2 * d * R + eps)) - \
                #torch.sqrt(s * (s - 2 * d) * (s - 2 * r) * (s - 2 * R)) / 2

        inter = torch.sqrt(s * (s - 2 * d) * (s - 2 * r) * (s - 2 * R)) / 2
        #inter[torch.isnan(inter)] = 0
        return inter
        #return 1 - torch.div(inter, squares - inter)
    """


    """

    print("\n"*6)

    gt = torch.Tensor([-1, -1, 1, 1, 0.5])
    pred = torch.Tensor([-1, -1, 1, 1, 0.5])
    if is_intersect(gt, pred):
        print("YES")
    else:
        print("NO")
        
    """



