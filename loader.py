import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image


"""
=================================================
        LOADING FRAMES AND GROUND TRUTH
=================================================
"""


CWD = os.getcwd()

TOTENSOR_CUDA = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        lambda x: x.cuda(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
TOTENSOR = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


def remove_ds_store(lst):
    return [x for x in lst if x != '.DS_Store']


class MyCustomVideoDataset(Dataset):

    def __init__(self, frames_root, gt, l, w, h, use_gpu):
        self.lst = [os.path.join(frames_root, x) for x in remove_ds_store(os.listdir(frames_root))]
        self.len = l
        self.width, self.height = w, h
        self.gt = self.normalize(gt)
        self.to_tensor = TOTENSOR_CUDA if use_gpu else TOTENSOR
        
    def __getitem__(self, index):
        img = Image.open(self.lst[index])
        return self.gt[index], self.to_tensor(img)

    def __len__(self):
        return self.len
    
    def normalize(self, gt):
        gt[:, 0] /= self.height
        gt[:, 1] /= self.width
        gt[:, 2] /= self.height
        gt[:, 3] /= self.width
        return gt


class VideoBuffer(object):
    
    def __init__(self, title, dataset, num_workers):
        self.title = title
        self.dataset = dataset
        self.num_workers = num_workers
        self.len = self.dataset.len
        self.predictions = []
        self.n_fails = 0

    def get_dataloader(self, T):
        return torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=T,
                                           num_workers=self.num_workers)


def load_videos(titles_list, use_gpu, set_type, num_workers, dir):

    if titles_list:
        print(f"--------------------{set_type}----------------------")

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    DIR = dir

    # all roots to all videos is list
    roots_list = [os.path.join(DIR, "videos", x) for x in titles_list]

    # set of videos, could be Training or Test or Valid
    videos_list = []

    for video_title, root in zip(titles_list, roots_list):

        print(f"video: {video_title}...")
        # load_info
        info_file = os.path.join(root, f"{video_title}.txt")
        with open(info_file) as f:
            info_lines = f.readlines()
            # delete '\n'
            info_lines[0] = info_lines[0].replace("\n", "")
            info_lines[1] = info_lines[1].replace("\n", "")
            sizes = list(info_lines[0].split(" "))
            x_size = int(sizes[0])
            y_size = int(sizes[1])
            n_frames = int(info_lines[1])

        # load ground truth
        gt_path = os.path.join(root, "groundtruth.txt")
        gt_txt = np.loadtxt(gt_path, delimiter=',', dtype=np.float32)
        gt_tens = torch.from_numpy(gt_txt)

        # create dataset
        dataset = MyCustomVideoDataset(frames_root=os.path.join(root, "frames"),
                                       gt=gt_tens, l=n_frames,
                                       w=x_size, h=y_size, use_gpu=use_gpu)

        # create videos buffer wrapper
        vid = VideoBuffer(title=video_title,
                          dataset=dataset,
                          num_workers=num_workers)

        videos_list.append(vid)

    return videos_list


if __name__ == "__main__":
    """Some tests:"""

    use_gpu = False
    T = 25

    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    titles_list = ['Vid_B_cup']
    videos_list = load_videos(titles_list, use_gpu=False,
                              set_type="train",
                              num_workers=0, dir='.')
    vb = videos_list[0]

    for a, b in vb.get_dataloader(50):
        print(a.size(), b.size())



