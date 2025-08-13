import os
import glob
import torch
import cv2
import numpy as np

from pathlib import Path
from typing import List

from FMANet.config import Config
from FMANet.model import FMANet

from img_utils.image import loadImage, writeImage

class Meshroom_Dataset:
    def __init__(self, frame_list, num_seq, ext, dim_max=640, start=0, stop=-1):
        self.num_seq = num_seq
        self.ext = ext
        self.seq_path = self.get_seq_path(frame_list)
        if stop > start:
            self.seq_path = self.seq_path[start:stop].copy()
        self.num_data = len(self.seq_path)
        if self.num_data > 0:
            img, h, w, par, orientation = loadImage(self.seq_path[0][(num_seq-1)//2])
            self.fx = min(dim_max / max(w, h), 1)
            self.win = w
            self.hin = h
            self.par = par
            self.orientation = orientation

    def __getitem__(self, idx):
        # input
        lr_blur_path = self.seq_path[idx]
        lr_blur_seq = [loadImage(path)[0] for path in lr_blur_path]
        
        lr_blur_seq = [cv2.resize(img, None, fx=self.fx, fy=self.fx) for img in lr_blur_seq]
        lr_blur_seq = np.stack(lr_blur_seq, axis=0)
        filename = lr_blur_path[(self.num_seq-1) // 2]
        return self.np2tensor(lr_blur_seq), filename

    def np2tensor(self, x):
        # x shape: [T, H, W, C]
        # reshape to [C, T, H, W]
        ts = (3, 0, 1, 2)
        x = torch.Tensor(x.transpose(ts).astype(float)).mul_(1.0)
        return x

    def get_seq_path(self, frame_list):
        seq_list = []
        extSize = (self.num_seq - 1) // 2
        ext_list_start = frame_list[1:1+extSize].copy()
        ext_list_start.reverse()
        ext_list_stop = frame_list[len(frame_list)-1-extSize:len(frame_list)-1].copy()
        ext_list_stop.reverse()
        ext_list_start.extend(frame_list)
        ext_list_start.extend(ext_list_stop)
        extended_frame_list = ext_list_start.copy()
        start = (self.num_seq - 1) // 2
        end = len(extended_frame_list) - (self.num_seq - 1) // 2
        for i in range(start, end):
            frame_seq = []
            for seq_num in range(self.num_seq):
                frame_seq.append(extended_frame_list[i + seq_num - start])
            seq_list.append(frame_seq)
        return seq_list

    def get_image_ratio(self):
        return self.fx

    def get_image_infos(self):
        return self.hin, self.win, self.orientation, self.par

    def __len__(self):
        return self.num_data


def FMANet_inference(
    frameList: List[str],
    ext: str,
    checkpoint_path: str,
    outPath: str,
    maxDim: int,
    start: int,
    stop: int): 

    num_seq = 3
    nThreads = 8
    config_path = checkpoint_path + '/inference.cfg'

    if not os.path.exists(outPath):
        Path(outPath).mkdir(parents=True, exist_ok=True)

    data = Meshroom_Dataset(frameList, num_seq, ext, maxDim, start, stop)
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(nThreads), pin_memory=True)
    config = Config(config_path)
    model = FMANet(config=config)
    model.cuda()

    D_state_dict = torch.load(checkpoint_path + '/model_D_best.pt')
    model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
    R_state_dict = torch.load(checkpoint_path + '/model_R_best.pt')
    model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])

    model.eval()
    h,w,orientation,par = data.get_image_infos()

    with torch.no_grad():
        for idx, (lr_blur_seq, filename) in enumerate(test_dataloader):
            lr_blur_seq = lr_blur_seq.cuda()

            result_dict = model(lr_blur_seq)
            output = result_dict['output']
            output = output.squeeze(dim=0)
            output = output.cpu().detach().numpy()
            output = np.transpose(output, (1, 2, 0)).copy()

            filename = filename[0]
            filename = os.path.basename(filename)
            filename = os.path.join(outPath, filename)
            writeImage(filename, output, h, w, orientation, par)

