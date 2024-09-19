# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import argparse
import os

import sys
sys.path.append("/home/ubuntu/DATA1/dingjuwang/OmniControl")
from visualize import vis_utils
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='results.npy')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()


    npy_path = params.input_path
    out_npy_path = params.input_path.replace('.npy', '_smpl_params.npy')
    assert os.path.exists(npy_path)

    sample_i = 0
    rep_i = 0
    npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                device=params.device, cuda=params.cuda)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    amass = npy2obj.save_amass(out_npy_path)
    
