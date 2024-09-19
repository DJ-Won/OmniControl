# This code is based on https://github.com/GuyTevet/motion-diffusion-model
from model.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl
from scipy.spatial.transform import Rotation as R

def rot6d_to_euler(poses):
    # Step 1: 将6D旋转表示转换为旋转矩阵
    def rot6d_to_rotmat(poses):
        batch_size = poses.shape[0]
        
        # Split the 6D representation into two 3D vectors
        x_raw = poses[:, 0:3]
        y_raw = poses[:, 3:6]
        
        # Normalize x_raw to get the x-axis of the rotation matrix
        x = x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)
        
        # Make y perpendicular to x
        z = np.cross(x, y_raw)
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        
        # Recompute y to ensure orthogonality
        y = np.cross(z, x)
        
        # Stack x, y, z to form the rotation matrix
        rot_mats = np.stack([x, y, z], axis=-1)
        
        return rot_mats  # shape: (batch_size, 3, 3)

    # Step 2: 从旋转矩阵转换为欧拉角
    rot_mats = rot6d_to_rotmat(poses)
    
    # 将旋转矩阵转为欧拉角（顺序可以是 'xyz', 'zyx' 等，取决于具体需求）
    euler_angles = np.zeros((poses.shape[0], 3))  # Placeholder for the output
    for i in range(poses.shape[0]):
        r = R.from_matrix(rot_mats[i])
        euler_angles[i] = r.as_euler('xyz', degrees=False)  # 返回弧度制的欧拉角
    
    return euler_angles

class npy2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        self.hint = None
        if self.motions['hint'] != []:
            self.hint = self.motions['hint'][self.absl_idx]
        self.rots = None
        self.trans = None

        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            self.motions['motion'] = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][self.absl_idx]

        self.vertices, self.rots, self.trans = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=False)
        self.trans = self.trans[0].numpy().transpose(1,0)
        _,joints,_,fnum = self.rots.shape
        self.rots = self.rots[0].numpy().transpose(2,0,1).reshape(-1,6)
        self.rots = rot6d_to_euler(self.rots)
        self.rots = self.rots.reshape(fnum,-1)
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
        self.vertices += self.root_loc
        # put one the floor y = 0
        floor_height = self.vertices[0].min(0)[0].min(-1)[0][1]
        self.vertices[:, :, 1] -= floor_height

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames].numpy(),
            'text': self.motions['text'][0],
            'length': self.real_num_frames,
            'hint': self.hint,
        }
        np.save(save_path, data_dict)

    def save_amass(self, save_path):
        poses = np.zeros((self.real_num_frames,156))
        poses[...,:72] = self.rots
        trans = self.trans
        data_dict = {
            'poses': poses,
            'trans': trans,
            'betas': 0.0,
            'mocap_framerate': 30,
            'gender': 'neutral',
        }
        np.save(save_path, data_dict)
        return data_dict