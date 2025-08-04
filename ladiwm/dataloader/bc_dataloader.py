import numpy as np
import random
import torch
from torchvision import transforms
from einops import rearrange
from ladiwm.dataloader.base_dataset import BaseDataset
from ladiwm.utils.flow_utils import sample_tracks_nearest_to_grids
import os
from torch.utils.data import Dataset
from natsort import natsorted
from ladiwm.dataloader.utils import ImgTrackColorJitter, ImgViewDiffTranslationAug, ImgViewDiffTranslationAug2, load_rgb
import json
from ladiwm.utils.transform_utils import quat2axisangle, axisangle2quat, quat2mat, mat2quat, matrix_inverse, mat2euler, euler2mat, \
    axisangle2quat_torch, quat2axisangle_torch


class BCDataset(BaseDataset):
    def __init__(self, track_obs_fs=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        task_embs = demo["root"]["task_emb_bert"]
        extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
                        demo['root']['extra_states'].items()}

        return obs, track_transformer_obs, track, task_embs, actions, extra_states

class BCDataset2(BaseDataset):
    def __init__(self, track_obs_fs=1, pred_frame=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs
        self.pred_frame = pred_frame

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            frames = demo['root'][view]["video"][image_indices]
            if len(frames) < num_frames:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_list_from_disk(self, demo_id, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            image_indices = np.arange(time_offset + 1 - num_frames, time_offset + 1)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
            # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # t h w c
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset, backward=True))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, time_offset, backward=True))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset, backward=True))  # t c h w
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        task_embs = demo["root"]["task_emb_bert"]
        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        extra_states = {}
        demo_length = demo["root"][view]["video"].shape[0]
        for k, v in demo['root']['extra_states'].items():
            image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            values = v[image_indices]
            if len(values) < self.frame_stack:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
                values = torch.cat([padding_values, values], dim=0)
            extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]

        return obs, track_transformer_obs, track, task_embs, actions, extra_states

class BCDataset3(BaseDataset):
    def __init__(self, track_obs_fs=1, his_frame=4, pred_frame=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs
        self.pred_frame = pred_frame
        self.his_frame = his_frame

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            frames = demo['root'][view]["video"][image_indices]
            if len(frames) < num_frames:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_list_from_disk(self, demo_id, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            image_indices = np.arange(time_offset + 1 - num_frames, time_offset + 1)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
            # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # t h w c
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset, backward=True))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, time_offset, backward=True))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.load_h5(demo_pth)
            task_text = demo_pth.split('/')[-3][:-5].replace('_', ' ')
            demo['task_text'] = task_text
            demo = self.process_demo(demo)
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset, backward=True))  # t c h w
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        task_embs = demo["task_text"]

        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        extra_states = {}
        demo_length = demo["root"][view]["video"].shape[0]
        for k, v in demo['root']['extra_states'].items():
            image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            values = v[image_indices]
            if len(values) < self.frame_stack:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
                values = torch.cat([padding_values, values], dim=0)
            if k == 'ee_states':
                values_ori = values[:, 3:6]
                values_quat = axisangle2quat_torch(values_ori)
                negetive_mask = values_quat[:, -1] < 0
                values_quat[negetive_mask] = torch.negative(values_quat[negetive_mask])
                values_ori = quat2axisangle_torch(values_quat)
                values[:, 3:6] = values_ori
            extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]

        return obs, track_transformer_obs, track, task_embs, actions, extra_states


class BCDatasetIDM(BaseDataset):
    def __init__(self, track_obs_fs=1, his_frame=4, pred_frame=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs
        self.pred_frame = pred_frame
        self.his_frame = his_frame

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        image_indices = np.arange(max(time_offset, 0), time_offset + self.pred_frame + 1)
        image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
        frames = demo['root'][view]["video"][image_indices]
        assert len(frames) == num_frames;
        # if len(frames) < num_frames:
        #     # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
        #     padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
        #     frames = torch.cat([padding_frames, frames], dim=0)
        return frames


    def _load_image_list_from_disk(self, demo_id, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)
        image_indices = np.arange(time_offset, time_offset + self.pred_frame + 1)
        image_indices = np.clip(image_indices, a_min=0, a_max=demo_length - 1)
        frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
        # if backward:
        #     # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
        #     # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
        #     image_indices = np.arange(time_offset + 1 - num_frames, time_offset + 1)
        #     image_indices = np.clip(image_indices, a_min=0, a_max=demo_length - 1)
        #     frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
        #     # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        # else:
        #     image_indices = np.arange(time_offset, time_offset + num_frames)
        #     image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
        #     frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # t h w c
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            # all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset, backward=True))  # t c h w
                    # all_view_track_transformer_frames.append(
                    #     torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    # )  # t tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, time_offset, backward=True))  # t c h w
                    # all_view_track_transformer_frames.append(
                    #     torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    # )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.load_h5(demo_pth)
            task_text = demo_pth.split('/')[-3][:-5].replace('_', ' ')
            demo['task_text'] = task_text
            demo = self.process_demo(demo)
            all_view_frames = []
            # all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset, backward=True))  # t c h w
                # all_view_track_transformer_frames.append(
                #     torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                # )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        # track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w
        track_transformer_obs = torch.rand(2, 1, 1, 3, 128, 128)

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        task_embs = demo["task_text"]

        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        extra_states = {}
        demo_length = demo["root"][view]["video"].shape[0]
        for k, v in demo['root']['extra_states'].items():
            image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            values = v[image_indices]
            if len(values) < self.frame_stack:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
                values = torch.cat([padding_values, values], dim=0)
            if k == 'ee_states':
                values_ori = values[:, 3:6]
                values_quat = axisangle2quat_torch(values_ori)
                negetive_mask = values_quat[:, -1] < 0
                values_quat[negetive_mask] = torch.negative(values_quat[negetive_mask])
                values_ori = quat2axisangle_torch(values_quat)
                values[:, 3:6] = values_ori
            extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]

        return obs, track_transformer_obs, track, task_embs, actions, extra_states


class BCDataset3Real(Dataset):
    def __init__(self, dataset_dir, track_obs_fs=1, pred_frame=4, his_frame=4, img_size=128,
                 num_demos=None, augment_track=True, aug_prob=0., *args, **kwargs):
        super().__init__()
        self._index_to_view_id = {}
        self.track_obs_fs = track_obs_fs
        self.dataset_dir = dataset_dir
        self.num_demos = num_demos
        self.augment_track = augment_track
        self.aug_prob = aug_prob
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = (img_size[0], img_size[1])
        self.buffer_fns = []
        task_names = os.listdir(self.dataset_dir)
        self.dataset_dir = [os.path.join(self.dataset_dir, task_name) for dir_idx, task_name in enumerate(task_names)]
        for dir_idx, d in enumerate(self.dataset_dir):
            fn_list = os.listdir(d)
            fn_list = [os.path.join(d, fn) for fn in fn_list]
            # fn_list = glob(os.path.join(d, "*.hdf5"))
            fn_list = natsorted(fn_list)
            if self.num_demos is None:
                n_demo = len(fn_list)
            else:
                assert 0 < self.num_demos <= 1, "num_demos means the ratio of training data among all the demos."
                n_demo = int(len(fn_list) * self.num_demos)
            for fn in fn_list[:n_demo]:
                self.buffer_fns.append(fn)

        assert (len(self.buffer_fns) > 0)
        print(f"found {len(self.buffer_fns)} trajectories in the specified folders: {self.dataset_dir}")
        self._cache = []
        self._index_to_demo_id, self._demo_id_to_path, self._demo_id_to_start_indices, self._demo_id_to_demo_length, \
            self._demo_id_to_ee_pos_path, self._demo_id_to_agent_view_path, self._demo_id_to_hand_view_path \
            = {}, {}, {}, {}, {}, {}, {}
        self.load_demo_info()

        self.his_frame = his_frame
        self.pred_frame = pred_frame
        self.load_image_func = load_rgb
        self.ori_scale = np.array([0.5, 0.5, 0.5])
        self.pos_scale = np.array([0.05, 0.05, 0.05])
        self.views = ['agent_view', 'hand_view']

        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        ])

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            # demo = self.load_h5(fn)

            # if self.views is None:
            #     self.views = ['agent_view', 'hand_view']
            # self.views = list(demo["root"].keys())
            # self.views.remove("actions")
            # self.views.remove("task_emb_bert")
            # self.views.remove("extra_states")
            # self.views.sort()
            filenames = os.listdir(fn)
            ee_pos_names = [filename for filename in filenames if filename.endswith(".json")]
            ee_pos_names = sorted(ee_pos_names)
            agent_view_img_names = ['agent_view' + filename[-25:-5] + '.png' for filename in ee_pos_names]
            hand_view_img_names = ['hand_view' + filename[-25:-5] + '.png' for filename in ee_pos_names]
            # ee_pos_path = glob(os.path.join(fn, '*.json'))
            # ee_pos_path = sorted(ee_pos_path)
            ee_pos_path = [os.path.join(fn, ee_pos_name) for ee_pos_name in ee_pos_names]
            agent_view_img_path = [os.path.join(fn, agent_view_img_name) for agent_view_img_name in
                                   agent_view_img_names]
            hand_view_img_path = [os.path.join(fn, hand_view_img_name) for hand_view_img_name in hand_view_img_names]
            # demo_len = demo["root"][self.views[0]]["video"][0].shape[0]
            demo_len = len(ee_pos_path)
            assert demo_len == len(ee_pos_path) == len(agent_view_img_path) == len(hand_view_img_path)

            demo = {
                "root": {},
                "extra_states": {},
                "task_emb_bert": {},
                "actions": {},
            }

            # if self.cache_all:
            #     demo = self.process_demo(demo)
            #     for v in self.views:
            #         del demo["root"][v]["video"]
            #     self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            self._demo_id_to_ee_pos_path[demo_idx] = ee_pos_path
            self._demo_id_to_agent_view_path[demo_idx] = agent_view_img_path
            self._demo_id_to_hand_view_path[demo_idx] = hand_view_img_path
            start_idx += demo_len

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def __len__(self):
        return len(self._index_to_demo_id)

    def _load_image_list_from_path(self, demo_id, time_offset, num_frames=None, backward=True):
        demo_path = self._demo_id_to_path[demo_id]
        ee_pos_path = self._demo_id_to_ee_pos_path[demo_id]
        agent_view_img_path = self._demo_id_to_agent_view_path[demo_id]
        hand_view_img_path = self._demo_id_to_hand_view_path[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_name = os.path.basename(demo_path).split(".")[0]
        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames_agent = [self.load_image_func(agent_view_img_path[img_idx]) for img_idx in
                            his_indices]
        his_frames_agent = np.stack(his_frames_agent)  # t h w c
        his_frames_agent = torch.Tensor(his_frames_agent)
        his_frames_agent = rearrange(his_frames_agent, "t h w c -> t c h w")
        his_frames_agent = torch.nn.functional.interpolate(his_frames_agent, size=self.img_size, mode='bilinear')

        his_frames_hand = [self.load_image_func(hand_view_img_path[img_idx]) for img_idx in
                           his_indices]
        his_frames_hand = np.stack(his_frames_hand)  # t h w c
        his_frames_hand = torch.Tensor(his_frames_hand)
        his_frames_hand = rearrange(his_frames_hand, "t h w c -> t c h w")
        his_frames_hand = torch.nn.functional.interpolate(his_frames_hand, size=self.img_size, mode='bilinear')
        frames = torch.stack([his_frames_agent, his_frames_hand], dim=0)  # v t c h w

        action_indices = np.arange(time_offset, time_offset + self.pred_frame + 1)
        action_indices = np.clip(action_indices, a_min=0, a_max=demo_length - 1)
        actions = [np.array(json.loads(open(ee_pos_path[img_idx], 'r').read())) for img_idx in action_indices]
        actions_gri = [action[-1] for action in actions]
        actions_gri = np.array(actions_gri[1:]).reshape(-1, 1)
        actions_pos = [action[:3] for action in actions]
        actions_ori = [action[3:6] for action in actions]
        # actions_ori = [axisangle2quat(action) for action in actions_ori]
        # actions_ori = [quat2mat(action) for action in actions_ori]
        actions_ori = [euler2mat(action) for action in actions_ori]
        base_axisangle = actions_ori[:-1].copy()
        actions_ori = np.array(actions_ori)
        actions_delta_ori = []
        for i in range(len(actions_ori) - 1):
            ad = np.dot(actions_ori[i + 1], matrix_inverse(actions_ori[i]))
            ad = mat2quat(ad)
            ad = quat2axisangle(ad)
            ad = ad / self.ori_scale
            actions_delta_ori.append(ad)
        actions_delta_pos = []
        for i in range(len(actions_pos) - 1):
            ad = actions_pos[i + 1] - actions_pos[i]
            ad = ad / self.pos_scale
            actions_delta_pos.append(ad)
        actions_delta = np.concatenate([np.array(actions_delta_pos), np.array(actions_delta_ori), actions_gri],
                                       axis=-1).astype(np.float32)

        action_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        action_indices = np.clip(action_indices, a_min=0, a_max=demo_length - 1)
        actions = [np.array(json.loads(open(ee_pos_path[img_idx], 'r').read())) for img_idx in action_indices]
        actions_pos = [action[:3] for action in actions]
        actions_ori = [action[3:6] for action in actions]
        actions_ori = [euler2mat(action) for action in actions_ori]
        actions_ori = [mat2quat(action) for action in actions_ori]
        actions_ori = [quat2axisangle(action) for action in actions_ori]
        actions_gri = [action[-1] for action in actions]
        ee_states = np.concatenate([np.array(actions_pos), np.array(actions_ori), np.array(actions_gri)[..., None]], axis=-1).astype(np.float32)
        ee_states =  torch.from_numpy(ee_states)
        if random.random() >= 0.5:
            # random_pos_mean = ee_states[:, :3].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            # random_pos = torch.randn_like(ee_states[:, :3]) * random_pos_mean * 0.02
            # random_ori_mean = ee_states[:, 3:6].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            # random_ori = torch.randn_like(ee_states[:, 3:6]) * random_ori_mean * 0.02
            random_mean = ee_states[:, :6].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            ee_states[:, :6] = ee_states[:, :6] + torch.randn_like(ee_states[:, :6]) * 0.01 #* random_mean * 0.1

        # mat to axisangle
        base_axisangle = [mat2quat(axa) for axa in base_axisangle]
        base_axisangle = [quat2axisangle(axa) for axa in base_axisangle]
        base_axisangle = torch.from_numpy(np.array(base_axisangle))

        states = {
            'ee_states': ee_states,
            'base_axisangle': base_axisangle,
        }
        return frames, actions_delta, states


    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(demo_path)
        demo_length = self._demo_id_to_demo_length[demo_id]
        task_name = os.path.basename(demo_parent_dir).replace('_', ' ')
        demo_name = os.path.basename(demo_path).split(".")[0]
        task_embs = task_name
        time_offset = index - demo_start_index

        # all_view_frames = []
        # all_view_track_transformer_frames = []
        obs, actions, extra_states = self._load_image_list_from_path(demo_id, time_offset, backward=True)

        # track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        track = torch.rand(2, obs.shape[1], 16, 10, 2)  # v t track_len n 2
        # vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        # track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w
        track_transformer_obs = torch.rand(3, 128, 128)

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        # sample_track, sample_vi = [], []
        # for i in range(len(self.views)):
        #     sample_track_per_time, sample_vi_per_time = [], []
        #     for t in range(self.frame_stack):
        #         track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
        #         sample_track_per_time.append(track_i_t)
        #         sample_vi_per_time.append(vi_i_t)
        #     sample_track.append(torch.stack(sample_track_per_time, dim=0))
        #     sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        # track = torch.stack(sample_track, dim=0)
        # vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        # task_embs = demo["task_text"]

        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        # actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        # extra_states = {}
        # demo_length = demo["root"][view]["video"].shape[0]
        # for k, v in demo['root']['extra_states'].items():
        #     image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
        #     image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
        #     values = v[image_indices]
        #     if len(values) < self.frame_stack:
        #         # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
        #         padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
        #         values = torch.cat([padding_values, values], dim=0)
        #     extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]

        return obs, track_transformer_obs, track, task_embs, actions, extra_states


class BCDataset3Real2(Dataset):
    def __init__(self, dataset_dir, track_obs_fs=1, pred_frame=4, his_frame=4, img_size=128,
                 num_demos=None, augment_track=True, aug_prob=0., *args, **kwargs):
        super().__init__()
        self._index_to_view_id = {}
        self.track_obs_fs = track_obs_fs
        self.dataset_dir = dataset_dir
        self.num_demos = num_demos
        self.augment_track = augment_track
        self.aug_prob = aug_prob
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if len(img_size) == 2:
            self.img_size = (img_size[0], img_size[1])
        self.buffer_fns = []
        task_names = os.listdir(self.dataset_dir)
        self.dataset_dir = [os.path.join(self.dataset_dir, task_name) for dir_idx, task_name in enumerate(task_names)]
        for dir_idx, d in enumerate(self.dataset_dir):
            fn_list = os.listdir(d)
            fn_list = [os.path.join(d, fn) for fn in fn_list]
            # fn_list = glob(os.path.join(d, "*.hdf5"))
            fn_list = natsorted(fn_list)
            if self.num_demos is None:
                n_demo = len(fn_list)
            else:
                assert 0 < self.num_demos <= 1, "num_demos means the ratio of training data among all the demos."
                n_demo = int(len(fn_list) * self.num_demos)
            for fn in fn_list[:n_demo]:
                self.buffer_fns.append(fn)

        assert (len(self.buffer_fns) > 0)
        print(f"found {len(self.buffer_fns)} trajectories in the specified folders: {self.dataset_dir}")
        self._cache = []
        self._index_to_demo_id, self._demo_id_to_path, self._demo_id_to_start_indices, self._demo_id_to_demo_length, \
            self._demo_id_to_ee_pos_path, self._demo_id_to_agent_view_path, self._demo_id_to_hand_view_path \
            = {}, {}, {}, {}, {}, {}, {}
        self.load_demo_info()

        self.his_frame = his_frame
        self.pred_frame = pred_frame
        self.load_image_func = load_rgb
        self.ori_scale = np.array([0.5, 0.5, 0.5])
        self.pos_scale = np.array([0.05, 0.05, 0.05])
        self.views = ['agent_view', 'hand_view']

        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        ])

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            # demo = self.load_h5(fn)

            # if self.views is None:
            #     self.views = ['agent_view', 'hand_view']
            # self.views = list(demo["root"].keys())
            # self.views.remove("actions")
            # self.views.remove("task_emb_bert")
            # self.views.remove("extra_states")
            # self.views.sort()
            filenames = os.listdir(fn)
            ee_pos_names = [filename for filename in filenames if filename.endswith(".json")]
            ee_pos_names = sorted(ee_pos_names)
            agent_view_img_names = ['agent_view' + filename[-25:-5] + '.png' for filename in ee_pos_names]
            hand_view_img_names = ['hand_view' + filename[-25:-5] + '.png' for filename in ee_pos_names]
            # ee_pos_path = glob(os.path.join(fn, '*.json'))
            # ee_pos_path = sorted(ee_pos_path)
            ee_pos_path = [os.path.join(fn, ee_pos_name) for ee_pos_name in ee_pos_names]
            agent_view_img_path = [os.path.join(fn, agent_view_img_name) for agent_view_img_name in
                                   agent_view_img_names]
            hand_view_img_path = [os.path.join(fn, hand_view_img_name) for hand_view_img_name in hand_view_img_names]
            # demo_len = demo["root"][self.views[0]]["video"][0].shape[0]
            demo_len = len(ee_pos_path)
            assert demo_len == len(ee_pos_path) == len(agent_view_img_path) == len(hand_view_img_path)

            demo = {
                "root": {},
                "extra_states": {},
                "task_emb_bert": {},
                "actions": {},
            }

            # if self.cache_all:
            #     demo = self.process_demo(demo)
            #     for v in self.views:
            #         del demo["root"][v]["video"]
            #     self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            self._demo_id_to_ee_pos_path[demo_idx] = ee_pos_path
            self._demo_id_to_agent_view_path[demo_idx] = agent_view_img_path
            self._demo_id_to_hand_view_path[demo_idx] = hand_view_img_path
            start_idx += demo_len

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def __len__(self):
        return len(self._index_to_demo_id)

    def _load_image_list_from_path(self, demo_id, time_offset, num_frames=None, backward=True):
        demo_path = self._demo_id_to_path[demo_id]
        ee_pos_path = self._demo_id_to_ee_pos_path[demo_id]
        agent_view_img_path = self._demo_id_to_agent_view_path[demo_id]
        hand_view_img_path = self._demo_id_to_hand_view_path[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_name = os.path.basename(demo_path).split(".")[0]
        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames_agent = [self.load_image_func(agent_view_img_path[img_idx]) for img_idx in
                            his_indices]
        his_frames_agent = np.stack(his_frames_agent)  # t h w c
        his_frames_agent = torch.Tensor(his_frames_agent)
        his_frames_agent = rearrange(his_frames_agent, "t h w c -> t c h w")
        his_frames_agent = torch.nn.functional.interpolate(his_frames_agent, size=self.img_size, mode='bilinear')

        his_frames_hand = [self.load_image_func(hand_view_img_path[img_idx]) for img_idx in
                           his_indices]
        his_frames_hand = np.stack(his_frames_hand)  # t h w c
        his_frames_hand = torch.Tensor(his_frames_hand)
        his_frames_hand = rearrange(his_frames_hand, "t h w c -> t c h w")
        his_frames_hand = torch.nn.functional.interpolate(his_frames_hand, size=self.img_size, mode='bilinear')
        frames = torch.stack([his_frames_agent, his_frames_hand], dim=0)  # v t c h w

        action_indices = np.arange(time_offset, time_offset + self.pred_frame + 1)
        action_indices = np.clip(action_indices, a_min=0, a_max=demo_length - 1)
        labels = [json.loads(open(ee_pos_path[img_idx], 'r').read()) for img_idx in action_indices]
        actions = np.array([label['ee_pose'] for label in labels])
        actions_gri = [action[-1] for action in actions]
        actions_gri = np.array(actions_gri[1:]).reshape(-1, 1)
        actions_pos = [action[:3] for action in actions]
        actions_ori = [action[3:6] for action in actions]
        # actions_ori = [axisangle2quat(action) for action in actions_ori]
        # actions_ori = [quat2mat(action) for action in actions_ori]
        actions_ori = [euler2mat(action) for action in actions_ori]
        base_axisangle = actions_ori[:-1].copy()
        actions_ori = np.array(actions_ori)
        actions_delta_ori = []
        for i in range(len(actions_ori) - 1):
            ad = np.dot(actions_ori[i + 1], matrix_inverse(actions_ori[i]))
            ad = mat2quat(ad)
            ad = quat2axisangle(ad)
            ad = ad / self.ori_scale
            actions_delta_ori.append(ad)
        actions_delta_pos = []
        for i in range(len(actions_pos) - 1):
            ad = actions_pos[i + 1] - actions_pos[i]
            ad = ad / self.pos_scale
            actions_delta_pos.append(ad)
        actions_delta = np.concatenate([np.array(actions_delta_pos), np.array(actions_delta_ori), actions_gri],
                                       axis=-1).astype(np.float32)

        action_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        action_indices = np.clip(action_indices, a_min=0, a_max=demo_length - 1)
        labels = [json.loads(open(ee_pos_path[img_idx], 'r').read()) for img_idx in action_indices]
        ee_states = np.array([label['ee_pose'] for label in labels])
        ee_pos = [ee[:3] for ee in ee_states]
        ee_ori = [ee[3:6] for ee in ee_states]
        ee_ori = [euler2mat(ee) for ee in ee_ori]
        ee_ori = [mat2quat(ee) for ee in ee_ori]
        ee_ori = [quat2axisangle(ee) for ee in ee_ori]
        ee_gri = [ee[-1] for ee in ee_states]
        ee_states = np.concatenate([np.array(ee_pos), np.array(ee_ori), np.array(ee_gri)[..., None]], axis=-1).astype(np.float32)
        ee_states =  torch.from_numpy(ee_states)

        jo_states = np.array([label['joint_states'] for label in labels])
        jo_states = torch.from_numpy(jo_states)
        if random.random() >= 0.5:
            # random_pos_mean = ee_states[:, :3].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            # random_pos = torch.randn_like(ee_states[:, :3]) * random_pos_mean * 0.02
            # random_ori_mean = ee_states[:, 3:6].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            # random_ori = torch.randn_like(ee_states[:, 3:6]) * random_ori_mean * 0.02
            ee_mean = ee_states[:, :6].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            ee_states[:, :6] = ee_states[:, :6] + torch.randn_like(ee_states[:, :6]) * 0.01#ee_mean * 0.1
            jo_mean = jo_states[:, :6].mean(0, keepdims=True).repeat(jo_states.shape[0], 1)
            jo_states[:, :6] = jo_states[:, :6] + torch.randn_like(jo_states[:, :6]) * 0.01#jo_mean * 0.1

        # mat to axisangle
        base_axisangle = [mat2quat(axa) for axa in base_axisangle]
        base_axisangle = [quat2axisangle(axa) for axa in base_axisangle]
        base_axisangle = torch.from_numpy(np.array(base_axisangle))

        states = {
            'ee_states': ee_states,
            # 'joint_states': jo_states,
            'base_axisangle': base_axisangle,
        }
        return frames, actions_delta, states


    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(demo_path)
        demo_length = self._demo_id_to_demo_length[demo_id]
        task_name = os.path.basename(demo_parent_dir).replace('_', ' ')
        demo_name = os.path.basename(demo_path).split(".")[0]
        task_embs = task_name
        time_offset = index - demo_start_index

        # all_view_frames = []
        # all_view_track_transformer_frames = []
        obs, actions, extra_states = self._load_image_list_from_path(demo_id, time_offset, backward=True)

        # track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        track = torch.rand(2, obs.shape[1], 16, 10, 2)  # v t track_len n 2
        # vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        # track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w
        track_transformer_obs = torch.rand(3, 128, 128)

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        # sample_track, sample_vi = [], []
        # for i in range(len(self.views)):
        #     sample_track_per_time, sample_vi_per_time = [], []
        #     for t in range(self.frame_stack):
        #         track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
        #         sample_track_per_time.append(track_i_t)
        #         sample_vi_per_time.append(vi_i_t)
        #     sample_track.append(torch.stack(sample_track_per_time, dim=0))
        #     sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        # track = torch.stack(sample_track, dim=0)
        # vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        # task_embs = demo["task_text"]

        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        # actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        # extra_states = {}
        # demo_length = demo["root"][view]["video"].shape[0]
        # for k, v in demo['root']['extra_states'].items():
        #     image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
        #     image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
        #     values = v[image_indices]
        #     if len(values) < self.frame_stack:
        #         # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
        #         padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
        #         values = torch.cat([padding_values, values], dim=0)
        #     extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]

        return obs, track_transformer_obs, track, task_embs, actions, extra_states


class BCDatasetActionMask(BaseDataset):
    def __init__(self, track_obs_fs=1, pred_frame=4, his_frame=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs
        self.his_frame = his_frame
        self.pred_frame = pred_frame
        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug2(input_shape=(128, 128), translation=8, augment_track=self.augment_track),
        ])

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            frames = demo['root'][view]["video"][image_indices]
            if len(frames) < num_frames:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_mask_from_demo(self, demo, view, time_offset, mask_dir, num_frames=None, backward=True):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
            his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][his_indices]

            if view == 'eye_in_hand':
                masks = np.ones((frames.shape[0], frames.shape[2], frames.shape[3])) * 255
            else:
                masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                         his_indices]
                masks = np.stack(masks)  # t h w c
            masks = torch.Tensor(masks).unsqueeze(-1)
            masks = rearrange(masks, "t h w c -> t c h w")

            return frames, masks
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_list_from_disk(self, demo_id, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            image_indices = np.arange(time_offset + 1 - num_frames, time_offset + 1)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
            # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # t h w c
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def _load_image_mask_from_disk(self, demo_id, view, time_offset, mask_dir, num_frames=None, backward=True):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
            his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in
                          his_indices]
            if view == 'eye_in_hand':
                masks = [np.ones((frame.shape[0], frame.shape[1])) * 255 for frame in frames]
            else:
                masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                         his_indices]
            # image_indices = np.arange(time_offset + 1 - num_frames, time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=0, a_max=demo_length - 1)
            # frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
            # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # t h w c
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        masks = np.stack(masks)  # t h w c
        masks = torch.Tensor(masks).unsqueeze(-1)
        masks = rearrange(masks, "t h w c -> t c h w")
        return frames, masks

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        masks_dir = os.path.join(demo_parent_dir, "masks1", demo_name)

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            all_views_masks = []
            for view in self.views:
                if self.cache_image:
                    frames, masks = self._load_image_mask_from_demo(demo, view, time_offset, masks_dir, backward=True)
                    all_view_frames.append(frames)
                    all_views_masks.append(masks)
                    # all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset, backward=True))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    frames, masks = self._load_image_mask_from_disk(demo_id, view, time_offset, masks_dir, backward=True)  # t c h w
                    all_view_frames.append(frames)
                    all_views_masks.append(masks)
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            all_view_frames = []
            all_views_masks = []
            all_view_track_transformer_frames = []
            for view in self.views:
                frames, masks = self._load_image_mask_from_demo(demo, view, time_offset, masks_dir, backward=True)
                all_view_frames.append(frames)
                all_views_masks.append(masks)
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        all_masks = torch.stack(all_views_masks, dim=0)  # v t 1 h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        task_embs = demo["root"]["task_emb_bert"]
        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        extra_states = {}
        demo_length = demo["root"][view]["video"].shape[0]
        for k, v in demo['root']['extra_states'].items():
            image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            values = v[image_indices]
            if len(values) < self.frame_stack:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
                values = torch.cat([padding_values, values], dim=0)
            extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]
        all_masks = all_masks / 255.
        assert all_masks.max() == 1.
        return obs, all_masks, track, task_embs, actions, extra_states


class BCDatasetAction(BaseDataset):
    def __init__(self, track_obs_fs=1, pred_frame=4, his_frame=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs
        self.his_frame = his_frame
        self.pred_frame = pred_frame
        # self.augmentor = transforms.Compose([
        #     ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        #     ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        # ])

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=True):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
            his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][his_indices]
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_mask_from_demo(self, demo, view, time_offset, mask_dir, num_frames=None, backward=True):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
            his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][his_indices]

            if view == 'eye_in_hand':
                masks = np.ones((frames.shape[0], frames.shape[2], frames.shape[3])) * 255
            else:
                masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                         his_indices]
                masks = np.stack(masks)  # t h w c
            masks = torch.Tensor(masks).unsqueeze(-1)
            masks = rearrange(masks, "t h w c -> t c h w")

            return frames, masks
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_list_from_disk(self, demo_id, view, time_offset, num_frames=None, backward=True):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
            his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in
                      his_indices]
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # t h w c
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def _load_image_mask_from_disk(self, demo_id, view, time_offset, mask_dir, num_frames=None, backward=True):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
            his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in
                          his_indices]
            if view == 'eye_in_hand':
                masks = [np.ones((frame.shape[0], frame.shape[1])) * 255 for frame in frames]
            else:
                masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                         his_indices]
            # image_indices = np.arange(time_offset + 1 - num_frames, time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=0, a_max=demo_length - 1)
            # frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
            # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        else:
            image_indices = np.arange(time_offset, time_offset + num_frames)
            image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        frames = np.stack(frames)  # t h w c
        frames = torch.Tensor(frames)
        frames = rearrange(frames, "t h w c -> t c h w")
        masks = np.stack(masks)  # t h w c
        masks = torch.Tensor(masks).unsqueeze(-1)
        masks = rearrange(masks, "t h w c -> t c h w")
        return frames, masks

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        task_embs = demo_parent_dir[:-5].split('/')[-1].replace('_', ' ')
        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    frames = self._load_image_list_from_demo(demo, view, time_offset, backward=True)
                    all_view_frames.append(frames)
                    # all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset, backward=True))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    frames = self._load_image_list_from_disk(demo_id, view, time_offset, backward=True)  # t c h w
                    all_view_frames.append(frames)
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                frames = self._load_image_list_from_demo(demo, view, time_offset, backward=True)
                all_view_frames.append(frames)
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            for track_start_index in range(time_offset, time_offset+self.frame_stack):
                all_time_step_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        # task_embs = demo["root"]["task_emb_bert"]
        # task_embs = demo["task_text"]
        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        extra_states = {}
        demo_length = demo["root"][view]["video"].shape[0]
        for k, v in demo['root']['extra_states'].items():
            image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            values = v[image_indices]
            if len(values) < self.frame_stack:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
                values = torch.cat([padding_values, values], dim=0)
            if k == 'ee_states':
                values_ori = values[:, 3:6]
                values_quat = axisangle2quat_torch(values_ori)
                negetive_mask = values_quat[:, -1] < 0
                values_quat[negetive_mask] = torch.negative(values_quat[negetive_mask])
                values_ori = quat2axisangle_torch(values_quat)
                values[:, 3:6] = values_ori
            extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]
        if time_offset == 0:
            use_action = torch.tensor(0.)
        else:
            use_action = torch.tensor(1.)
        return obs, track, task_embs, actions, extra_states, use_action



class BCDatasetActionReal(Dataset):
    def __init__(self, dataset_dir,
                 img_size, track_obs_fs=1, pred_frame=4, his_frame=4,
                 num_demos=None, augment_track=True, aug_prob=0., *args, **kwargs):
        super().__init__()
        self.dataset_dir = dataset_dir
        self._index_to_view_id = {}
        self.track_obs_fs = track_obs_fs
        self.num_demos = num_demos
        self.augment_track = augment_track
        self.aug_prob = aug_prob
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = (img_size[0], img_size[1])
        self.buffer_fns = []
        task_names = os.listdir(self.dataset_dir)
        self.dataset_dir = [os.path.join(self.dataset_dir, task_name) for dir_idx, task_name in enumerate(task_names)]
        for dir_idx, d in enumerate(self.dataset_dir):
            fn_list = os.listdir(d)
            fn_list = [os.path.join(d, fn) for fn in fn_list]
            # fn_list = glob(os.path.join(d, "*.hdf5"))
            fn_list = natsorted(fn_list)
            if self.num_demos is None:
                n_demo = len(fn_list)
            else:
                assert 0 < self.num_demos <= 1, "num_demos means the ratio of training data among all the demos."
                n_demo = int(len(fn_list) * self.num_demos)
            for fn in fn_list[:n_demo]:
                self.buffer_fns.append(fn)

        assert (len(self.buffer_fns) > 0)
        print(f"found {len(self.buffer_fns)} trajectories in the specified folders: {self.dataset_dir}")

        self._cache = []
        self._index_to_demo_id, self._demo_id_to_path, self._demo_id_to_start_indices, self._demo_id_to_demo_length, \
            self._demo_id_to_ee_pos_path, self._demo_id_to_agent_view_path, self._demo_id_to_hand_view_path \
            = {}, {}, {}, {}, {}, {}, {}
        self.load_demo_info()

        self.his_frame = his_frame
        self.pred_frame = pred_frame
        self.load_image_func = load_rgb
        self.ori_scale = np.array([0.5, 0.5, 0.5])
        self.pos_scale = np.array([0.05, 0.05, 0.05])
        self.views = ['agent_view', 'hand_view']

        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        ])

    def __len__(self):
        return len(self._index_to_demo_id)

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            # demo = self.load_h5(fn)

            # if self.views is None:
            #     self.views = ['agent_view', 'hand_view']
                # self.views = list(demo["root"].keys())
                # self.views.remove("actions")
                # self.views.remove("task_emb_bert")
                # self.views.remove("extra_states")
                # self.views.sort()
            filenames = os.listdir(fn)
            ee_pos_names = [filename for filename in filenames if filename.endswith(".json")]
            ee_pos_names = sorted(ee_pos_names)
            agent_view_img_names = ['agent_view' + filename[-25:-5]+ '.png' for filename in ee_pos_names]
            hand_view_img_names = ['hand_view' + filename[-25:-5] + '.png' for filename in ee_pos_names]
            # ee_pos_path = glob(os.path.join(fn, '*.json'))
            # ee_pos_path = sorted(ee_pos_path)
            ee_pos_path = [os.path.join(fn, ee_pos_name) for ee_pos_name in ee_pos_names]
            agent_view_img_path = [os.path.join(fn, agent_view_img_name) for agent_view_img_name in agent_view_img_names]
            hand_view_img_path = [os.path.join(fn, hand_view_img_name) for hand_view_img_name in hand_view_img_names]
            # demo_len = demo["root"][self.views[0]]["video"][0].shape[0]
            demo_len = len(ee_pos_path)
            assert demo_len == len(ee_pos_path) == len(agent_view_img_path) == len(hand_view_img_path)

            demo = {
                "root": {},
                "extra_states": {},
                "task_emb_bert": {},
                "actions": {},
            }

            # if self.cache_all:
            #     demo = self.process_demo(demo)
            #     for v in self.views:
            #         del demo["root"][v]["video"]
            #     self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            self._demo_id_to_ee_pos_path[demo_idx] = ee_pos_path
            self._demo_id_to_agent_view_path[demo_idx] = agent_view_img_path
            self._demo_id_to_hand_view_path[demo_idx] = hand_view_img_path
            start_idx += demo_len

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx


    def _load_image_list_from_path(self, demo_id, time_offset, num_frames=None, backward=True):
        demo_path = self._demo_id_to_path[demo_id]
        ee_pos_path = self._demo_id_to_ee_pos_path[demo_id]
        agent_view_img_path = self._demo_id_to_agent_view_path[demo_id]
        hand_view_img_path = self._demo_id_to_hand_view_path[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_name = os.path.basename(demo_path).split(".")[0]
        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames_agent = [self.load_image_func(agent_view_img_path[img_idx]) for img_idx in
                           his_indices]
        his_frames_agent = np.stack(his_frames_agent)  # t h w c
        his_frames_agent = torch.Tensor(his_frames_agent)
        his_frames_agent = rearrange(his_frames_agent, "t h w c -> t c h w")
        his_frames_agent = torch.nn.functional.interpolate(his_frames_agent, size=self.img_size, mode='bilinear')

        his_frames_hand = [self.load_image_func(hand_view_img_path[img_idx]) for img_idx in
                      his_indices]
        his_frames_hand = np.stack(his_frames_hand)  # t h w c
        his_frames_hand = torch.Tensor(his_frames_hand)
        his_frames_hand = rearrange(his_frames_hand, "t h w c -> t c h w")
        his_frames_hand = torch.nn.functional.interpolate(his_frames_hand, size=self.img_size, mode='bilinear')
        frames = torch.stack([his_frames_agent, his_frames_hand], dim=0)  # v t c h w

        action_indices = np.arange(time_offset, time_offset + self.pred_frame + 1)
        action_indices = np.clip(action_indices, a_min=0, a_max=demo_length - 1)
        actions = [np.array(json.loads(open(ee_pos_path[img_idx], 'r').read())) for img_idx in action_indices]
        actions_gri = [action[-1] for action in actions]
        actions_gri = np.array(actions_gri[1:]).reshape(-1, 1)
        actions_pos = [action[:3] for action in actions]
        actions_ori = [action[3:6] for action in actions]
        # actions_ori = [axisangle2quat(action) for action in actions_ori]
        # actions_ori = [quat2mat(action) for action in actions_ori]
        actions_ori = [euler2mat(action) for action in actions_ori]
        base_axisangle = actions_ori[:-1].copy()
        actions_ori = np.array(actions_ori)
        actions_delta_ori = []
        for i in range(len(actions_ori) - 1):
            ad = np.dot(actions_ori[i + 1], matrix_inverse(actions_ori[i]))
            ad = mat2quat(ad)
            ad = quat2axisangle(ad)
            ad = ad / self.ori_scale
            actions_delta_ori.append(ad)
        actions_delta_pos = []
        for i in range(len(actions_pos) - 1):
            ad = actions_pos[i + 1] - actions_pos[i]
            ad = ad / self.pos_scale
            actions_delta_pos.append(ad)
        actions_delta = np.concatenate([np.array(actions_delta_pos), np.array(actions_delta_ori), actions_gri],
                                       axis=-1).astype(np.float32)

        action_indices = np.arange(time_offset-self.his_frame + 1, time_offset + 1)
        action_indices = np.clip(action_indices, a_min=0, a_max=demo_length - 1)
        actions = [np.array(json.loads(open(ee_pos_path[img_idx], 'r').read())) for img_idx in action_indices]
        actions_pos = [action[:3] for action in actions]
        actions_ori = [action[3:6] for action in actions]
        actions_ori = [euler2mat(action) for action in actions_ori]
        actions_ori = [mat2quat(action) for action in actions_ori]
        actions_ori = [quat2axisangle(action) for action in actions_ori]
        actions_gri = [action[-1] for action in actions]
        ee_states = np.concatenate([np.array(actions_pos), np.array(actions_ori), np.array(actions_gri)[..., None]],
                                   axis=-1).astype(np.float32)
        # ee_states = np.concatenate([np.array(actions_pos), np.array(actions_ori)], axis=-1).astype(np.float32)
        ee_states = torch.from_numpy(ee_states)
        if random.random() >= 0.5:
            # random_pos_mean = ee_states[:, :3].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            # random_pos = torch.randn_like(ee_states[:, :3]) * random_pos_mean * 0.02
            # random_ori_mean = ee_states[:, 3:6].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            # random_ori = torch.randn_like(ee_states[:, 3:6]) * random_ori_mean * 0.02
            ee_mean = ee_states[:, :6].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            ee_states[:, :6] = ee_states[:, :6] + torch.randn_like(ee_states[:, :6]) * 0.01  # ee_mean * 0.1
        base_axisangle = [mat2quat(axa) for axa in base_axisangle]
        base_axisangle = [quat2axisangle(axa) for axa in base_axisangle]
        base_axisangle = torch.from_numpy(np.array(base_axisangle))
        states = {
            'ee_states': ee_states,
            'base_axisangle': base_axisangle,
        }
        return frames, actions_delta, states


    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(demo_path)
        demo_length = self._demo_id_to_demo_length[demo_id]
        task_name = os.path.basename(demo_parent_dir).replace('_', ' ')
        demo_name = os.path.basename(demo_path).split(".")[0]
        task_embs = task_name
        time_offset = index - demo_start_index

        # all_view_frames = []
        # all_view_track_transformer_frames = []
        obs, actions, extra_states = self._load_image_list_from_path(demo_id, time_offset, backward=True)

        # obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        # track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        track = torch.rand(2, obs.shape[1], 16, 10, 2)
        # vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        # track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        # sample_track, sample_vi = [], []
        # for i in range(len(self.views)):
        #     sample_track_per_time, sample_vi_per_time = [], []
        #     for t in range(self.frame_stack):
        #         track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
        #         sample_track_per_time.append(track_i_t)
        #         sample_vi_per_time.append(vi_i_t)
        #     sample_track.append(torch.stack(sample_track_per_time, dim=0))
        #     sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        # track = torch.stack(sample_track, dim=0)
        # vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        # task_embs = demo["root"]["task_emb_bert"]
        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        # actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        # extra_states = {}
        # demo_length = demo["root"][view]["video"].shape[0]
        # for k, v in demo['root']['extra_states'].items():
        #     image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
        #     image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
        #     values = v[image_indices]
        #     if len(values) < self.frame_stack:
        #         # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
        #         padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
        #         values = torch.cat([padding_values, values], dim=0)
        #     extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]
        if time_offset == 0:
            use_action = torch.tensor(0.)
        else:
            use_action = torch.tensor(1.)
        return obs, track, task_embs, actions, extra_states, use_action

class BCDatasetActionReal2(Dataset):
    def __init__(self, dataset_dir,
                 img_size, track_obs_fs=1, pred_frame=4, his_frame=4,
                 num_demos=None, augment_track=True, aug_prob=0., *args, **kwargs):
        super().__init__()
        self.dataset_dir = dataset_dir
        self._index_to_view_id = {}
        self.track_obs_fs = track_obs_fs
        self.num_demos = num_demos
        self.augment_track = augment_track
        self.aug_prob = aug_prob
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = (img_size[0], img_size[1])
        self.buffer_fns = []
        task_names = os.listdir(self.dataset_dir)
        self.dataset_dir = [os.path.join(self.dataset_dir, task_name) for dir_idx, task_name in enumerate(task_names)]
        for dir_idx, d in enumerate(self.dataset_dir):
            fn_list = os.listdir(d)
            fn_list = [os.path.join(d, fn) for fn in fn_list]
            # fn_list = glob(os.path.join(d, "*.hdf5"))
            fn_list = natsorted(fn_list)
            if self.num_demos is None:
                n_demo = len(fn_list)
            else:
                assert 0 < self.num_demos <= 1, "num_demos means the ratio of training data among all the demos."
                n_demo = int(len(fn_list) * self.num_demos)
            for fn in fn_list[:n_demo]:
                self.buffer_fns.append(fn)

        assert (len(self.buffer_fns) > 0)
        print(f"found {len(self.buffer_fns)} trajectories in the specified folders: {self.dataset_dir}")

        self._cache = []
        self._index_to_demo_id, self._demo_id_to_path, self._demo_id_to_start_indices, self._demo_id_to_demo_length, \
            self._demo_id_to_ee_pos_path, self._demo_id_to_agent_view_path, self._demo_id_to_hand_view_path \
            = {}, {}, {}, {}, {}, {}, {}
        self.load_demo_info()

        self.his_frame = his_frame
        self.pred_frame = pred_frame
        self.load_image_func = load_rgb
        self.ori_scale = np.array([0.5, 0.5, 0.5])
        self.pos_scale = np.array([0.05, 0.05, 0.05])
        self.views = ['agent_view', 'hand_view']

        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        ])

    def __len__(self):
        return len(self._index_to_demo_id)

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            # demo = self.load_h5(fn)

            # if self.views is None:
            #     self.views = ['agent_view', 'hand_view']
                # self.views = list(demo["root"].keys())
                # self.views.remove("actions")
                # self.views.remove("task_emb_bert")
                # self.views.remove("extra_states")
                # self.views.sort()
            filenames = os.listdir(fn)
            ee_pos_names = [filename for filename in filenames if filename.endswith(".json")]
            ee_pos_names = sorted(ee_pos_names)
            agent_view_img_names = ['agent_view' + filename[-25:-5]+ '.png' for filename in ee_pos_names]
            hand_view_img_names = ['hand_view' + filename[-25:-5] + '.png' for filename in ee_pos_names]
            # ee_pos_path = glob(os.path.join(fn, '*.json'))
            # ee_pos_path = sorted(ee_pos_path)
            ee_pos_path = [os.path.join(fn, ee_pos_name) for ee_pos_name in ee_pos_names]
            agent_view_img_path = [os.path.join(fn, agent_view_img_name) for agent_view_img_name in agent_view_img_names]
            hand_view_img_path = [os.path.join(fn, hand_view_img_name) for hand_view_img_name in hand_view_img_names]
            # demo_len = demo["root"][self.views[0]]["video"][0].shape[0]
            demo_len = len(ee_pos_path)
            assert demo_len == len(ee_pos_path) == len(agent_view_img_path) == len(hand_view_img_path)

            demo = {
                "root": {},
                "extra_states": {},
                "task_emb_bert": {},
                "actions": {},
            }

            # if self.cache_all:
            #     demo = self.process_demo(demo)
            #     for v in self.views:
            #         del demo["root"][v]["video"]
            #     self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            self._demo_id_to_ee_pos_path[demo_idx] = ee_pos_path
            self._demo_id_to_agent_view_path[demo_idx] = agent_view_img_path
            self._demo_id_to_hand_view_path[demo_idx] = hand_view_img_path
            start_idx += demo_len

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx


    def _load_image_list_from_path(self, demo_id, time_offset, num_frames=None, backward=True):
        demo_path = self._demo_id_to_path[demo_id]
        ee_pos_path = self._demo_id_to_ee_pos_path[demo_id]
        agent_view_img_path = self._demo_id_to_agent_view_path[demo_id]
        hand_view_img_path = self._demo_id_to_hand_view_path[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_name = os.path.basename(demo_path).split(".")[0]
        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames_agent = [self.load_image_func(agent_view_img_path[img_idx]) for img_idx in
                           his_indices]
        his_frames_agent = np.stack(his_frames_agent)  # t h w c
        his_frames_agent = torch.Tensor(his_frames_agent)
        his_frames_agent = rearrange(his_frames_agent, "t h w c -> t c h w")
        his_frames_agent = torch.nn.functional.interpolate(his_frames_agent, size=self.img_size, mode='bilinear')

        his_frames_hand = [self.load_image_func(hand_view_img_path[img_idx]) for img_idx in
                      his_indices]
        his_frames_hand = np.stack(his_frames_hand)  # t h w c
        his_frames_hand = torch.Tensor(his_frames_hand)
        his_frames_hand = rearrange(his_frames_hand, "t h w c -> t c h w")
        his_frames_hand = torch.nn.functional.interpolate(his_frames_hand, size=self.img_size, mode='bilinear')
        frames = torch.stack([his_frames_agent, his_frames_hand], dim=0)  # v t c h w

        action_indices = np.arange(time_offset, time_offset + self.pred_frame + 1)
        action_indices = np.clip(action_indices, a_min=0, a_max=demo_length - 1)
        labels = [json.loads(open(ee_pos_path[img_idx], 'r').read()) for img_idx in action_indices]
        actions = np.array([label['ee_pose'] for label in labels])
        actions_gri = [action[-1] for action in actions]
        actions_gri = np.array(actions_gri[1:]).reshape(-1, 1)
        actions_pos = [action[:3] for action in actions]
        actions_ori = [action[3:6] for action in actions]
        # actions_ori = [axisangle2quat(action) for action in actions_ori]
        # actions_ori = [quat2mat(action) for action in actions_ori]
        actions_ori = [euler2mat(action) for action in actions_ori]
        base_axisangle = actions_ori[:-1].copy()
        actions_ori = np.array(actions_ori)
        actions_delta_ori = []
        for i in range(len(actions_ori) - 1):
            ad = np.dot(actions_ori[i + 1], matrix_inverse(actions_ori[i]))
            ad = mat2quat(ad)
            ad = quat2axisangle(ad)
            ad = ad / self.ori_scale
            actions_delta_ori.append(ad)
        actions_delta_pos = []
        for i in range(len(actions_pos) - 1):
            ad = actions_pos[i + 1] - actions_pos[i]
            ad = ad / self.pos_scale
            actions_delta_pos.append(ad)
        actions_delta = np.concatenate([np.array(actions_delta_pos), np.array(actions_delta_ori), actions_gri],
                                       axis=-1).astype(np.float32)

        action_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        action_indices = np.clip(action_indices, a_min=0, a_max=demo_length - 1)
        labels = [json.loads(open(ee_pos_path[img_idx], 'r').read()) for img_idx in action_indices]
        ee_states = np.array([label['ee_pose'] for label in labels])
        ee_pos = [ee[:3] for ee in ee_states]
        ee_ori = [ee[3:6] for ee in ee_states]
        ee_ori = [euler2mat(ee) for ee in ee_ori]
        ee_ori = [mat2quat(ee) for ee in ee_ori]
        ee_ori = [quat2axisangle(ee) for ee in ee_ori]
        ee_gri = [ee[-1] for ee in ee_states]
        ee_states = np.concatenate([np.array(ee_pos), np.array(ee_ori), np.array(ee_gri)[..., None]], axis=-1).astype(
            np.float32)
        ee_states = torch.from_numpy(ee_states)

        jo_states = np.array([label['joint_states'] for label in labels])
        jo_states = torch.from_numpy(jo_states)
        if random.random() >= 0.5:
            # random_pos_mean = ee_states[:, :3].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            # random_pos = torch.randn_like(ee_states[:, :3]) * random_pos_mean * 0.02
            # random_ori_mean = ee_states[:, 3:6].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            # random_ori = torch.randn_like(ee_states[:, 3:6]) * random_ori_mean * 0.02
            ee_mean = ee_states[:, :6].mean(0, keepdims=True).repeat(ee_states.shape[0], 1)
            ee_states[:, :6] = ee_states[:, :6] + torch.randn_like(ee_states[:, :6]) * 0.01  # ee_mean * 0.1
            jo_mean = jo_states[:, :6].mean(0, keepdims=True).repeat(jo_states.shape[0], 1)
            jo_states[:, :6] = jo_states[:, :6] + torch.randn_like(jo_states[:, :6]) * 0.01  # jo_mean * 0.1

        # mat to axisangle
        base_axisangle = [mat2quat(axa) for axa in base_axisangle]
        base_axisangle = [quat2axisangle(axa) for axa in base_axisangle]
        base_axisangle = torch.from_numpy(np.array(base_axisangle))

        states = {
            'ee_states': ee_states,
            # 'joint_states': jo_states,
            'base_axisangle': base_axisangle,
        }
        return frames, actions_delta, states

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=True):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
            his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][his_indices]
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_mask_from_demo(self, demo, view, time_offset, mask_dir, num_frames=None, backward=True):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
            his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][his_indices]

            if view == 'eye_in_hand':
                masks = np.ones((frames.shape[0], frames.shape[2], frames.shape[3])) * 255
            else:
                masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                         his_indices]
                masks = np.stack(masks)  # t h w c
            masks = torch.Tensor(masks).unsqueeze(-1)
            masks = rearrange(masks, "t h w c -> t c h w")

            return frames, masks
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]


    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(demo_path)
        demo_length = self._demo_id_to_demo_length[demo_id]
        task_name = os.path.basename(demo_parent_dir).replace('_', ' ')
        demo_name = os.path.basename(demo_path).split(".")[0]
        task_embs = task_name
        time_offset = index - demo_start_index

        # all_view_frames = []
        # all_view_track_transformer_frames = []
        obs, actions, extra_states = self._load_image_list_from_path(demo_id, time_offset, backward=True)

        # obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        # track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        track = torch.rand(2, obs.shape[1], 16, 10, 2)
        # vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        # track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        # sample_track, sample_vi = [], []
        # for i in range(len(self.views)):
        #     sample_track_per_time, sample_vi_per_time = [], []
        #     for t in range(self.frame_stack):
        #         track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
        #         sample_track_per_time.append(track_i_t)
        #         sample_vi_per_time.append(vi_i_t)
        #     sample_track.append(torch.stack(sample_track_per_time, dim=0))
        #     sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        # track = torch.stack(sample_track, dim=0)
        # vi = torch.stack(sample_vi, dim=0)

        # actions = demo["root"]["actions"][time_offset:time_offset + self.frame_stack]
        # task_embs = demo["root"]["task_emb_bert"]
        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        # actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        # extra_states = {}
        # demo_length = demo["root"][view]["video"].shape[0]
        # for k, v in demo['root']['extra_states'].items():
        #     image_indices = np.arange(max(time_offset + 1 - self.frame_stack, 0), time_offset + 1)
        #     image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
        #     values = v[image_indices]
        #     if len(values) < self.frame_stack:
        #         # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
        #         padding_values = values[:1].repeat(self.frame_stack - len(values), 1)  # padding with first images
        #         values = torch.cat([padding_values, values], dim=0)
        #     extra_states[k] = values
            # demo['root'][view]["video"][time_offset:time_offset + self.frame_stack]
        if time_offset == 0:
            use_action = torch.tensor(0.)
        else:
            use_action = torch.tensor(1.)
        return obs, track, task_embs, actions, extra_states, use_action


if __name__ =='__main__':
    dataset_dir = ['/home/huang/code/ATM/data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/']
    dataset_dir = ['/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo/bc_train_10',
                   '/home/huang/code/ATM/data/atm_libero//libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo/bc_train_10']
    dataset_dir = [
        '/home/huang/code/ATM/data/atm_libero/libero_complex/KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl_demo/bc_train_10'
    ]

    dataset_dir = ['./data/atm_libero//libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo/bc_train_45',
 # './data/atm_libero//libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo/bc_train_45',
 # './data/atm_libero//libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo/bc_train_45',
 # './data/atm_libero//libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo/bc_train_45',
 # './data/atm_libero//libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo/bc_train_45',
 # './data/atm_libero//libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo/bc_train_45',
 # './data/atm_libero//libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo/bc_train_45',
 # './data/atm_libero//libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo/bc_train_45',
 # './data/atm_libero//libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo/bc_train_45',
 # './data/atm_libero//libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo/bc_train_45'
                   ]
    dataset_cfg = {
        'img_size': 128,
    'frame_stack': 5,
    'num_track_ts': 16,
    'num_track_ids': 32,
    'track_obs_fs': 1,
    'augment_track': False,
    'extra_state_keys': ["joint_states", "gripper_states"],
    'cache_all': False,
    'cache_image': False,
    }
    # dataset = BCDatasetActionMask(dataset_dir=dataset_dir, **dataset_cfg, aug_prob=0.9)

    # dataset = BCDatasetIDM(dataset_dir=dataset_dir, **dataset_cfg, aug_prob=0.9)
    # d = dataset[9]

    dataset = BCDatasetAction(dataset_dir=dataset_dir, **dataset_cfg, aug_prob=0.9)
    actions = []
    for i in range(len(dataset)):
        act = dataset[i][3][0].numpy()
        actions.append(act)
    actions = np.stack(actions, axis=0)
    d = dataset[1]
    pos_max = 0.03586133
    pos_min = -0.029739289
    dataset = BCDatasetActionReal(dataset_dir='/media/huang/T7/real_exp/worldmodel_data_plus', **dataset_cfg, aug_prob=0.5)
    dataset = BCDataset3Real2(dataset_dir='/media/huang/T7/real_exp/worldmodel_data_plus', **dataset_cfg,
                                   aug_prob=0.5)
    d = dataset[200]
    dataset = BCDataset3Real(dataset_dir='/media/huang/T7/real_exp/worldmodel_data2', **dataset_cfg, aug_prob=0.9)
    d = dataset[0]
    for i in range(0, len(dataset)):
        d = dataset[i]
        action_pos = d[-2][:, :3]
        pos_max = max(pos_max, action_pos.max())
        pos_min = min(pos_min, action_pos.min())
    print(pos_min, pos_max)

    dataset = BCDataset3Real(dataset_dir='/media/huang/T7/real_exp/worldmodel_data2', **dataset_cfg, aug_prob=0.9)
    d = dataset[0]

    dataset = BCDataset3Real2(dataset_dir='/media/huang/T7/real_exp/worldmodel_data_plus', **dataset_cfg, aug_prob=0.9)
    d = dataset[0]

    dataset = BCDataset3(dataset_dir=dataset_dir, **dataset_cfg, aug_prob=0.9)
    pos_max = -1
    pos_min = 1
    ori_max = -10
    ori_min = 10
    for i in range(1171):
        d = dataset[i]
        action = d[-2]
        pos = action[:, :3]
        if torch.max(pos) > pos_max:
            pos_max = torch.max(pos)
        if torch.min(pos) < pos_min:
            pos_min = pos.min()
        ori = action[:, 3:6]
        if torch.max(ori) > ori_max:
            ori_max = torch.max(ori)
        if torch.min(ori) < ori_min:
            ori_min = torch.min(ori)
    d = dataset[98]
    d = dataset[99]
    pass