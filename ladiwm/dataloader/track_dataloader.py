import numpy as np
import torch
from torchvision import transforms
from ladiwm.dataloader.base_dataset import BaseDataset
from ladiwm.utils.flow_utils import sample_tracks_visible_first
import os
from glob import glob
from natsort import natsorted
from einops import rearrange
from ladiwm.dataloader.utils import load_rgb, ImgTrackColorJitter, ImgViewDiffTranslationAug, ImgTrackColorJitter2, ImgViewDiffTranslationAug2
from torch.utils.data import Dataset
import json
from ladiwm.utils.transform_utils import quat2axisangle, axisangle2quat, quat2mat, mat2quat, matrix_inverse, mat2euler, euler2mat


class ATMPretrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self._index_to_view_id = {}
        super().__init__(*args, **kwargs)

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)

            if self.views is None:
                self.views = list(demo["root"].keys())
                self.views.remove("actions")
                self.views.remove("task_emb_bert")
                self.views.remove("extra_states")
                self.views.sort()

            demo_len = demo["root"][self.views[0]]["video"][0].shape[0]

            if self.cache_all:
                demo = self.process_demo(demo)
                for v in self.views:
                    del demo["root"][v]["video"]
                self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        view = self.views[self._index_to_view_id[index]]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w
            else:
                vids = self._load_image_list_from_disk(demo_id, view, time_offset, backward=True)  # t c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w

        tracks = demo["root"][view]["tracks"][time_offset:time_offset + self.num_track_ts]  # track_len n 2
        vis = demo["root"][view]['vis'][time_offset:time_offset + self.num_track_ts]  # track_len n
        task_emb = demo["root"]["task_emb_bert"]  # (dim,)

        # augment videos
        if np.random.rand() < self.aug_prob:
            vids = vids[None]  # expand to (1, t, c, h, w) to fit the input shape of random shift augmentation
            tracks = tracks[None, None]  # expand to (1, 1, track_len, n, 2) to fit the input shape of random shift augmentation
            vids, tracks = self.augmentor((vids / 255., tracks))
            vids = vids[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        # sample tracks
        tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)

        return vids, tracks, vis, task_emb


class ATMPretrainDataset2(BaseDataset):
    def __init__(self, *args, **kwargs):
        self._index_to_view_id = {}
        super().__init__(*args, **kwargs)

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)

            if self.views is None:
                self.views = list(demo["root"].keys())
                self.views.remove("actions")
                self.views.remove("task_emb_bert")
                self.views.remove("extra_states")
                self.views.sort()

            demo_len = demo["root"][self.views[0]]["video"][0].shape[0]

            if self.cache_all:
                demo = self.process_demo(demo)
                for v in self.views:
                    del demo["root"][v]["video"]
                self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            time_offset = max(time_offset, self.frame_stack)
            image_indices = np.arange(time_offset-self.frame_stack, time_offset)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
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
            time_offset = max(time_offset, self.frame_stack)
            image_indices = np.arange(time_offset - self.frame_stack, time_offset)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
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
        view = self.views[self._index_to_view_id[index]]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w
            else:
                vids = self._load_image_list_from_disk(demo_id, view, time_offset, backward=True)  # t c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w

        tracks = demo["root"][view]["tracks"][time_offset:time_offset + self.num_track_ts]  # track_len n 2
        vis = demo["root"][view]['vis'][time_offset:time_offset + self.num_track_ts]  # track_len n
        task_emb = demo["root"]["task_emb_bert"]  # (dim,)

        # augment videos
        if np.random.rand() < self.aug_prob:
            vids = vids[None]  # expand to (1, t, c, h, w) to fit the input shape of random shift augmentation
            tracks = tracks[None, None]  # expand to (1, 1, track_len, n, 2) to fit the input shape of random shift augmentation
            vids, tracks = self.augmentor((vids / 255., tracks))
            vids = vids[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        # sample tracks
        tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)

        return vids, tracks, vis, task_emb


class ATMPretrainDataset3(BaseDataset):
    def __init__(self, *args, **kwargs):
        self._index_to_view_id = {}
        super().__init__(*args, **kwargs)
        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            # ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        ])

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)

            if self.views is None:
                self.views = list(demo["root"].keys())
                self.views.remove("actions")
                self.views.remove("task_emb_bert")
                self.views.remove("extra_states")
                self.views.sort()

            demo_len = demo["root"][self.views[0]]["video"][0].shape[0]

            if self.cache_all:
                demo = self.process_demo(demo)
                for v in self.views:
                    del demo["root"][v]["video"]
                self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            time_offset = max(time_offset, self.frame_stack)
            image_indices = np.arange(time_offset-self.frame_stack, time_offset)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][image_indices]
            if len(frames) < num_frames:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_mask_from_demo(self, demo, view, time_offset, mask_dir, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]

        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            time_offset = max(time_offset, self.frame_stack-1)
            image_indices = np.arange(time_offset-self.frame_stack+1, time_offset+1)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][image_indices]
            if view == 'eye_in_hand':
                masks = np.zeros_like(frames)
            else:
                masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                      image_indices]
                masks = np.stack(masks)  # t h w c
                masks = torch.Tensor(masks).unsqueeze(-1)
                masks = rearrange(masks, "t h w c -> t c h w")
            if len(frames) < num_frames:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
                frames = torch.cat([padding_frames, frames], dim=0)
                padding_masks = masks[:1].repeat(num_frames - len(masks), 1, 1, 1)  # padding with first images
                masks = torch.cat([padding_masks, masks], dim=0)

            return frames, masks
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_mask_from_disk(self, demo_id, view, time_offset, mask_dir, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        if backward:
            time_offset = max(time_offset, self.frame_stack)
            image_indices = np.arange(time_offset - self.frame_stack, time_offset)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
            if view == 'eye_in_hand':
                masks = [np.zeros_like(frame) for frame in frames]
            else:
                masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                         image_indices]
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
        view = self.views[self._index_to_view_id[index]]
        view = 'agentview'
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        masks_dir = os.path.join(demo_parent_dir, "masks", demo_name)

        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w
            else:
                vids, masks = self._load_image_mask_from_disk(demo_id, view, time_offset, masks_dir, backward=True)  # t c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.load_h5(demo_pth)
            demo = self.process_demo(demo)
            vids, masks = self._load_image_mask_from_demo(demo, view, time_offset, masks_dir, backward=True)  # t c h w

        tracks = demo["root"][view]["tracks"][time_offset:time_offset + self.num_track_ts]  # track_len n 2
        vis = demo["root"][view]['vis'][time_offset:time_offset + self.num_track_ts]  # track_len n
        task_emb = demo["root"]["task_emb_bert"]  # (dim,)

        # augment videos
        if np.random.rand() < self.aug_prob:
            vids = vids[None]  # expand to (1, t, c, h, w) to fit the input shape of random shift augmentation
            tracks = tracks[None, None]  # expand to (1, 1, track_len, n, 2) to fit the input shape of random shift augmentation
            vids, tracks = self.augmentor((vids / 255., tracks))
            vids = vids[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        # sample tracks
        tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)

        return vids, tracks, vis, task_emb, masks


class ATMPretrainDataset4(BaseDataset):
    def __init__(self, his_frame=4, pred_frame=4, *args, **kwargs):
        self._index_to_view_id = {}
        super().__init__(*args, **kwargs)
        self.augmentor = transforms.Compose([
            ImgTrackColorJitter2(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug2(input_shape=[128, 128], translation=8, augment_track=self.augment_track),
        ])
        self.his_frame = his_frame
        self.pred_frame = pred_frame

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)

            if self.views is None:
                self.views = list(demo["root"].keys())
                self.views.remove("actions")
                self.views.remove("task_emb_bert")
                self.views.remove("extra_states")
                self.views.sort()

            demo_len = demo["root"][self.views[0]]["video"][0].shape[0]

            if self.cache_all:
                demo = self.process_demo(demo)
                for v in self.views:
                    del demo["root"][v]["video"]
                self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            time_offset = max(time_offset, self.frame_stack)
            image_indices = np.arange(time_offset-self.frame_stack, time_offset)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][image_indices]
            if len(frames) < num_frames:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_mask_from_demo(self, demo, view, time_offset, mask_dir, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        # pred_frames = []
        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames = demo['root'][view]["video"][his_indices]

        pred_indices = np.arange(time_offset + 1, time_offset + 1 + self.pred_frame)
        pred_indices = np.clip(pred_indices, a_min=0, a_max=demo_length - 1)
        pred_frames = demo['root'][view]["video"][pred_indices]

        if view == 'eye_in_hand':
            masks = np.ones((his_frames.shape[0], his_frames.shape[2], his_frames.shape[3])) * 255
        else:
            masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                     his_indices]
            masks = np.stack(masks)  # t h w c
        masks = torch.Tensor(masks).unsqueeze(-1)
        masks = rearrange(masks, "t h w c -> t c h w")
        if len(his_frames) < self.his_frame:
            # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
            padding_frames = his_frames[:1].repeat(num_frames - len(his_frames), 1, 1, 1)  # padding with first images
            his_frames = torch.cat([padding_frames, his_frames], dim=0)
            padding_masks = masks[:1].repeat(num_frames - len(masks), 1, 1, 1)  # padding with first images
            masks = torch.cat([padding_masks, masks], dim=0)
        vids = torch.cat([his_frames, pred_frames], dim=0)  # t c h w
        return vids, masks


    def _load_image_mask_from_disk(self, demo_id, view, time_offset, mask_dir, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in his_indices]

        pred_indices = np.arange(time_offset + 1, time_offset + 1 + self.pred_frame)
        pred_indices = np.clip(pred_indices, a_min=0, a_max=demo_length - 1)
        pred_frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in
                      pred_indices]
        if view == 'eye_in_hand':
            masks = [np.ones((frame.shape[0], frame.shape[1])) * 255 for frame in his_frames]
        else:
            masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
                     his_indices]

        # if backward:
        #     time_offset = max(time_offset, self.frame_stack)
        #     image_indices = np.arange(time_offset - self.frame_stack, time_offset)
        #     image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
        #     frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
        #     if view == 'eye_in_hand':
        #         masks = [np.ones_like(frame) * 255 for frame in frames]
        #     else:
        #         masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
        #                  image_indices]
        #     # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        # else:
        #     image_indices = np.arange(time_offset, time_offset + num_frames)
        #     image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
        #     frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        his_frames = np.stack(his_frames)  # t h w c
        his_frames = torch.Tensor(his_frames)
        his_frames = rearrange(his_frames, "t h w c -> t c h w")
        pred_frames = np.stack(pred_frames)  # t h w c
        pred_frames = torch.Tensor(pred_frames)
        pred_frames = rearrange(pred_frames, "t h w c -> t c h w")
        masks = np.stack(masks)  # t h w c
        masks = torch.Tensor(masks).unsqueeze(-1)
        masks = rearrange(masks, "t h w c -> t c h w")
        vids = torch.cat([his_frames, pred_frames], dim=0)
        return vids, masks

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        view = self.views[self._index_to_view_id[index]]
        # view = 'agentview'
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        masks_dir = os.path.join(demo_parent_dir, "masks1", demo_name)

        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w
            else:
                vids, masks = self._load_image_mask_from_disk(demo_id, view, time_offset, masks_dir, backward=True)  # t c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.load_h5(demo_pth)
            demo = self.process_demo(demo)
            vids, masks = self._load_image_mask_from_demo(demo, view, time_offset, masks_dir, backward=True)  # t c h w

        tracks = demo["root"][view]["tracks"][time_offset:time_offset + self.num_track_ts]  # track_len n 2
        vis = demo["root"][view]['vis'][time_offset:time_offset + self.num_track_ts]  # track_len n
        task_emb = demo["root"]["task_emb_bert"]  # (dim,)
        actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]

        # augment videos
        if np.random.rand() < self.aug_prob:
            vids = vids[None]  # expand to (1, t, c, h, w) to fit the input shape of random shift augmentation
            masks = masks[None]
            tracks = tracks[None, None]  # expand to (1, 1, track_len, n, 2) to fit the input shape of random shift augmentation
            vids, tracks, masks = self.augmentor((vids / 255., tracks, masks))
            vids = vids[0, ...] * 255.
            masks = masks[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        # sample tracks
        tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)
        his_frames = vids[:self.his_frame]
        pred_frames = vids[self.his_frame:]
        return his_frames, pred_frames, tracks, vis, task_emb, masks, actions


class ATMPretrainDataset5(BaseDataset):
    def __init__(self, his_frame=4, pred_frame=4, *args, **kwargs):
        self._index_to_view_id = {}
        super().__init__(*args, **kwargs)
        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=self.img_size, translation=8, augment_track=self.augment_track),
        ])
        self.his_frame = his_frame
        self.pred_frame = pred_frame


    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)

            if self.views is None:
                self.views = list(demo["root"].keys())
                self.views.remove("actions")
                self.views.remove("task_emb_bert")
                self.views.remove("extra_states")
                self.views.sort()

            demo_len = demo["root"][self.views[0]]["video"][0].shape[0]

            if self.cache_all:
                demo = self.process_demo(demo)
                for v in self.views:
                    del demo["root"][v]["video"]
                self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            time_offset = max(time_offset, self.frame_stack)
            image_indices = np.arange(time_offset-self.frame_stack, time_offset)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][image_indices]
            if len(frames) < num_frames:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_mask_from_demo(self, demo, view, time_offset, mask_dir, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        # pred_frames = []
        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames = demo['root'][view]["video"][his_indices]

        pred_indices = np.arange(time_offset + 1, time_offset + 1 + self.pred_frame)
        pred_indices = np.clip(pred_indices, a_min=0, a_max=demo_length - 1)
        pred_frames = demo['root'][view]["video"][pred_indices]

        # if view == 'eye_in_hand':
        #     masks = np.ones((his_frames.shape[0], his_frames.shape[2], his_frames.shape[3])) * 255
        # else:
        #     masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
        #              his_indices]
        #     masks = np.stack(masks)  # t h w c
        # masks = torch.Tensor(masks).unsqueeze(-1)
        # masks = rearrange(masks, "t h w c -> t c h w")
        if len(his_frames) < self.his_frame:
            # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
            padding_frames = his_frames[:1].repeat(num_frames - len(his_frames), 1, 1, 1)  # padding with first images
            his_frames = torch.cat([padding_frames, his_frames], dim=0)
            # padding_masks = masks[:1].repeat(num_frames - len(masks), 1, 1, 1)  # padding with first images
            # masks = torch.cat([padding_masks, masks], dim=0)
        vids = torch.cat([his_frames, pred_frames], dim=0)  # t c h w
        return vids#, masks


    def _load_image_mask_from_disk(self, demo_id, view, time_offset, mask_dir, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in his_indices]

        pred_indices = np.arange(time_offset + 1, time_offset + 1 + self.pred_frame)
        pred_indices = np.clip(pred_indices, a_min=0, a_max=demo_length - 1)
        pred_frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in
                      pred_indices]
        # if view == 'eye_in_hand':
        #     masks = [np.ones((frame.shape[0], frame.shape[1])) * 255 for frame in his_frames]
        # else:
        #     masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
        #              his_indices]

        # if backward:
        #     time_offset = max(time_offset, self.frame_stack)
        #     image_indices = np.arange(time_offset - self.frame_stack, time_offset)
        #     image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
        #     frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
        #     if view == 'eye_in_hand':
        #         masks = [np.ones_like(frame) * 255 for frame in frames]
        #     else:
        #         masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
        #                  image_indices]
        #     # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        # else:
        #     image_indices = np.arange(time_offset, time_offset + num_frames)
        #     image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
        #     frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        his_frames = np.stack(his_frames)  # t h w c
        his_frames = torch.Tensor(his_frames)
        his_frames = rearrange(his_frames, "t h w c -> t c h w")
        pred_frames = np.stack(pred_frames)  # t h w c
        pred_frames = torch.Tensor(pred_frames)
        pred_frames = rearrange(pred_frames, "t h w c -> t c h w")
        # masks = np.stack(masks)  # t h w c
        # masks = torch.Tensor(masks).unsqueeze(-1)
        # masks = rearrange(masks, "t h w c -> t c h w")
        vids = torch.cat([his_frames, pred_frames], dim=0)
        return vids#, masks

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        view = self.views[self._index_to_view_id[index]]
        # view = 'agentview'
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        masks_dir = os.path.join(demo_parent_dir, "masks1", demo_name)

        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w
            else:
                vids = self._load_image_mask_from_disk(demo_id, view, time_offset, masks_dir, backward=True)  # t c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.load_h5(demo_pth)
            demo = self.process_demo(demo)
            vids = self._load_image_mask_from_demo(demo, view, time_offset, masks_dir, backward=True)  # t c h w

        tracks = demo["root"][view]["tracks"][time_offset:time_offset + self.num_track_ts]  # track_len n 2
        vis = demo["root"][view]['vis'][time_offset:time_offset + self.num_track_ts]  # track_len n
        task_emb = demo["root"]["task_emb_bert"]  # (dim,)
        actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        # actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]

        # augment videos
        if np.random.rand() < self.aug_prob:
            vids = vids[None]  # expand to (1, t, c, h, w) to fit the input shape of random shift augmentation
            tracks = tracks[None, None]  # expand to (1, 1, track_len, n, 2) to fit the input shape of random shift augmentation
            vids, tracks = self.augmentor((vids / 255., tracks))
            vids = vids[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        # sample tracks
        tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)
        his_frames = vids[:self.his_frame]
        pred_frames = vids[self.his_frame:]
        return his_frames, pred_frames, tracks, vis, task_emb, actions



class ATMPretrainDataset_Real(Dataset):
    def __init__(self, dataset_dir,
                 img_size,
                 num_track_ts,
                 num_track_ids,
                 frame_stack=1,
                 cache_all=False,
                 cache_image=False,
                 num_demos=None,
                 vis=False,
                 aug_prob=0.,
                 augment_track=True,
                 views=['agent_view', 'hand_view'],
                 extra_state_keys=None,
                 his_frame=4, pred_frame=4, *args, **kwargs):
        self._index_to_view_id = {}
        super().__init__()
        self.dataset_dir = dataset_dir
        self.ori_scale = np.array([0.5, 0.5, 0.5])
        self.pos_scale = np.array([0.05, 0.05, 0.05])
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        # elif isinstance(img_size, list):
        #     img_size = (img_size[0], img_size[1])
        self.img_size = tuple(img_size)
        self.vis = vis
        self.frame_stack = frame_stack
        self.num_demos = num_demos
        self.num_track_ts = num_track_ts
        self.num_track_ids = num_track_ids
        self.aug_prob = aug_prob
        self.augment_track = augment_track
        self.extra_state_keys = extra_state_keys
        self.cache_all = cache_all
        self.cache_image = cache_image
        if not cache_all:
            assert not cache_image, "cache_image is only supported when cache_all is True."

        self.load_image_func = load_rgb

        self.views = views
        if self.views is not None:
            self.views.sort()

        if self.extra_state_keys is None:
            self.extra_state_keys = []

        # if isinstance(self.dataset_dir, str):
        #     self.dataset_dir = [self.dataset_dir]

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

        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=self.img_size, translation=8, augment_track=self.augment_track),
        ])
        self.his_frame = his_frame
        self.pred_frame = pred_frame

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
            agent_view_img_names = ['agent_view' + filename[2:-5]+ '.png' for filename in ee_pos_names]
            hand_view_img_names = ['hand_view' + filename[2:-5] + '.png' for filename in ee_pos_names]
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
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            self._demo_id_to_ee_pos_path[demo_idx] = ee_pos_path
            self._demo_id_to_agent_view_path[demo_idx] = agent_view_img_path
            self._demo_id_to_hand_view_path[demo_idx] = hand_view_img_path
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def _load_image_list_from_demo(self, demo, view, time_offset, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        if backward:
            # image_indices = np.arange(max(time_offset + 1 - num_frames, 0), time_offset + 1)
            # image_indices = np.clip(image_indices, a_min=None, a_max=demo_length-1)
            time_offset = max(time_offset, self.frame_stack)
            image_indices = np.arange(time_offset-self.frame_stack, time_offset)
            image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
            frames = demo['root'][view]["video"][image_indices]
            if len(frames) < num_frames:
                # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
                padding_frames = frames[:1].repeat(num_frames - len(frames), 1, 1, 1)  # padding with first images
                frames = torch.cat([padding_frames, frames], dim=0)
            return frames
        else:
            return demo['root'][view]["video"][time_offset:time_offset + num_frames]

    def _load_image_mask_from_demo(self, demo, view, time_offset, mask_dir, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames
        demo_length = demo["root"][view]["video"].shape[0]
        # pred_frames = []
        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames = demo['root'][view]["video"][his_indices]

        pred_indices = np.arange(time_offset + 1, time_offset + 1 + self.pred_frame)
        pred_indices = np.clip(pred_indices, a_min=0, a_max=demo_length - 1)
        pred_frames = demo['root'][view]["video"][pred_indices]

        # if view == 'eye_in_hand':
        #     masks = np.ones((his_frames.shape[0], his_frames.shape[2], his_frames.shape[3])) * 255
        # else:
        #     masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
        #              his_indices]
        #     masks = np.stack(masks)  # t h w c
        # masks = torch.Tensor(masks).unsqueeze(-1)
        # masks = rearrange(masks, "t h w c -> t c h w")
        if len(his_frames) < self.his_frame:
            # padding_frames = torch.zeros((num_frames - len(frames), *frames.shape[1:]))  # padding with black images
            padding_frames = his_frames[:1].repeat(num_frames - len(his_frames), 1, 1, 1)  # padding with first images
            his_frames = torch.cat([padding_frames, his_frames], dim=0)
            # padding_masks = masks[:1].repeat(num_frames - len(masks), 1, 1, 1)  # padding with first images
            # masks = torch.cat([padding_masks, masks], dim=0)
        vids = torch.cat([his_frames, pred_frames], dim=0)  # t c h w
        return vids#, masks


    def _load_image_mask_from_disk(self, demo_id, view, time_offset, mask_dir, num_frames=None, backward=False):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(os.path.dirname(demo_path))
        demo_name = os.path.basename(demo_path).split(".")[0]
        images_dir = os.path.join(demo_parent_dir, "images", demo_name)

        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in his_indices]

        pred_indices = np.arange(time_offset + 1, time_offset + 1 + self.pred_frame)
        pred_indices = np.clip(pred_indices, a_min=0, a_max=demo_length - 1)
        pred_frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in
                      pred_indices]
        # if view == 'eye_in_hand':
        #     masks = [np.ones((frame.shape[0], frame.shape[1])) * 255 for frame in his_frames]
        # else:
        #     masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
        #              his_indices]

        # if backward:
        #     time_offset = max(time_offset, self.frame_stack)
        #     image_indices = np.arange(time_offset - self.frame_stack, time_offset)
        #     image_indices = np.clip(image_indices, a_min=None, a_max=demo_length - 1)
        #     frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]
        #     if view == 'eye_in_hand':
        #         masks = [np.ones_like(frame) * 255 for frame in frames]
        #     else:
        #         masks = [self.load_image_func(os.path.join(mask_dir, f"{view}_{img_idx}.png")) for img_idx in
        #                  image_indices]
        #     # frames = [np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))] + frames  # padding with black images
        # else:
        #     image_indices = np.arange(time_offset, time_offset + num_frames)
        #     image_indices = np.clip(image_indices, a_min=0, a_max=demo_length-1)
        #     frames = [self.load_image_func(os.path.join(images_dir, f"{view}_{img_idx}.png")) for img_idx in image_indices]

        his_frames = np.stack(his_frames)  # t h w c
        his_frames = torch.Tensor(his_frames)
        his_frames = rearrange(his_frames, "t h w c -> t c h w")
        pred_frames = np.stack(pred_frames)  # t h w c
        pred_frames = torch.Tensor(pred_frames)
        pred_frames = rearrange(pred_frames, "t h w c -> t c h w")
        # masks = np.stack(masks)  # t h w c
        # masks = torch.Tensor(masks).unsqueeze(-1)
        # masks = rearrange(masks, "t h w c -> t c h w")
        vids = torch.cat([his_frames, pred_frames], dim=0)
        return vids#, masks

    def _load_image_from_demo_path(self, demo_id, view, time_offset, num_frames=None,):
        num_frames = self.frame_stack if num_frames is None else num_frames

        demo_path = self._demo_id_to_path[demo_id]
        ee_pos_path = self._demo_id_to_ee_pos_path[demo_id]
        agent_view_img_path = self._demo_id_to_agent_view_path[demo_id]
        hand_view_img_path = self._demo_id_to_hand_view_path[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_name = os.path.basename(demo_path).split(".")[0]
        if view == 'agent_view':
            images_path = agent_view_img_path
        elif view == 'hand_view':
            images_path = hand_view_img_path
        else:
            raise ValueError(f"Unsupported view: {view}")

        his_indices = np.arange(time_offset - self.his_frame + 1, time_offset + 1)
        his_indices = np.clip(his_indices, a_min=0, a_max=demo_length - 1)
        his_frames = [self.load_image_func(images_path[img_idx]) for img_idx in
                      his_indices]

        pred_indices = np.arange(time_offset + 1, time_offset + 1 + self.pred_frame)
        pred_indices = np.clip(pred_indices, a_min=0, a_max=demo_length - 1)
        pred_frames = [self.load_image_func(images_path[img_idx]) for img_idx in
                       pred_indices]

        his_frames = np.stack(his_frames)  # t h w c
        his_frames = torch.Tensor(his_frames)
        his_frames = rearrange(his_frames, "t h w c -> t c h w")
        pred_frames = np.stack(pred_frames)  # t h w c
        pred_frames = torch.Tensor(pred_frames)
        pred_frames = rearrange(pred_frames, "t h w c -> t c h w")
        # masks = np.stack(masks)  # t h w c
        # masks = torch.Tensor(masks).unsqueeze(-1)
        # masks = rearrange(masks, "t h w c -> t c h w")
        vids = torch.cat([his_frames, pred_frames], dim=0)
        vids = torch.nn.functional.interpolate(vids, size=self.img_size, mode='bilinear')
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
        actions_ori = np.array(actions_ori)
        actions_delta_ori = []
        for i in range(len(actions_ori) - 1):
            ad = np.dot(actions_ori[i+1], matrix_inverse(actions_ori[i]))
            ad = mat2quat(ad)
            ad = quat2axisangle(ad)
            ad = ad / self.ori_scale
            actions_delta_ori.append(ad)
        actions_delta_pos = []
        for i in range(len(actions_pos) - 1):
            ad = actions_pos[i+1] - actions_pos[i]
            ad = ad / self.pos_scale
            actions_delta_pos.append(ad)
        actions_delta = np.concatenate([np.array(actions_delta_pos), np.array(actions_delta_ori), actions_gri], axis=-1).astype(np.float32)
        # actions_delta = np.einsum('ijk, ikn -> ijn', actions_ori[1:], matrix_inverse(actions_ori[:-1]))
        # actions_delta = mat2quat(actions_delta)
        # actions_delta = quat2axisangle(actions_delta)
        return vids, actions_delta  # , masks

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        view = self.views[self._index_to_view_id[index]]
        # view = 'agentview'
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        demo_length = self._demo_id_to_demo_length[demo_id]
        demo_path = self._demo_id_to_path[demo_id]
        demo_parent_dir = os.path.dirname(demo_path)
        task_name = os.path.basename(demo_parent_dir).replace('_', ' ')
        demo_name = os.path.basename(demo_path).split(".")[0]
        # masks_dir = os.path.join(demo_parent_dir, "masks1", demo_name)

        # if self.cache_all:
        #     demo = self._cache[demo_id]
        #     if self.cache_image:
        #         vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w
        #     else:
        #         vids = self._load_image_mask_from_disk(demo_id, view, time_offset, masks_dir, backward=True)  # t c h w
        # else:
        #     demo_pth = self._demo_id_to_path[demo_id]
        #     demo = self.load_h5(demo_pth)
        #     demo = self.process_demo(demo)
        #     vids = self._load_image_mask_from_demo(demo, view, time_offset, masks_dir, backward=True)  # t c h w


        vids, actions = self._load_image_from_demo_path(demo_id, view, time_offset)  # t c h w

        tracks = torch.rand(16, 10, 2)  # track_len n 2
        vis = torch.rand(16, 10) # demo["root"][view]['vis'][time_offset:time_offset + self.num_track_ts]  # track_len n
        # task_emb = demo["root"]["task_emb_bert"]  # (dim,)
        task_emb = torch.rand(512, )

        # actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]
        # actions = demo["root"]["actions"][time_offset:time_offset + self.pred_frame]

        # augment videos
        if np.random.rand() < self.aug_prob:
            vids = vids[None]  # expand to (1, t, c, h, w) to fit the input shape of random shift augmentation
            tracks = tracks[None, None]  # expand to (1, 1, track_len, n, 2) to fit the input shape of random shift augmentation
            vids, tracks = self.augmentor((vids / 255., tracks))
            vids = vids[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        # sample tracks
        # tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)
        his_frames = vids[:self.his_frame]
        pred_frames = vids[self.his_frame:]
        return his_frames, pred_frames, tracks, vis, task_emb, actions



if __name__ == '__main__':
    cfg = {
        'img_size': 128,
        'frame_stack': 9,
        'num_track_ts': 16,
        'num_track_ids': 32,
        'cache_all': False,
        'cache_image': False,
    }

    dataset_dir = [
        '/media/huang/T7/data/atm_libero/libero_base/put_the_cream_cheese_in_the_bowl_demo/train'
        ]
    dataset = ATMPretrainDataset5(dataset_dir=dataset_dir, **cfg, aug_prob=0.)
    d = dataset[10]

    dataset_dir = '/home/huang/worldmodel_data'
    dataset = ATMPretrainDataset_Real(dataset_dir=dataset_dir, **cfg, aug_prob=0.9)
    # d = dataset[11046]
    pos_max = -1
    pos_min = 1
    ori_max = -10
    ori_min = 10
    for i in range(11046):
        d = dataset[i]
        action = d[-1]
        pos = action[:, :3]
        if np.max(pos) > pos_max:
            pos_max = np.max(pos)
        if np.min(pos) < pos_min:
            pos_min = np.min(pos)
        ori = action[:, 3:6]
        if np.max(ori) > ori_max:
            ori_max = np.max(ori)
        if np.min(ori) < ori_min:
            ori_min = np.min(ori)

    dataset_dir = ['/home/huang/code/ATM/data/atm_libero/libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo/train/']
    dataset = ATMPretrainDataset5(dataset_dir=dataset_dir, **cfg, aug_prob=0.)
    d = dataset[10]
    pass