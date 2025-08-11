import os
from typing  import List
import torch
import numpy as np
from tqdm import tqdm
import wandb
from PIL import Image
from einops import rearrange
from ladiwm.utils.flow_utils import combine_track_and_img, draw_traj_on_images
from ladiwm.utils.video_utils import video_pad_time
import torchvision as tv
from ladiwm.utils.transform_utils import quat2axisangle

# from get_image_mask import load_model, get_masks

obs_key_mapping = {
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
    # "ee_states": "robot0_eef_pos",
    "ee_pos": "robot0_eef_pos",
}


def rearrange_videos(videos, success, success_vid_first, fail_vid_first):
    success = np.array(success)
    rearrange_idx = np.arange(len(success))
    if success_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[success_idx], videos[fail_idx]], axis=0)
        rearrange_idx = np.concatenate([success_idx, fail_idx], axis=0)
    if fail_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[fail_idx], videos[success_idx]], axis=0)
        rearrange_idx = np.concatenate([fail_idx, success_idx], axis=0)
    return videos, rearrange_idx


def render_done_to_boundary(frame, success, color=(0, 255, 0)):
    """
    If done, render a color boundary to the frame.
    Args:
        frame: (b, c, h, w)
        success: (b, 1)
        color: rgb value to illustrate success, default: (0, 255, 0)
    """
    if any(success):
        b, c, h, w = frame.shape
        color = np.array(color, dtype=frame.dtype)[None, :, None, None]
        boundary = int(min(h, w) * 0.015)
        frame[success, :, :boundary, :] = color
        frame[success, :, -boundary:, :] = color
        frame[success, :, :, :boundary] = color
        frame[success, :, :, -boundary:] = color
    return frame


@torch.no_grad()
def rollout(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                task_emb = obs.get("task_emb", None)
                extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
                a, _tracks = policy.act(rgb, task_emb, extra_states)
                # print(a.shape)
                obs, r, done, info = env.step(a)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results


@torch.no_grad()  # with mask for wm
def rollout2(env_dict, policy, image_predictor, dino_model, device, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []
    text_prompts = ['robotic effector. black bowl', 'robotic effector. black bowls']

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            action = torch.zeros(obs["image"].shape[0], *(policy.act_shape))
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                task_emb = obs.get("task_emb", None)
                extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
                if step_i == 0:
                    use_action = False
                else:
                    use_action = True
                masks = get_masks(rgb, image_predictor, dino_model, text_prompts[0], device)
                a, _tracks, action = policy.act(rgb, task_emb, extra_states, action, use_action=use_action, masks=masks)
                # print(a.shape)
                obs, r, done, info = env.step(a)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results

@torch.no_grad()  # with action for wm
def rollout3(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []
    # text_prompts = ['robotic effector. black bowl', 'robotic effector. black bowls']

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            action = torch.zeros(obs["image"].shape[0], *(policy.act_shape))
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                task_emb = obs.get("task_emb", None)
                extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
                if step_i == 0:
                    use_action = torch.zeros(rgb.shape[0])
                else:
                    use_action = torch.ones(rgb.shape[0])
                # masks = get_masks(rgb, image_predictor, dino_model, text_prompts[0], device)
                a, _tracks, action = policy.act(rgb, task_emb, extra_states, action, use_action=use_action)

                # for kk in range(rgb.shape[0]):
                #     os.makedirs(f'./rollouts/seq_{kk}', exist_ok=True)
                #     pred_view1 = _tracks[kk, 0]
                #     pred_view2 = _tracks[kk, 1]
                #     tv.utils.save_image((pred_view1+1)/2, f'./rollouts/seq_{kk}/view1_{step_i}.jpg')
                #     tv.utils.save_image((pred_view2+1)/2, f'./rollouts/seq_{kk}/view2_{step_i}.jpg')

                # print(a.shape)
                obs, r, done, info = env.step(a)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results

@torch.no_grad()  # with action for wm
def rollout3(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []
    # text_prompts = ['robotic effector. black bowl', 'robotic effector. black bowls']

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            action = torch.zeros(obs["image"].shape[0], *(policy.act_shape))
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                task_emb = obs.get("task_emb", None)
                extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
                if step_i == 0:
                    use_action = torch.zeros(rgb.shape[0])
                else:
                    use_action = torch.ones(rgb.shape[0])
                # masks = get_masks(rgb, image_predictor, dino_model, text_prompts[0], device)
                a, _tracks, action = policy.act(rgb, task_emb, extra_states, action, use_action=use_action)

                # for kk in range(rgb.shape[0]):
                #     os.makedirs(f'./rollouts/seq_{kk}', exist_ok=True)
                #     pred_view1 = _tracks[kk, 0]
                #     pred_view2 = _tracks[kk, 1]
                #     tv.utils.save_image((pred_view1+1)/2, f'./rollouts/seq_{kk}/view1_{step_i}.jpg')
                #     tv.utils.save_image((pred_view2+1)/2, f'./rollouts/seq_{kk}/view2_{step_i}.jpg')

                # print(a.shape)
                obs, r, done, info = env.step(a)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results



@torch.no_grad()
def rollout_text(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                b = rgb.shape[0]
                # task_emb = obs.get("task_emb", None)
                task_desc = env_description.split('/')[-1].replace('_', ' ')
                task_emb = [task_desc for _ in range(b)]
                # print(obs.keys())
                # eef_pos = obs['robot0_eef_pos']
                # eef_quat = obs['robot0_eef_quat']
                # eef_rot = np.zeros((eef_quat.shape[0], 3))
                # for i in range(eef_quat.shape[0]):
                #     eef_rot[i] = quat2axisangle(eef_quat[i])
                # extra_states = {'ee_states': np.concatenate([eef_pos, eef_rot], axis=-1)}
                # extra_states = {k: obs[obs_key_mapping[k]]  for k in policy.extra_state_keys
                #                 }
                extra_states = {}
                for k in policy.extra_state_keys:
                    if k in ['gripper_states', 'joint_states']:
                        dicts = {k: obs[obs_key_mapping[k]]}
                        extra_states.update(dicts)
                if 'ee_states' in policy.extra_state_keys:
                    eef_pos = obs['robot0_eef_pos']
                    eef_quat = obs['robot0_eef_quat']
                    eef_ori = np.zeros((eef_quat.shape[0], 3))
                    for i in range(eef_quat.shape[0]):
                        tmp = eef_quat[i]
                        if tmp[-1] < 0:
                            np.negative(tmp, tmp)
                        eef_ori[i] = quat2axisangle(tmp)
                    ee_states = np.concatenate([eef_pos, eef_ori], axis=-1)
                    dicts = {'ee_states': ee_states}
                    extra_states.update(dicts)
                if 'ee_pos' in policy.extra_state_keys:
                    dicts = {'ee_pos': obs['robot0_eef_pos']}
                    extra_states.update(dicts)
                a, _tracks = policy.act(rgb, task_emb, extra_states)
                # print(a.shape)
                obs, r, done, info = env.step(a)
                # print(r, done, info)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results


@torch.no_grad()
def rollout_text2(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            action = torch.zeros(obs["image"].shape[0], policy.act_shape[0]-1, policy.act_shape[1])
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                b = rgb.shape[0]
                # task_emb = obs.get("task_emb", None)
                task_desc = env_description.split('/')[-1].replace('_', ' ')
                task_emb = [task_desc for _ in range(b)]
                if step_i == 0:
                    use_action = torch.zeros(rgb.shape[0])
                else:
                    use_action = torch.ones(rgb.shape[0])
                # print(obs.keys())
                # eef_pos = obs['robot0_eef_pos']
                # eef_quat = obs['robot0_eef_quat']
                # eef_rot = np.zeros((eef_quat.shape[0], 3))
                # for i in range(eef_quat.shape[0]):
                #     eef_rot[i] = quat2axisangle(eef_quat[i])
                # extra_states = {'ee_states': np.concatenate([eef_pos, eef_rot], axis=-1)}
                # extra_states = {k: obs[obs_key_mapping[k]]  for k in policy.extra_state_keys
                #                 }
                extra_states = {}
                for k in policy.extra_state_keys:
                    if k in ['gripper_states', 'joint_states']:
                        dicts = {k: obs[obs_key_mapping[k]]}
                        extra_states.update(dicts)
                if 'ee_states' in policy.extra_state_keys:
                    eef_pos = obs['robot0_eef_pos']
                    eef_quat = obs['robot0_eef_quat']
                    eef_ori = np.zeros((eef_quat.shape[0], 3))
                    for i in range(eef_quat.shape[0]):
                        tmp = eef_quat[i]
                        if tmp[-1] < 0:
                            np.negative(tmp, tmp)
                        eef_ori[i] = quat2axisangle(tmp)
                    ee_states = np.concatenate([eef_pos, eef_ori], axis=-1)
                    dicts = {'ee_states': ee_states}
                    extra_states.update(dicts)
                if 'ee_pos' in policy.extra_state_keys:
                    dicts = {'ee_pos': obs['robot0_eef_pos']}
                    extra_states.update(dicts)
                a, _tracks, action = policy.act(rgb, task_emb, extra_states, action, use_action)
                action = action[:, 1:]
                # print(a.shape)
                obs, r, done, info = env.step(a)
                # print(r, done, info)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results


@torch.no_grad()
def rollout_text2_loop(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False, loop_times=1):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            # save_mesh(env, './robot_mesh/0.obj')
            action = torch.zeros(obs["image"].shape[0], policy.act_shape[0]-1, policy.act_shape[1])
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                b = rgb.shape[0]
                # task_emb = obs.get("task_emb", None)
                task_desc = env_description.split('/')[-1].replace('_', ' ')
                task_emb = [task_desc for _ in range(b)]
                if step_i == 0:
                    use_action = torch.zeros(rgb.shape[0])
                else:
                    use_action = torch.ones(rgb.shape[0])
                # print(obs.keys())
                # eef_pos = obs['robot0_eef_pos']
                # eef_quat = obs['robot0_eef_quat']
                # eef_rot = np.zeros((eef_quat.shape[0], 3))
                # for i in range(eef_quat.shape[0]):
                #     eef_rot[i] = quat2axisangle(eef_quat[i])
                # extra_states = {'ee_states': np.concatenate([eef_pos, eef_rot], axis=-1)}
                # extra_states = {k: obs[obs_key_mapping[k]]  for k in policy.extra_state_keys
                #                 }
                extra_states = {}
                for k in policy.extra_state_keys:
                    if k in ['gripper_states', 'joint_states']:
                        dicts = {k: obs[obs_key_mapping[k]]}
                        extra_states.update(dicts)
                if 'ee_states' in policy.extra_state_keys:
                    eef_pos = obs['robot0_eef_pos']
                    eef_quat = obs['robot0_eef_quat']
                    eef_ori = np.zeros((eef_quat.shape[0], 3))
                    for i in range(eef_quat.shape[0]):
                        tmp = eef_quat[i]
                        if tmp[-1] < 0:
                            np.negative(tmp, tmp)
                        eef_ori[i] = quat2axisangle(tmp)
                    ee_states = np.concatenate([eef_pos, eef_ori], axis=-1)
                    dicts = {'ee_states': ee_states}
                    extra_states.update(dicts)
                if 'ee_pos' in policy.extra_state_keys:
                    dicts = {'ee_pos': obs['robot0_eef_pos']}
                    extra_states.update(dicts)
                a, _tracks, action = policy.act(rgb, task_emb, extra_states, action, use_action)
                use_action = torch.ones(rgb.shape[0])
                for l in range(loop_times):
                    a, _tracks, action = policy.act(rgb, task_emb, extra_states, action[:, :-1], use_action)
                action = action[:, 1:]
                # print(a.shape)
                obs, r, done, info = env.step(a)
                # print(r, done, info)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1
                # save_mesh(env, f'./robot_mesh/{step_i}.obj')

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break
            # exit()
            with open('steps.txt', 'a') as f:
                f.write(str(step_i))
                f.write('\n')
            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results


@torch.no_grad()
def rollout_text_real(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False, save_image_dir=''):
    try:
        policy.eval()
    except:
        print('Policy has no attribute eval()')
        pass
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset(save_image_dir=save_image_dir)
            try:
                policy.reset()
            except:
                print('Policy has no attribute reset()')
                pass
            done = False
            step_i = 0
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                # save_img = torch.from_numpy(rgb[0]).permute(0, 3, 1, 2)
                # save_img = torch.nn.functional.interpolate(save_img, size=(256, 256), mode='bilinear')
                # tv.utils.save_image(save_img / 255., os.path.join(save_image_dir, str(step_i)+'.png'))

                b = rgb.shape[0]
                # task_emb = obs.get("task_emb", None)
                task_desc = env_description.split('/')[-1].replace('_', ' ')
                task_emb = [task_desc for _ in range(b)]
                # print(obs.keys())
                # eef_pos = obs['robot0_eef_pos']
                # eef_quat = obs['robot0_eef_quat']
                # eef_rot = np.zeros((eef_quat.shape[0], 3))
                # for i in range(eef_quat.shape[0]):
                #     eef_rot[i] = quat2axisangle(eef_quat[i])
                # extra_states = {'ee_states': np.concatenate([eef_pos, eef_rot], axis=-1)}
                # extra_states = {k: obs[obs_key_mapping[k]]  for k in policy.extra_state_keys
                #                 }
                #import pdb; pdb.set_trace()
                extra_states = {}
                for k in policy.extra_state_keys:
                    if k in ['gripper_states', 'joint_states']:
                        dicts = {k: obs[obs_key_mapping[k]]}
                        extra_states.update(dicts)
                if 'ee_states' in policy.extra_state_keys:
                    dicts = {'ee_states': obs['ee_pos'][None]}
                    extra_states.update(dicts)
                if 'ee_pos' in policy.extra_state_keys:
                    dicts = {'ee_pos': obs['ee_pos'][None]}
                    extra_states.update(dicts)
                a_, _tracks = policy.act(rgb, task_emb, extra_states)
                # print(a.shape)
                #a_[:, :6] *= 0.01
                obs, r, done, info = env.step(a_, step_i=step_i, save_image_dir=save_image_dir)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results

@torch.no_grad()
def rollout_text_action_real(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            action = torch.zeros(obs["image"].shape[0], policy.act_shape[0] - 1, policy.act_shape[1])
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                b = rgb.shape[0]
                # task_emb = obs.get("task_emb", None)
                task_desc = env_description.split('/')[-1].replace('_', ' ')
                task_emb = [task_desc for _ in range(b)]
                if step_i == 0:
                    use_action = torch.zeros(rgb.shape[0])
                else:
                    use_action = torch.ones(rgb.shape[0])
                # print(obs.keys())
                # eef_pos = obs['robot0_eef_pos']
                # eef_quat = obs['robot0_eef_quat']
                # eef_rot = np.zeros((eef_quat.shape[0], 3))
                # for i in range(eef_quat.shape[0]):
                #     eef_rot[i] = quat2axisangle(eef_quat[i])
                # extra_states = {'ee_states': np.concatenate([eef_pos, eef_rot], axis=-1)}
                # extra_states = {k: obs[obs_key_mapping[k]]  for k in policy.extra_state_keys
                #                 }
                extra_states = {}
                for k in policy.extra_state_keys:
                    if k in ['gripper_states', 'joint_states']:
                        dicts = {k: obs[obs_key_mapping[k]]}
                        extra_states.update(dicts)
                if 'ee_states' in policy.extra_state_keys:
                    dicts = {'ee_states': obs['ee_pos'][None]}
                    extra_states.update(dicts)
                a, _tracks, action = policy.act(rgb, task_emb, extra_states, action, use_action)
                # print(a.shape)
                obs, r, done, info = env.step(a)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape
                _tracks = None
                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results


def merge_results(results: List[dict], compute_avg=True):
    merged_results = {}
    for result_dict in results:
        for k, v in result_dict.items():
            if k in merged_results:
                if isinstance(v, list):
                    merged_results[k].append(v)
                else:
                    merged_results[k] = [merged_results[k], v]
            else:
                merged_results[k] = v

    if compute_avg:
        merged_results["rollout/return_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/return_env" in k]).flatten())
        merged_results["rollout/horizon_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/horizon_env" in k]).flatten())
        merged_results["rollout/success_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/success_env" in k]).flatten())
    return merged_results


def save_mesh(env, save_path):
    from scipy.spatial.transform import Rotation as R
    import trimesh

    model = env.workers[0].env.env.env.env.sim.model
    data = env.workers[0].env.env.env.env.sim.data
    mesh_dict = {}
    for mesh_id in range(model.nmesh):
        name = model.mesh_id2name(mesh_id)
        # if not any(k in name for k in ["robot0", "panda", "gripper", "link", "hand"]):
        if not any(k in name for k in ["robot0", "gripper", "link", 'finger']):
            continue

        # 确保顶点/面数据为numpy数组
        vert = np.array(model.mesh_vert[
                        model.mesh_vertadr[mesh_id]:
                        model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]
                        ], dtype=np.float64)

        face = np.array(model.mesh_face[
                        model.mesh_faceadr[mesh_id]:
                        model.mesh_faceadr[mesh_id] + model.mesh_facenum[mesh_id]
                        ], dtype=np.int64)

        mesh_dict[name] = trimesh.Trimesh(vertices=vert, faces=face)
        # print(f"📦 加载网格: {name} (顶点: {len(vert)}, 面: {len(face)})")

    # mesh_dict['gripper0_finger_vis2'] = mesh_dict['gripper0_finger_vis']

    # 2. 实时位姿计算（使用data而非model）
    def get_world_transform(body_id):
        """获取考虑父子关系的世界变换矩阵"""
        xpos = data.body_xpos[body_id]
        xmat = data.body_xmat[body_id].reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :3] = xmat
        transform[:3, 3] = xpos
        return transform

    keep_mesh = ['robot0_link0', 'robot0_link1', 'robot0_link2', 'robot0_link3', 'robot0_link4', 'robot0_link5',
                 'robot0_link6', 'robot0_link7',
                 'gripper0_hand_vis', 'gripper0_finger_vis', ]
    mesh_dict2 = {}
    for k in keep_mesh:
        if k in mesh_dict:
            mesh_dict2[k] = mesh_dict[k]
    # mesh_dict2 = mesh_dict
    mesh_dict2['gripper0_finger_vis2'] = mesh_dict['gripper0_finger_vis'].copy()
    # 3. 精确位姿应用
    applied_count = 0
    for geom_id in range(model.ngeom):
        geom_name = model.geom(geom_id).name or ""
        if not any(k in geom_name for k in ["robot0", "panda", "gripper"]):
            # if not any(k in geom_name for k in ["robot0", "panda"]):
            continue

        # 智能网格名称匹配（处理所有命名变体）
        mesh_candidates = [
            geom_name.replace("_collision", "_vis"),
            # geom_name + "_vis",
            # geom_name.replace("gripper0", "gripper"),
            geom_name.replace("finger1_collision", "finger_vis"),
            geom_name.replace("finger2_collision", "finger_vis2"),
            geom_name.split("_collision")[0],
            # *[n for n in mesh_dict2.keys() if n.startswith(geom_name.split("_collision")[0])]
        ]
        # print('ge:', geom_name)
        # print('ca:', mesh_candidates)
        for candidate in mesh_candidates:
            # if candidate == 'robot0_link1_vis' or candidate == 'robot0_link2_vis':
            #     continue

            if candidate in mesh_dict2:
                if candidate == 'robot0_link1_vis' or candidate == 'robot0_link2_vis':
                    candidate = candidate[:-4]
                body_id = model.geom_bodyid[geom_id]

                # 计算全局变换（包含几何体局部偏移）
                world_transform = get_world_transform(body_id)
                geom_offset = np.eye(4)
                geom_offset[:3, :3] = R.from_quat(model.geom_quat[geom_id][[1, 2, 3, 0]]).as_matrix()
                geom_offset[:3, 3] = model.geom_pos[geom_id]

                final_transform = world_transform @ geom_offset
                # if "gripper0_finger2" in geom_name:
                #     print('1111')
                #     mirror_transform = np.eye(4)
                #     mirror_transform[1, 1] = -1  # Y轴镜像
                #     final_transform = final_transform @ mirror_transform
                #     final_transform = final_transform.astype(np.float32)

                # 应用变换并标记
                mesh_dict2[candidate].apply_transform(final_transform)
                applied_count += 1
                # print(f"✅ 应用位姿: {geom_name.ljust(25)} → {candidate}")
                # if geom_name == 'gripper0_finger2_collision':
                #     mesh_dict2['gripper0_finger_vis2'].apply_transform(final_transform)
                #     print(f"✅ 应用位姿: {geom_name.ljust(25)} → {'gripper0_finger_vis2'}")
                break

    # 4. 安全导出（处理编码问题）
    if applied_count == 0:
        available = [n for n in mesh_dict2.keys() if any(k in n for k in ["vis", "link", "hand"])]
        # available = [n for n in mesh_dict.keys() if any(k in n for k in ["vis", "link"])]
        raise ValueError(
            "⚠️ 位姿应用失败！请检查:\n"
            f"匹配的几何体: {[model.geom(i).name for i in range(model.ngeom) if 'robot0' in (model.geom(i).name or '')]}\n"
            f"可用网格: {available}"
        )

    # 合并前验证变换
    for name, mesh in mesh_dict2.items():
        if not np.allclose(mesh.bounds, [[-1, -1, -1], [1, 1, 1]], atol=10):  # 检查是否已应用变换
            print(f"⚠️ 可能未变换的网格: {name} (边界: {mesh.bounds})")

    # 二进制模式写入防止编码错误
    combined = trimesh.util.concatenate(list(mesh_dict2.values()))
    with open(save_path, 'wb') as f:
        f.write(trimesh.exchange.obj.export_obj(combined).encode('ascii', errors='ignore'))