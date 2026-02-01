import h5py
import json
import numpy as np
import os
import argparse
import random
from visualize_data import visualize_data

# Since the control_freq of the robocasa environment is 20Hz,
# setting H=20 compares the current frame with the scene 1 second later.
H = 30

def extract_episode_data(dataset_path: str, episode_id: int):
    """
    Extracts specific information from a Robocasa HDF5 dataset file.
    Extracts data at time t and t+H to describe the action.

    [Extracted Information]
    1. episode_id
    2. env_name
    3. lang
    4. objects_info
    5. robot0_base_to_eef_pos
    6. robot0_base_to_eef_quat
    7. gripper
    8. robot0_agentview_left_image
    9. robot0_agentview_right_image
    10. robot0_eye_in_hand_image
    
    Args:
        dataset_path (str): Path to the .hdf5 file.
        episode_id (int): ID of the episode to extract.
        
    Returns:
        list: A list of dictionaries, each containing data for one episode.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # 1. Retrieve data from environment metadata =========================================
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])

    env_name = env_meta.get("env_name")
    env_kwargs = env_meta.get("env_kwargs", {})
    robot = env_kwargs.get("robots")

    if not (env_name and env_kwargs and robot):
        raise ValueError(
            "Required environment metadata (env_name, env_kwargs, or robots) "
            "is missing or empty"
        )
    print(f"Identified Task/Episode Name: {env_name}")

    # 2. Retrieve data from episode metadata =========================================
    all_demos = sorted(list(f["data"].keys()))
    if episode_id is None:
        selected_demo = random.choice(all_demos)
    else:
        selected_demo = f"demo_{episode_id}"
        if selected_demo not in all_demos:
            raise ValueError(f"Episode {selected_demo} not found in dataset. Available episodes: {len(all_demos)}")

    print(f"Found {len(all_demos)} episodes. Selected: {selected_demo}.")
    print("Extracting data...")

    ep = selected_demo
    ep_grp = f[f"data/{ep}"]
    
    if "ep_meta" not in ep_grp.attrs:
        raise ValueError(f"Episode metadata 'ep_meta' not found for episode {ep}")

    ep_meta = json.loads(ep_grp.attrs["ep_meta"])
    lang = ep_meta.get("lang")
    object_configs = ep_meta.get("object_cfgs")

    if not (lang and object_configs):
        raise ValueError(
            "Required episode metadata (lang, or object_cfgs) "
            "is missing or empty"
        )
    
    # Extract important object information
    objects_info = []
    for cfg in object_configs:
        obj_info = {
            "name": cfg.get("name"),
            "type": cfg.get("type"),
            "category": cfg.get("info", {}).get("cat"),
            "fixture": cfg.get("placement", {}).get("fixture")
        }
        objects_info.append(obj_info)


    # 3. Observations (robot state, images)
    if "obs" not in ep_grp:
        raise ValueError(f"Observation group 'obs' not found for episode {ep}")
        
    obs_grp = ep_grp["obs"]
    
    # Extract data at time t and t+H
    
    num_steps = obs_grp["robot0_base_to_eef_pos"].shape[0]
    
    if num_steps <= H:
        raise ValueError(f"Episode length {num_steps} is shorter than defined horizon {H}")
        
    t = random.randint(0, num_steps - H - 1)
    t_next = t + H
    
    print(f"Sampled time step: t={t}, t_next={t_next} (Total steps: {num_steps})")

    # Extract slices
    robot0_base_to_eef_pos = obs_grp["robot0_base_to_eef_pos"][t]
    robot0_base_to_eef_pos_next = obs_grp["robot0_base_to_eef_pos"][t_next]
    
    robot0_base_to_eef_quat = obs_grp["robot0_base_to_eef_quat"][t]
    robot0_base_to_eef_quat_next = obs_grp["robot0_base_to_eef_quat"][t_next]
    
    gripper = obs_grp["robot0_gripper_qpos"][t]
    gripper_next = obs_grp["robot0_gripper_qpos"][t_next]
    
    robot0_agentview_left_image = obs_grp["robot0_agentview_left_image"][t]
    robot0_agentview_left_image_next = obs_grp["robot0_agentview_left_image"][t_next]
    
    robot0_agentview_right_image = obs_grp["robot0_agentview_right_image"][t]
    robot0_agentview_right_image_next = obs_grp["robot0_agentview_right_image"][t_next]
    
    robot0_eye_in_hand_image = obs_grp["robot0_eye_in_hand_image"][t]
    robot0_eye_in_hand_image_next = obs_grp["robot0_eye_in_hand_image"][t_next]

    # Store in dictionary
    episode_data = {
        "episode_id": ep,                      # e.g., "demo_0"
        "env_name": env_name,                  # e.g., "PnPCounterToCab"
        "lang": lang,                          # Language instruction
        "objects_info": objects_info,          # Object Info
        
        # Sliced Data Arrays at t and t+H
        "robot0_base_to_eef_pos": robot0_base_to_eef_pos,
        "robot0_base_to_eef_pos_next": robot0_base_to_eef_pos_next,
        
        "robot0_base_to_eef_quat": robot0_base_to_eef_quat,
        "robot0_base_to_eef_quat_next": robot0_base_to_eef_quat_next,
        
        "gripper": gripper,
        "gripper_next": gripper_next,
        
        "robot0_agentview_left_image": robot0_agentview_left_image,
        "robot0_agentview_left_image_next": robot0_agentview_left_image_next,
        
        "robot0_agentview_right_image": robot0_agentview_right_image,
        "robot0_agentview_right_image_next": robot0_agentview_right_image_next,
        
        "robot0_eye_in_hand_image": robot0_eye_in_hand_image,
        "robot0_eye_in_hand_image_next": robot0_eye_in_hand_image_next,
    }
         
    f.close()

    return episode_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--episode_id",
        type=int,
        default=None,
        help="ID of the episode to extract",
    )
    args = parser.parse_args()

    try:
        episode_data = extract_episode_data(args.dataset, args.episode_id)

        print(f"\nExtracted: {episode_data['env_name']} | Episode: {episode_data['episode_id']}")
        
        # Prepare text and images for visualization
        text_info = (
            f"Env: {episode_data['env_name']}\n"
            f"Task: {episode_data['lang']}\n"
            f"Objects: {len(episode_data['objects_info'])}\n"
            f"Gripper (t): {episode_data['gripper']}\n"
            f"Gripper (t+H): {episode_data['gripper_next']}"
        )
        
        images_to_show = [
            episode_data["robot0_agentview_left_image"],
            episode_data["robot0_agentview_right_image"],
            episode_data["robot0_eye_in_hand_image"],
            episode_data["robot0_agentview_left_image_next"],
            episode_data["robot0_agentview_right_image_next"],
            episode_data["robot0_eye_in_hand_image_next"]
        ]
        
        visualize_data(text_info, images_to_show)

    except Exception as e:
        print(f"An error occurred: {e}")
