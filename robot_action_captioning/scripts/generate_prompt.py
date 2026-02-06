import numpy as np
import argparse
import random
import os
from extract_data_from_hdf5 import extract_episode_data

TEMPLATE_PATH = os.path.expanduser("~/robot_action_captioning/src/action_captioning_prompt")

def format_array(arr):
    """Formats a numpy array to a string with 3 decimal places."""
    return np.array2string(arr, formatter={'float_kind':lambda x: "%.3f" % x}, separator=', ')

def generate_prompt(episode_data):
    """
    Generates the prompt for action captioning.

    Args:
        episode_data (dict): Data extracted from HDF5 containing states at t and t+H.

    Returns:
        tuple:
            - str: The complete prompt text.
            - list[np.ndarray]: List of 6 images in order (t: left, right, hand; t+H: left, right, hand).
    """
    
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")

    # 2. Extract specific data
    env_name = episode_data.get("env_name", "Unknown")
    lang = episode_data.get("lang", "Unknown")
    objects_info = episode_data.get("objects_info", [])
    
    # Extract states at t and t+H
    # NOTE: extract_data_from_hdf5.py now returns data at sampled t and t+H
    
    eef_pos_t = episode_data["robot0_base_to_eef_pos"]
    eef_quat_t = episode_data["robot0_base_to_eef_quat"]
    gripper_t = episode_data["gripper"]
    
    eef_pos_next = episode_data["robot0_base_to_eef_pos_next"]
    eef_quat_next = episode_data["robot0_base_to_eef_quat_next"]
    gripper_next = episode_data["gripper_next"]
    
    # Format Object Info
    obj_info_str = ""
    for obj in objects_info:
        obj_info_str += f"- {obj['name']} ({obj['type']}): {obj['category']}\n"
    
    # 3. Construct the Information section
    info_text = f"""
Task: {env_name}
Instruction: {lang}

Objects:
{obj_info_str}
[State at t]
EEF Position: {format_array(eef_pos_t)}
EEF Quaternion: {format_array(eef_quat_t)}
Gripper State: {gripper_t} ({format_array(gripper_t)})

[State at t+H]
EEF Position: {format_array(eef_pos_next)}
EEF Quaternion: {format_array(eef_quat_next)}
Gripper State: {gripper_next} ({format_array(gripper_next)})
"""

    # 4. Append to template
    # The template ends with [Information]
    full_prompt = template.strip() + "\n\n" + info_text.strip()
    
    # 5. Collect images in order
    images = [
        episode_data["robot0_agentview_left_image"],
        episode_data["robot0_agentview_right_image"],
        episode_data["robot0_eye_in_hand_image"],
    ]

    images_next = [
        episode_data["robot0_agentview_left_image_next"],
        episode_data["robot0_agentview_right_image_next"],
        episode_data["robot0_eye_in_hand_image_next"]
    ]
    
    return full_prompt, images + images_next

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

    print(f"Loading data from {args.dataset}...")
    
    try:
        # Extract real data
        episode_data = extract_episode_data(args.dataset, args.episode_id)
            
        print(f"Generating prompt for Episode: {episode_data['episode_id']}")

        prompt, images = generate_prompt(episode_data)
        
        print("\n" + "="*50)
        print("GENERATED PROMPT:")
        print("="*50)
        print(prompt)
        print("="*50)
        print(f"Returned {len(images)} images (Type: {type(images[0])})")
        print("="*50)

    except Exception as e:
        print(f"An error occurred: {e}")
