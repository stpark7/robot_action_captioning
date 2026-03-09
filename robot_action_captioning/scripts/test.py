import os
import json
import cv2
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

# 모델 ID 및 태스크 설정
MODEL_ID = "mistralai/Ministral-3-14B-Reasoning-2512"
BASE_DIR = Path("/home/lee/sangtae/robocasa/datasets/v1.0/pretrain/atomic")
TASKS = ["PickPlaceCounterToMicrowave", "OpenCabinet", "SlideOvenRack"]

def sample_frames_1fps(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    # Get FPS, default to 30 if not available
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if not fps or fps != fps or fps == 0 else fps
    fps_int = max(1, int(round(fps)))
    
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % fps_int == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_count += 1
    cap.release()
    return frames

def build_prompt(ep_meta):
    instruction = ep_meta.get("lang", "No instruction provided.")
    objects = [obj.get("name", "") for obj in ep_meta.get("object_cfgs", []) if obj.get("name")]
    fixtures = [f"{k} ({v})" for k, v in ep_meta.get("fixture_refs", {}).items()]
    
    context_str = f"The robot's goal is: '{instruction}'."
    if objects: context_str += f" Objects: {', '.join(objects)}."
    if fixtures: context_str += f" Environment: {', '.join(fixtures)}."

    return (
        "You are an AI assistant specialized in analyzing robot manipulation videos. "
        "Please describe the robot's actions in detail.\n"
        "[Task Context]\n" + context_str + "\n"
        "Based on these images, please comprehensively describe the robot's actions."
    )

def analyze_tokens():
    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    for task_name in TASKS:
        print(f"\nAnalyzing Task: {task_name}")
        task_dir = BASE_DIR / task_name
        # Find directory that is not 'videos'
        date_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name != "videos"]
        if not date_dirs: continue
        date_dir = date_dirs[0]
        
        videos_dir = date_dir / "lerobot" / "videos" / "chunk-000"
        paths = {
            "left": videos_dir / "observation.images.robot0_agentview_left" / "episode_000000.mp4",
            "right": videos_dir / "observation.images.robot0_agentview_right" / "episode_000000.mp4",
            "wrist": videos_dir / "observation.images.robot0_eye_in_hand" / "episode_000000.mp4",
            "meta": date_dir / "lerobot" / "extras" / "episode_000000" / "ep_meta.json"
        }
        
        if not all(p.exists() for p in paths.values()):
            print(f"Skipping {task_name}: Missing files")
            continue
            
        with open(paths["meta"], 'r') as f:
            ep_meta = json.load(f)
        
        frames_left = sample_frames_1fps(paths["left"])
        frames_right = sample_frames_1fps(paths["right"])
        frames_wrist = sample_frames_1fps(paths["wrist"])
        num_frames = min(len(frames_left), len(frames_right), len(frames_wrist))
        
        content = []
        for i in range(num_frames):
            content.extend([
                {"type": "text", "text": f"Step {i+1}:"},
                {"type": "image", "image": frames_left[i]},
                {"type": "image", "image": frames_right[i]},
                {"type": "image", "image": frames_wrist[i]}
            ])
        content.append({"type": "text", "text": build_prompt(ep_meta)})
        
        messages = [{"role": "user", "content": content}]
        
        # Tokenize to get counts
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"]
        total_tokens = input_ids.shape[1]
        
        # Count total images
        num_images = num_frames * 3
        
        # Approximate breakdown:
        prompt_text = build_prompt(ep_meta)
        step_texts = "".join([f"Step {i+1}:" for i in range(num_frames)])
        full_text = prompt_text + step_texts
        text_tokens = len(processor.tokenizer.encode(full_text))
        
        vision_tokens = total_tokens - text_tokens
        
        print(f"  - Number of steps: {num_frames}")
        print(f"  - Total images: {num_images}")
        print(f"  - Total Input Tokens: {total_tokens:,}")
        print(f"  - Text Tokens: {text_tokens:,}")
        print(f"  - Vision Tokens: {vision_tokens:,}")
        print(f"  - Context window usage: {(total_tokens/128000)*100:.2f}%")

if __name__ == "__main__":
    analyze_tokens()