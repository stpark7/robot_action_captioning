import os
import json
import cv2
import re
import torch
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any

from transformers import AutoProcessor, Mistral3ForConditionalGeneration

MODEL_ID = "mistralai/Ministral-3-14B-Reasoning-2512"
MAX_TOKENS = 8092

def sample_frames_1fps(video_path: str | Path) -> List[Tuple[float, Image.Image]]:
    """
    비디오 파일에서 1 FPS 간격으로 프레임을 추출합니다.
    시작 프레임과 마지막 프레임을 보장하여 중요한 순간의 누락을 방지합니다.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if not fps or fps != fps or fps == 0 else fps
    fps_int = max(1, int(round(fps)))
    
    frames = []
    frame_count = 0
    last_frame_tuple = None
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % fps_int == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append((frame_count / fps, pil_img))
            
        last_frame_tuple = (frame_count, frame)
        frame_count += 1
        
    cap.release()
    
    if last_frame_tuple is not None:
        last_count, last_frame = last_frame_tuple
        if not frames or abs(frames[-1][0] - (last_count / fps)) > 1e-5:
            frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append((last_count / fps, pil_img))
            
    return frames

def build_prompt(ep_meta: Dict[str, Any]) -> str:
    """
    메타데이터(목표, 객체, 환경)를 기반으로 모델에게 전달할 상황 문맥 및 프롬프트를 구성합니다.
    """
    instruction = ep_meta.get("lang", "No instruction provided.")
    
    objects = []
    for obj in ep_meta.get("object_cfgs", []):
        obj_name = obj.get("name", "")
        obj_cat = obj.get("info", {}).get("cat", "")
        if obj_name and obj_cat:
            objects.append(f"{obj_name} ({obj_cat})")
        elif obj_name:
            objects.append(obj_name)
                
    fixtures = [f"{k} ({v})" for k, v in ep_meta.get("fixture_refs", {}).items()]
    
    context_str = f"The robot's goal is: '{instruction}'."
    if objects:
        context_str += f" The objects involved include: {', '.join(objects)}."
    if fixtures:
        context_str += f" The environment fixtures involved include: {', '.join(fixtures)}."

    prompt = (
        "You are an AI assistant specialized in analyzing robot manipulation videos. "
        "Please describe the robot's actions in detail.\n"

        "[Task Context]\n"
        f"{context_str}\n"

        "[Reasoning Steps]\n"
        "Following these steps will help you accurately describe the robot's actions:\n"
        "1. Thoroughly analyze the Task Context. This provides crucial information regarding the instruction given to the robot, surrounding objects, and the target object it intends to manipulate.\n"
        "2. Once you grasp the robot's goal from the context, examine the first frame of the video to understand the overall environment. Identify the objects present and the overall setting.\n"
        "3. Carefully analyze the multiple images provided for the exact same timestep (left, right, and wrist views) to perform a fine-grained analysis of the robot's state and the current situation. If an action or object is occluded or unclear in one view, actively use the other views from the same timestep as supplementary visual context to fully understand the scene.\n"
        "4. Recognize that the background environment and stationary objects generally remain constant. Therefore, focus your attention on the robot's behavior and interactions.\n"

        "[Input Format]\n"
        "The images are sampled at 1 frame per second (1 FPS). "
        "Each step is followed by 3 images in this specific order: left view, right view, then wrist view.\n"
        "For example: 'Step 1: [left_image], [right_image], [wrist_image]'.\n"
        
        "[Output Format]\n"
        "Instead of describing actions strictly chronologically second-by-second, you must group the robot's actions into meaningful, logical phases or task units.\n"
        "For example, if the task is to grab a hotdog and put it in a microwave, structure your response like this:\n"
        "1. Phase: Reaching for the hotdog\n"
        "- [Detailed description of the robot's actions during this phase]\n"
        "2. Phase: Grabbing the hotdog and putting it in the microwave\n"
        "- [Detailed description of the robot's actions during this phase]\n"
        "3. Phase: Returning after placing the hotdog in the microwave\n"
        "- [Detailed description of the robot's actions during this phase]\n"

        "Based on these images, please comprehensively describe the robot's actions following the phased output structure above.\n"

        "# HOW YOU SHOULD THINK AND ANSWER\n"
        "First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. \n"
        "Write both your thoughts and the response in the same language as the input.\n"
        "Your thinking process must follow the template below:\n"
        "[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]\n\n"
        "Here, provide a self-contained response."
    )
    return prompt

def clean_response(raw_response: str) -> str:
    """
    모델의 응답 텍스트에서 내부 사고 과정(Thinking block)과 불필요한 특수 토큰을 정리하여 반환합니다.
    """
    if "</think>" in raw_response:
        assistant_response = raw_response.split("</think>")[-1]
    elif "[/THINK]" in raw_response:
        assistant_response = raw_response.split("[/THINK]")[-1]
    else:
        assistant_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
        assistant_response = re.sub(r"\[THINK\].*?\[/THINK\]", "", assistant_response, flags=re.DOTALL)
        
    assistant_response = re.sub(r"<\|.*?\|>", "", assistant_response)
    assistant_response = assistant_response.replace("</s>", "").replace("<s>", "").strip()
    return assistant_response

def setup_model_and_processor():
    """설정된 Ministral 모델과 프로세서를 GPU 메모리에 로드합니다."""
    print(f"Loading Ministral model ({MODEL_ID})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    return processor, model

def process_task(
    task_name: str, 
    base_dir: Path, 
    output_dir: Path, 
    processor: AutoProcessor, 
    model: Mistral3ForConditionalGeneration
):
    """
    단일 태스크에 대한 다중 시점(좌/우/손목) 비디오 로드, 프레임 추출,
    프롬프트 구성 및 모델 추론을 수행하여 최종 결과를 파일에 저장하는 메인 파이프라인.
    """
    print(f"\n{'='*50}\nProcessing Task: {task_name}\n{'='*50}")
    
    task_dir = base_dir / task_name
    if not task_dir.exists():
        print(f"[Warning] Task directory not found: {task_dir}")
        return
        
    date_dir = next((d for d in task_dir.iterdir() if d.is_dir() and d.name != "videos"), None)
    if not date_dir:
        print(f"[Warning] No date directory found in {task_dir}")
        return
        
    videos_dir = date_dir / "lerobot" / "videos" / "chunk-000"
    paths = {
        "left": videos_dir / "observation.images.robot0_agentview_left" / "episode_000000.mp4",
        "right": videos_dir / "observation.images.robot0_agentview_right" / "episode_000000.mp4",
        "wrist": videos_dir / "observation.images.robot0_eye_in_hand" / "episode_000000.mp4",
        "meta": date_dir / "lerobot" / "extras" / "episode_000000" / "ep_meta.json"
    }
    
    for name, path in paths.items():
        if not path.exists():
            print(f"[Warning] Required {name} file not found: {path}")
            return
            
    with open(paths["meta"], 'r', encoding='utf-8') as f:
        ep_meta = json.load(f)
        
    prompt = build_prompt(ep_meta)
    print(f"Prompt Context:\n{prompt}\n")
    
    print("Extracting frames at 1 fps for each view...")
    frames_left = sample_frames_1fps(paths["left"])
    frames_right = sample_frames_1fps(paths["right"])
    frames_wrist = sample_frames_1fps(paths["wrist"])
    
    min_len = min(len(frames_left), len(frames_right), len(frames_wrist))
    print(f"Extracted frames: left={len(frames_left)}, right={len(frames_right)}, wrist={len(frames_wrist)}. Using {min_len} frames.")
    
    if min_len == 0:
        print("[Warning] No frames extracted.")
        return
        
    content = []
    for i in range(min_len):
        content.extend([
            {"type": "text", "text": f"Step {i+1}:"},
            {"type": "image", "image": frames_left[i][1]},
            {"type": "image", "image": frames_right[i][1]},
            {"type": "image", "image": frames_wrist[i][1]}
        ])
    content.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content}]
    
    try:
        print("Running inference...")
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_TOKENS, 
            pad_token_id=pad_token_id
        )
        
        raw_response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        assistant_response = clean_response(raw_response)
        
        print("Result:")
        print(assistant_response)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{task_name}_ministral_1fps.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(assistant_response)
            
        print(f"Saved result to {output_file}")
        
    except Exception as e:
        print(f"Error during inference: {e}")

def main():
    """정의된 태스크 목록에 대해 로봇 행동 분석 스크립트를 순차적으로 실행합니다."""
    processor, model = setup_model_and_processor()
    
    base_dir = Path("/home/lee/sangtae/robocasa/datasets/v1.0/pretrain/atomic")
    output_dir = Path("/home/lee/sangtae/robot_action_captioning/datasets/sample_results")
    tasks = ["PickPlaceCounterToMicrowave", "OpenCabinet", "SlideOvenRack"]
    
    for task_name in tasks:
        process_task(task_name, base_dir, output_dir, processor, model)

if __name__ == '__main__':
    main()
