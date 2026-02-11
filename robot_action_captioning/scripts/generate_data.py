"""
Robot Action Captioning 데이터 생성 스크립트

HDF5 파일에서 데이터를 로드하여:
1. LLM을 통해 action caption 생성
2. 시각화 이미지 저장
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional

import h5py
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from robot_action_captioning.utils.utils import get_hdf5_files, get_demo_ids, generate_prompt
from robot_action_captioning.datasets.dataloader import DataLoader
from robot_action_captioning.datasets.dataconfig import DataConfig, TimeOffset
from robot_action_captioning.config.config import SAVE_DIR, LLM_MODEL


def convert_to_pil(images: dict) -> List[Image.Image]:
    """이미지 딕셔너리를 PIL Image 리스트로 변환합니다."""
    pil_images = []
    for key in sorted(images.keys()):
        pil_images.append(Image.fromarray(images[key]))
    return pil_images


def generate_action_caption(
    model,
    processor,
    images: List[Image.Image],
    prompt_text: str,
) -> str:
    """LLM을 사용하여 action caption을 생성합니다."""
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        add_vision_id=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=2000)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])


def save_caption(caption: str, output_path: str) -> None:
    """생성된 caption을 파일로 저장합니다."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(caption)


def visualize_and_save(
    sample,
    output_path: str,
) -> None:
    """스냅샷의 이미지들을 시각화하여 저장합니다. (3열 격자 배치)"""
    from PIL import Image

    frames = sample.frames
    all_images = []

    for frame in frames:
        if frame.images:
            for key in sorted(frame.images.keys()):
                all_images.append(frame.images[key])

    if not all_images:
        return

    pil_images = [Image.fromarray(img) for img in all_images]
    
    # 3열 격자 계산
    cols = 3
    rows = (len(pil_images) + cols - 1) // cols
    
    # 그리드 셀 크기 결정을 위한 최대 너비/높이 계산
    widths, heights = zip(*(img.size for img in pil_images))
    max_width = max(widths)
    max_height = max(heights)
    
    total_width = max_width * cols
    total_height = max_height * rows

    combined_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    
    for idx, img in enumerate(pil_images):
        col = idx % cols
        row = idx // cols
        
        x_offset = col * max_width
        y_offset = row * max_height
        
        combined_image.paste(img, (x_offset, y_offset))

    combined_image.save(output_path)


def main(args):
    print(f"Loading model: {LLM_MODEL}")
    processor = AutoProcessor.from_pretrained(LLM_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(LLM_MODEL)

    # * 모델에게 제공하고 싶은 포맷대로 DataConfig를 수정
    data_config = DataConfig(
        time_offsets=[
            TimeOffset(offset=0, include_image=True, include_robot_state=True, include_action=False),
            TimeOffset(offset=30, include_image=True, include_robot_state=True, include_action=False),
        ],
    )

    # HDF5 파일 리스트 가져오기
    hdf5_files = get_hdf5_files(HDF5_PATH)
    print(f"Found {len(hdf5_files)} HDF5 files")

    # TODO : 한 샘플당 생성하는데 몇 분정도 걸리는지 테스트
    for hdf5_path in hdf5_files:
        hdf5_name = Path(hdf5_path).stem
        print(f"\nProcessing: {hdf5_name}")

        # demo ID 순회
        demo_ids = get_demo_ids(hdf5_path)
        print(f"  Found {len(demo_ids)} demos")

        for demo_id in demo_ids:
            print(f"  Processing demo: {demo_id}")

            try:
                loader = DataLoader(hdf5_path, demo_id, data_config)
            except Exception as e:
                print(f"    Error loading demo {demo_id}: {e}")
                continue

            # 각 sample 순회
            for idx, sample in enumerate(loader):
                frames = sample.frames
                episode = sample.episode
                environment = sample.environment

                # DataConfig에 맞게 이미지가 포함된 모든 프레임에서 동적으로 수집
                all_pil_images = []
                for frame in frames:
                    if frame.images:
                        all_pil_images.extend(convert_to_pil(frame.images))

                # 저장 경로 생성: SAVE_DIR / hdf5_name / demo_id / idx
                save_path = SAVE_DIR / hdf5_name / demo_id / f"sample_{idx}"
                save_path.mkdir(parents=True, exist_ok=True)

                # Prompt 생성 및 저장
                prompt_text = generate_prompt(sample, data_config)
                with open(save_path / "prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt_text)

                # 시각화 저장
                visualize_and_save(sample, str(save_path / "image.png"))

                # Caption 생성 및 저장
                caption = generate_action_caption(
                    model, processor, all_pil_images, prompt_text
                )

                save_caption(caption, str(save_path / "caption.txt"))

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--save_captions", 
        action="store_true", 
        default=True, 
        help="Save generated captions"
    )
    
    parser.add_argument(
        "--save_visualizations", 
        action="store_true", 
        default=True, 
        help="Save visualizations"
    )
    
    args = parser.parse_args()
    
    main(args)