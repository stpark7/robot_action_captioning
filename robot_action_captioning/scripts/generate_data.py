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

from robot_action_captioning.utils.utils import get_hdf5_files, get_demo_ids
from robot_action_captioning.datasets.dataloader import DataLoader
from robot_action_captioning.datasets.dataconfig import DataConfig, TimeOffset
from robot_action_captioning.config.config import SAVE_DIR


def convert_to_pil(images: dict) -> List[Image.Image]:
    """이미지 딕셔너리를 PIL Image 리스트로 변환합니다."""
    pil_images = []
    for key in sorted(images.keys()):
        pil_images.append(Image.fromarray(images[key]))
    return pil_images


def generate_action_caption(
    model,
    processor,
    images_current: List[Image.Image],
    images_next: List[Image.Image],
    prompt_text: str,
) -> str:
    """LLM을 사용하여 action caption을 생성합니다."""
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images_current],
                *[{"type": "image", "image": img} for img in images_next],
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
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
    """스냅샷의 이미지들을 시각화하여 저장합니다."""
    from PIL import Image
    import numpy as np

    frames = sample.frames
    all_images = []

    for frame in frames:
        if frame.images:
            for key in sorted(frame.images.keys()):
                all_images.append(frame.images[key])

    if not all_images:
        return

    # 이미지들을 가로로 연결
    pil_images = [Image.fromarray(img) for img in all_images]
    widths, heights = zip(*(img.size for img in pil_images))
    
    total_width = sum(widths)
    max_height = max(heights)

    combined_image = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in pil_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    combined_image.save(output_path)


def main():

    # 출력 디렉토리 생성
    captions_dir = SAVE_DIR / "captions"
    visualizations_dir = SAVE_DIR / "visualizations"

    if args.save_captions:
        captions_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        visualizations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(args.model_name)

    # * 모델에게 제공하고 싶은 포맷대로 DataConfig를 수정
    data_config = DataConfig(
        time_offsets=[
            TimeOffset(offset=0, include_image=True, include_robot_state=True, include_action=False),
            TimeOffset(offset=args.frame_offset, include_image=True, include_robot_state=True, include_action=False),
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

                # 현재 시점(t)과 다음 시점(t+offset)의 이미지 추출
                images_current = frames[0].images if frames[0].images else {}
                images_next = frames[1].images if len(frames) > 1 and frames[1].images else {}

                if not images_current or not images_next:
                    continue

                # Prompt 생성
                prompt_text = generate_prompt(sample)

                # Caption 생성 및 저장
                if args.save_captions and model and processor:
                    pil_current = convert_to_pil(images_current)
                    pil_next = convert_to_pil(images_next)

                    caption = generate_action_caption(
                        model, processor, pil_current, pil_next, prompt_text
                    )

                    caption_filename = f"{hdf5_name}_{demo_id}_frame_{idx:06d}.txt"
                    caption_path = captions_dir / caption_filename
                    save_caption(caption, str(caption_path))

                # 시각화 저장
                if args.save_visualizations:
                    viz_filename = f"{hdf5_name}_{demo_id}_frame_{idx:06d}.png"
                    viz_path = visualizations_dir / viz_filename
                    visualize_and_save(sample, str(viz_path))

    print("\nDone!")


if __name__ == "__main__":
    main()