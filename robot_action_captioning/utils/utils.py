import os
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import h5py
import numpy as np
from PIL import Image

from robot_action_captioning.datasets.datatype import Sample
from robot_action_captioning.datasets.dataconfig import DataConfig
from robot_action_captioning.config.config import DATA_HZ

# Camera key -> human-readable view name
CAMERA_VIEW_NAMES: Dict[str, str] = {
    "robot0_agentview_left_image": "left camera view",
    "robot0_agentview_right_image": "right camera view",
    "robot0_eye_in_hand_image": "eye-in-hand view",
}

PURPOSE = (
    "Describes a robot's actions based on the provided robot's observations and proprioception."
)

CONTEXT = (
    "Images are provided in chronological order with timestamps. "
    "Each timestamp has three camera views: left camera view, right camera view, "
    "and eye-in-hand view. "
    "The timestamps indicate the elapsed time from the start of the observation sequence.\n"
    "The Gripper state (robot0_gripper_qpos) indicates "
    "how much the gripper is opened or closed, "
    "with the maximum opening being 8cm. "
    "In other words, values closer to 0.04 indicate an open state, "
    "while values closer to 0 indicate a closed gripper state.\n"
    "In the Eye-in-hand view, if the object and the gripper are in contact, "
    "the gripper is closed rather than open.\n"
    "Coordinate directions:\\n"
    "  - x-axis: decreasing x moves away from the base (forward), "
    "increasing x moves toward the robot's base.\\n"
    "  - y-axis: decreasing y moves to the left, "
    "increasing y moves to the right.\\n"
    "  - z-axis: increasing z moves upward, "
    "decreasing z moves downward."
)

INSTRUCTIONS = (
    "1. Describe the robot's actions.\n"
    "2. Explain the intention behind the robot's actions (including Affordance).\n"
)

OUTPUT_FORMAT = (
    "Output should be a single caption that describes the robot's actions. "
    "Do not list the requirements from the Instructions separately. "
    "Instead, integrate them into 3 to 4 sentences."
)

CAUTION = (
    "Verify temporal consistency: Check the FIRST timestamp as reference point, "
    "then COMPARE all subsequent timestamps to ensure logical sequence.\n"  
)


def get_hdf5_files(path: str) -> List[str]:
    """
    주어진 경로에서 모든 HDF5 파일을 재귀적으로 찾아 리스트로 반환합니다.
    
    Args:
        path: HDF5 파일을 검색할 디렉토리 경로
        
    Returns:
        HDF5 파일 경로들의 리스트
    """
    hdf5_files = []
    root_path = Path(path)
    
    if not root_path.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path}")
    
    if not root_path.is_dir():
        raise NotADirectoryError(f"디렉토리가 아닙니다: {path}")
    
    # 재귀적으로 모든 .hdf5 및 .h5 파일 검색
    for file_path in root_path.rglob("*.hdf5"):
        hdf5_files.append(str(file_path))
    
    return sorted(hdf5_files)


def get_demo_ids(hdf5_path: str) -> List[str]:
    """HDF5 파일에서 모든 demo ID를 가져옵니다."""
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return []
        return sorted([key for key in f["data"].keys() if key.startswith("demo_")])


def _format_value(value) -> str:
    """numpy array 또는 스칼라 값을 읽기 좋은 문자열로 변환합니다."""
    if isinstance(value, np.ndarray):
        # 소수점 4자리로 포맷
        return np.array2string(value, precision=4, suppress_small=True, separator=", ")
    return str(value)


def _build_information(sample: "Sample", data_config: "DataConfig") -> str:
    """Sample의 Environment, Task, Objects 메타데이터로 [Information] 섹션을 생성합니다.
    
    Args:
        sample: DataLoader가 yield한 Sample 객체
        data_config: 현재 사용 중인 DataConfig
    
    Returns:
        포맷팅된 Information 문자열
    """
    lines: List[str] = []

    # --- Environment 정보 ---
    if sample.environment:
        lines.append(f"Environment: {sample.environment.env_name}")

    # --- Episode 정보 ---
    if sample.episode:
        lines.append(f"Task: {sample.episode.lang}")
        if sample.episode.objects_info:
            lines.append("Objects:")
            for obj in sample.episode.objects_info:
                parts = [obj.name]
                if obj.category:
                    parts.append(obj.category)
                lines.append(f"  - {' | '.join(parts)}")

    return "\n".join(lines).rstrip()


def generate_prompt(sample: "Sample", data_config: "DataConfig") -> str:
    """Sample과 DataConfig를 기반으로 텍스트 전용 프롬프트를 생성합니다.

    이미지 인터리빙 없이 텍스트만 반환합니다.
    이미지와 텍스트를 혼합한 프롬프트는 generate_prompt_messages()를 사용하세요.

    Args:
        sample: DataLoader가 yield한 Sample 객체
        data_config: 어떤 데이터를 포함할지 정의하는 DataConfig

    Returns:
        완성된 프롬프트 문자열
    """
    information = _build_information(sample, data_config)

    sections = [
        f"[Purpose]\n{PURPOSE}",
        f"[Context]\n{CONTEXT}",
        f"[Instructions]\n\n{INSTRUCTIONS}",
        f"[Output Format]\n\n{OUTPUT_FORMAT}",
        f"[Caution]\n\n{CAUTION}",
        f"[Information]\n\n{information}",
    ]
    return "\n\n".join(sections)


def generate_prompt_messages(
    sample: "Sample", data_config: "DataConfig"
) -> List[Dict[str, Any]]:
    """텍스트를 먼저 전달하고, 이미지를 뒤에 모아서 전달하는 content 리스트를 생성합니다.

    반환 구조:
        [
            {"type": "text", "text": "[Text Prompt + Image Description + Proprioception]"},
            {"type": "image", "image": PIL.Image},  # image 1
            {"type": "image", "image": PIL.Image},  # image 2
            ...
        ]

    Args:
        sample: DataLoader가 yield한 Sample 객체
        data_config: 어떤 데이터를 포함할지 정의하는 DataConfig

    Returns:
        messages의 content 리스트 (text/image dict 혼합)
    """
    # 1. Build base text prompt
    information = _build_information(sample, data_config)
    text_sections = [
        f"[Purpose]\n{PURPOSE}",
        f"[Context]\n{CONTEXT}",
        f"[Instructions]\n\n{INSTRUCTIONS}",
        f"[Output Format]\n\n{OUTPUT_FORMAT}",
        f"[Caution]\n\n{CAUTION}",
        f"[Information]\n\n{information}",
    ]

    # 2. Build [Image Description] and [Proprioception] sections, collect images
    image_desc_lines: List[str] = []
    prop_sections: List[str] = []
    all_images: List[Image.Image] = []
    image_index = 1

    for frame, time_offset in zip(sample.frames, data_config.time_offsets):
        timestamp_sec = frame.offset / DATA_HZ

        # 2a. Collect images and build descriptions
        if time_offset.include_image and frame.images:
            for image_key in data_config.image_keys:
                if image_key not in frame.images:
                    continue

                view_name = CAMERA_VIEW_NAMES.get(
                    image_key,
                    image_key.replace("robot0_", "").replace("_image", "")
                )

                image_desc_lines.append(
                    f"  Image {image_index}: {view_name} at <{timestamp_sec:.1f} seconds>"
                )
                all_images.append(Image.fromarray(frame.images[image_key]))
                image_index += 1

        # 2b. Collect proprioception
        prop_lines: List[str] = []

        if time_offset.include_robot_state and frame.robot_state:
            for key in data_config.robot_state_keys:
                if key in frame.robot_state:
                    prop_lines.append(f"  {key}: {_format_value(frame.robot_state[key])}")

        if time_offset.include_action and frame.action:
            for key in data_config.action_keys:
                if key in frame.action:
                    prop_lines.append(f"  {key}: {_format_value(frame.action[key])}")

        if prop_lines:
            prop_header = f"<{timestamp_sec:.1f} seconds>"
            prop_sections.append(prop_header + "\n" + "\n".join(prop_lines))

    # 3. Append Image Description section
    if image_desc_lines:
        text_sections.append(
            "[Image Description]\n" + "\n".join(image_desc_lines)
        )

    # 4. Append Proprioception section
    if prop_sections:
        text_sections.append(
            "[Proprioception]\n" + "\n\n".join(prop_sections)
        )

    # 5. Build content: text first, then all images
    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": "\n\n".join(text_sections)})

    for pil_img in all_images:
        content.append({"type": "image", "image": pil_img})

    return content
