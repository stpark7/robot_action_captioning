import os
from pathlib import Path
from typing import List, TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from robot_action_captioning.datasets.datatype import Sample
    from robot_action_captioning.datasets.dataconfig import DataConfig


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


# =============================================================================
# Prompt 상수 (src/prompt.py에서 이동)
# =============================================================================

PURPOSE = (
    "You need to create text description that describes a robot's actions "
    "based on simulation information. "
    "Refer to the [Information] section in the prompt for detailed simulation data."
)

CONTEXT = (
    "For each step, three images are provided: "
    "Left view, Right view, and Eye-in-hand view. "
    "The specific frame indices for each image are described "
    "in the [Information] section.\n"
    "\n"
    "The Gripper state (robot0_gripper_qpos) indicates "
    "how much the gripper is opened or closed, "
    "with the maximum opening being 8cm. "
    "In other words, values closer to 0.04 indicate an open state, "
    "while values closer to 0 indicate a closed gripper state.\n"
    "\n"
    "robot0_base_pos and robot0_base_quat represent "
    "the position (xyz) and orientation (quaternion) of the robot's base "
    "in the world coordinate frame.\n"
    "\n"
    "robot0_base_to_eef_pos and robot0_base_to_eef_quat represent "
    "the position and orientation of the robot's end-effector "
    "relative to the robot's base frame.\\n"
    "Coordinate directions:\\n"
    "  - x-axis: decreasing x moves toward the robot's base, "
    "increasing x moves away from the base (forward).\\n"
    "  - y-axis: decreasing y moves to the left, "
    "increasing y moves to the right.\\n"
    "  - z-axis: increasing z moves upward, "
    "decreasing z moves downward."
)

INSTRUCTIONS = (
    "1. Describe the robot's actions.\n"
    "   - Include specific movement directions "
    "(up/down, left/right, forward/backward)\n"
    "   - Describe gripper state changes "
    "(opening, closing, maintaining)\n"
    "   \n"
    "2. Explain the intention behind the robot's actions (including Affordance).\n"
    "   - Identify which part of the object is being targeted "
    "(handle, surface, edge, etc.)\n"
    "   - Explain why that specific grasp approach is suitable for the object\n"
    "   \n"
    "3. Describe the spatial relationship between objects.\n"
    "   - Mention relative positions using reference objects (container, distractors)\n"
    "   \n"
)

OUTPUT_FORMAT = (
    "Please write the output concisely in 1-2 sentences."
    "Write in the format of intention + action, and include spatial information if necessary."
    "Example: The robot arm moves forward to grab the mug inside the drawer."
    "The robot arm adjusts the orientation of its end-effector to open the drawer."
)


# =============================================================================
# 동적 Prompt 생성
# =============================================================================

def _format_value(value) -> str:
    """numpy array 또는 스칼라 값을 읽기 좋은 문자열로 변환합니다."""
    if isinstance(value, np.ndarray):
        # 소수점 4자리로 포맷
        return np.array2string(value, precision=4, suppress_small=True, separator=", ")
    return str(value)


def _build_information(sample: "Sample", data_config: "DataConfig") -> str:
    """Sample과 DataConfig를 기반으로 [Information] 섹션 내용을 동적으로 생성합니다.
    
    - environment / episode 메타데이터 포함
    - 각 frame별 이미지 설명 (offset 정보) + robot_state + action 포맷팅
    - 이미지 데이터 자체는 포함하지 않되, 몇 번째 offset의 이미지인지 설명
    
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
        lines.append("")

    # --- 이미지 프레임 설명 ---
    # 이미지가 포함된 frame들의 offset을 설명
    image_frames = [
        (i, to) for i, to in enumerate(data_config.time_offsets) 
        if to.include_image
    ]
    if image_frames and data_config.image_keys:
        num_cameras = len(data_config.image_keys)
        lines.append("Images:")
        img_idx = 1
        for _, to in image_frames:
            start = img_idx
            end = img_idx + num_cameras - 1
            offset_label = f"t" if to.offset == 0 else f"t{to.offset:+d}"
            lines.append(
                f"  Images {start}-{end}: {num_cameras} images at frame {offset_label} "
                f"({', '.join(k.replace('robot0_', '').replace('_image', '') for k in data_config.image_keys)})"
            )
            img_idx = end + 1
        lines.append("")

    # --- 각 Frame별 수치 데이터 ---
    for frame, time_offset in zip(sample.frames, data_config.time_offsets):
        offset_label = f"t" if frame.offset == 0 else f"t{frame.offset:+d}"
        lines.append(f"Frame ({offset_label}):")

        # Robot state
        if time_offset.include_robot_state and frame.robot_state:
            for key in data_config.robot_state_keys:
                if key in frame.robot_state:
                    lines.append(f"  {key}: {_format_value(frame.robot_state[key])}")

        # Action
        if time_offset.include_action and frame.action:
            for key in data_config.action_keys:
                if key in frame.action:
                    lines.append(f"  {key}: {_format_value(frame.action[key])}")

        lines.append("")

    return "\n".join(lines).rstrip()


def generate_prompt(sample: "Sample", data_config: "DataConfig") -> str:
    """Sample과 DataConfig를 기반으로 전체 프롬프트를 동적으로 생성합니다.

    DataConfig에 정의된 robot_state_keys, action_keys에 따라
    [Information] 섹션이 자동으로 구성됩니다.

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
        f"[Information]\n\n{information}",
    ]
    return "\n\n".join(sections)
