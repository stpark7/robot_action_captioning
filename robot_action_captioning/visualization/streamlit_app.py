"""
Robot Action Captioning - Streamlit 시각화 앱

DataLoader를 사용하여 HDF5 데이터를 인터랙티브하게 시각화합니다.
3가지 DataConfig preset을 지원합니다:
1. Singleframe (offset=0)
2. Three Frames (offset=0, 15, 30)
3. Negative Offset (offset=-10, 0, 10)
"""

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_action_captioning.datasets.dataloader import DataLoader
from robot_action_captioning.datasets.dataconfig import DataConfig, TimeOffset
from robot_action_captioning.datasets.datatype import Sample, FrameData
from robot_action_captioning.config.config import HDF5_DATA
from robot_action_captioning.utils.utils import get_hdf5_files, get_demo_ids


# =============================================================================
# DataConfig Presets (test_dataloader.py와 동일)
# =============================================================================

PRESETS = {
    "Singleframe (offset=0)": DataConfig(
        time_offsets=[TimeOffset(offset=0)]
    ),
    "Three Frames (offset=0, 15, 30)": DataConfig(
        time_offsets=[
            TimeOffset(offset=0),
            TimeOffset(offset=15),
            TimeOffset(offset=30),
        ]
    ),
    "Negative Offset (offset=-10, 0, 10)": DataConfig(
        time_offsets=[
            TimeOffset(offset=-10),
            TimeOffset(offset=0),
            TimeOffset(offset=10),
        ]
    ),
}

# Mock caption 예시 (실제 LLM caption 생성 전까지 placeholder)
MOCK_CAPTIONS = [
    "The robot is reaching its right arm toward the cabinet door handle, preparing to grasp it.",
    "The robot extends its gripper to pick up the object from the countertop surface.",
    "The robot is rotating its wrist while maintaining grip on the container, positioning it for placement.",
    "The robotic arm moves upward, lifting the object away from the table surface.",
    "The robot is opening its gripper to release the object into the designated area.",
]


# =============================================================================
# Cached Loader
# =============================================================================

@st.cache_resource
def load_dataloader(hdf5_path: str, demo_id: str, preset_name: str) -> DataLoader:
    """DataLoader를 캐싱하여 반복 로딩 방지."""
    config = PRESETS[preset_name]
    return DataLoader(hdf5_path, demo_id, config)


def get_mock_caption(idx: int) -> str:
    """인덱스에 따라 mock caption 반환."""
    return MOCK_CAPTIONS[idx % len(MOCK_CAPTIONS)]


def render_frame_images(frame: FrameData, frame_label: str):
    """하나의 프레임에 속한 카메라 이미지들을 렌더링."""
    st.markdown(f"#### 📸 Frame: `t{frame.offset:+d}` ({frame_label})")

    if not frame.images:
        st.info("이 프레임에는 이미지가 없습니다.")
        return

    camera_names = sorted(frame.images.keys())
    cols = st.columns(len(camera_names))

    for col, cam_name in zip(cols, camera_names):
        with col:
            img_array = frame.images[cam_name]
            pil_img = Image.fromarray(img_array)
            # 카메라 이름에서 prefix 제거하여 간결하게 표시
            short_name = cam_name.replace("robot0_", "").replace("_image", "")
            st.image(pil_img, caption=short_name, use_container_width=True)


def render_robot_state(frame: FrameData):
    """프레임의 로봇 상태를 표시."""
    if not frame.robot_state:
        return

    with st.expander(f"🤖 Robot State (t{frame.offset:+d})", expanded=False):
        for key, value in sorted(frame.robot_state.items()):
            short_key = key.replace("robot0_", "")
            if isinstance(value, np.ndarray):
                formatted = np.array2string(value, precision=4, suppress_small=True)
            else:
                formatted = str(value)
            st.code(f"{short_key}: {formatted}", language=None)


def render_metadata(sample: Sample, hdf5_name: str, demo_id: str, sample_idx: int):
    """샘플의 메타 정보를 표시."""
    cols = st.columns(4)

    with cols[0]:
        st.metric("HDF5 File", hdf5_name)
    with cols[1]:
        st.metric("Demo ID", demo_id)
    with cols[2]:
        st.metric("Sample Index", sample_idx)
    with cols[3]:
        st.metric("Frames", len(sample.frames))

    if sample.environment:
        env = sample.environment
        st.markdown(
            f"**Environment:** `{env.env_name}` · "
            f"**Robot:** `{env.robot}` · "
            f"**Cameras:** {', '.join(f'`{c}`' for c in env.camera_names)} · "
            f"**Resolution:** {env.camera_width}×{env.camera_height}"
        )

    if sample.episode:
        ep = sample.episode
        st.info(f"🗣️ **Task Description:** {ep.lang}")
        if ep.objects_info:
            obj_names = [obj.name for obj in ep.objects_info]
            st.markdown(f"**Objects:** {', '.join(f'`{n}`' for n in obj_names)}")


def render_caption(sample_idx: int):
    """Mock caption 표시."""
    st.markdown("---")
    st.markdown("### 💬 Generated Caption (Mock)")
    caption = get_mock_caption(sample_idx)
    st.success(caption)
    st.caption("⚠️ 이 caption은 아직 LLM으로 생성된 것이 아닌 예시 데이터입니다.")


# =============================================================================
# Main App
# =============================================================================

def main():
    st.set_page_config(
        page_title="Robot Action Captioning Viewer",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Robot Action Captioning Viewer")
    st.caption("DataLoader를 통해 HDF5 데이터를 시각화하고, LLM 생성 caption을 확인합니다.")

    # ─── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # HDF5 파일 경로
        hdf5_path = st.text_input(
            "HDF5 File Path",
            value=str(HDF5_DATA),
            help="HDF5 파일의 절대 경로를 입력하세요.",
        )

        # 경로 유효성 검사
        hdf5_file = Path(hdf5_path)
        if not hdf5_file.exists():
            st.error(f"파일을 찾을 수 없습니다: {hdf5_path}")
            st.stop()

        # Demo ID 선택
        try:
            demo_ids = get_demo_ids(hdf5_path)
        except Exception as e:
            st.error(f"Demo ID 로딩 실패: {e}")
            st.stop()

        if not demo_ids:
            st.error("HDF5 파일에 demo가 없습니다.")
            st.stop()

        demo_id = st.selectbox("Demo ID", demo_ids)

        st.markdown("---")

        # DataConfig Preset 선택
        preset_name = st.radio(
            "DataConfig Preset",
            list(PRESETS.keys()),
            help="테스트에서 사용한 3가지 DataConfig 중 선택",
        )

        # 선택된 preset 정보 표시
        selected_config = PRESETS[preset_name]
        offsets = [t.offset for t in selected_config.time_offsets]
        st.caption(f"Time offsets: {offsets}")

    # ─── DataLoader 로딩 ─────────────────────────────────────────────────
    try:
        loader = load_dataloader(hdf5_path, demo_id, preset_name)
    except Exception as e:
        st.error(f"DataLoader 생성 실패: {e}")
        st.stop()

    total_samples = len(loader)
    start, end = loader.get_valid_index_range()

    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Valid range:** `[{start}, {end})`")
        st.markdown(f"**Total samples:** `{total_samples}`")

        if total_samples == 0:
            st.warning("유효한 샘플이 없습니다.")
            st.stop()

        # Sample index 선택 — 슬라이더 + number_input 동시 제공
        sample_idx = st.slider(
            "Sample Index",
            min_value=0,
            max_value=total_samples - 1,
            value=0,
            help="조회할 샘플의 인덱스",
        )

    # ─── Sample 로딩 ─────────────────────────────────────────────────────
    actual_idx = start + sample_idx
    sample = loader._load_sample(actual_idx)

    # ─── 메타 정보 ───────────────────────────────────────────────────────
    hdf5_name = Path(hdf5_path).stem
    render_metadata(sample, hdf5_name, demo_id, sample_idx)

    st.markdown("---")

    # ─── 프레임 이미지 ───────────────────────────────────────────────────
    st.markdown("### 🖼️ Frame Images")

    for i, frame in enumerate(sample.frames):
        # 프레임 라벨 생성
        if frame.offset < 0:
            label = "과거"
        elif frame.offset == 0:
            label = "현재"
        else:
            label = "미래"

        render_frame_images(frame, label)

        # Robot State (접이식)
        render_robot_state(frame)

        if i < len(sample.frames) - 1:
            st.markdown("")  # 프레임 간 여백

    # ─── Mock Caption ────────────────────────────────────────────────────
    render_caption(sample_idx)

    # ─── Prompt Preview ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📝 Prompt Preview (Mock)")
    with st.expander("LLM에게 전달될 프롬프트 확인", expanded=False):
        prompt_text = (
            f"You are observing a robot performing a task.\n"
            f"Environment: {sample.environment.env_name if sample.environment else 'N/A'}\n"
            f"Robot: {sample.environment.robot if sample.environment else 'N/A'}\n"
            f"Task: {sample.episode.lang if sample.episode else 'N/A'}\n\n"
            f"The images show the robot at different time steps "
            f"(offsets: {[f.offset for f in sample.frames]}).\n\n"
            f"Describe what action the robot is performing between these frames."
        )
        st.code(prompt_text, language=None)


if __name__ == "__main__":
    main()
