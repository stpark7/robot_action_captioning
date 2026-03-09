"""
Robot Action Captioning - Training Data Visualization App

caption_data 디렉토리의 학습 데이터를 시각화합니다.
왼쪽 사이드바에서 Task / Demo / Sample을 선택하고,
image.png, prompt.txt의 [Information] 블록, caption.txt를 순서대로 표시합니다.
"""

import re
import streamlit as st
from pathlib import Path
from PIL import Image

# =============================================================================
# Constants
# =============================================================================

CAPTION_DATA_DIR = Path("/home/lee/sangtae/robot_action_captioning/src/caption_data")


# =============================================================================
# Helpers
# =============================================================================

def get_sorted_subdirs(parent: Path) -> list[str]:
    """Return sorted list of subdirectory names under *parent*."""
    if not parent.is_dir():
        return []
    dirs = [d.name for d in parent.iterdir() if d.is_dir()]
    # Natural sort: demo_2 before demo_10
    def _natural_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
    return sorted(dirs, key=_natural_key)


def extract_information_block(prompt_path: Path) -> str:
    """prompt.txt에서 [Information] 블록 이하의 텍스트를 추출."""
    text = prompt_path.read_text(encoding="utf-8")
    # [Information]이 줄의 시작에 오는 경우만 매칭 (본문 내 참조 제외)
    match = re.search(r"(?m)^\[Information\]", text)
    if match is None:
        return text  # 마커가 없으면 전체 반환
    return text[match.start():]


def read_caption(caption_path: Path) -> str:
    """caption.txt 내용을 읽어 반환."""
    text = caption_path.read_text(encoding="utf-8").strip()
    # 끝에 붙을 수 있는 special token 제거
    text = re.sub(r"<\|.*?\|>$", "", text).strip()
    return text


# =============================================================================
# Main App
# =============================================================================

def main():
    st.set_page_config(
        page_title="Caption Data Viewer",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Caption Data Viewer")
    st.caption("학습 데이터(image / prompt info / caption)를 확인합니다.")

    # ─── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📂 Data Selection")

        # 1) DataConfig 선택
        configs = get_sorted_subdirs(CAPTION_DATA_DIR)
        if not configs:
            st.error(f"데이터가 없습니다: {CAPTION_DATA_DIR}")
            st.stop()

        config = st.selectbox("DataConfig", configs)

        # 2) Task 선택
        config_dir = CAPTION_DATA_DIR / config
        tasks = get_sorted_subdirs(config_dir)
        if not tasks:
            st.warning(f"'{config}'에 task가 없습니다.")
            st.stop()

        task = st.selectbox("Task", tasks)

        # 3) Demo 선택
        task_dir = config_dir / task
        demos = get_sorted_subdirs(task_dir)
        if not demos:
            st.warning(f"'{config}/{task}'에 demo가 없습니다.")
            st.stop()

        demo = st.selectbox("Demo", demos)

        # 4) Sample 선택
        demo_dir = task_dir / demo
        samples = get_sorted_subdirs(demo_dir)
        if not samples:
            st.warning(f"'{config}/{task}/{demo}'에 sample이 없습니다.")
            st.stop()

        sample = st.selectbox("Sample", samples)

        st.markdown("---")
        st.info(
            f"**Path:** `{config}/{task}/{demo}/{sample}`\n\n"
            f"총 {len(configs)} configs · {len(tasks)} tasks · {len(demos)} demos · {len(samples)} samples"
        )

    # ─── 파일 경로 ──────────────────────────────────────────────────────
    sample_dir = CAPTION_DATA_DIR / config / task / demo / sample
    image_path = sample_dir / "image.png"
    prompt_path = sample_dir / "prompt.txt"
    caption_path = sample_dir / "caption.txt"

    # ─── 1. Image (left) + Prompt [Information] (right) ────────────────
    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.markdown("### 🖼️ Image")
        if image_path.exists():
            img = Image.open(image_path)
            st.image(img, use_container_width=True)
        else:
            st.warning("image.png 파일이 없습니다.")

    with col_info:
        st.markdown("### � Prompt — [Information]")
        if prompt_path.exists():
            info_block = extract_information_block(prompt_path)
            st.code(info_block, language=None)
        else:
            st.warning("prompt.txt 파일이 없습니다.")

    st.markdown("---")

    # ─── 2. Caption (full width) ─────────────────────────────────────────
    st.markdown("### � Caption")
    if caption_path.exists():
        caption = read_caption(caption_path)
        st.success(caption)
    else:
        st.warning("caption.txt 파일이 없습니다.")


if __name__ == "__main__":
    main()
    # streamlit run robot_action_captioning/visualization/streamlit_app.py