from pathlib import Path

HDF5_PATH = Path("~/sangtae/robocasa/datasets/v0.1/single_stage").expanduser()
HDF5_DATA = Path("~/sangtae/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5").expanduser()

LLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

SAVE_DIR = Path("~/sangtae/robot_action_captioning/src/caption_data").expanduser()