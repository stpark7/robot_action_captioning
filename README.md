## Installation

conda create -c conda-forge -n robocasa python=3.10

conda activate robocasa

git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
pip install -e .

cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .
pip install pre-commit; pre-commit install           # Optional: set up code formatter.

(optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)

python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
python robocasa/scripts/setup_macros.py              # Set up system variables.

1. git clone https://github.com/stpark7/robot_action_captioning.git


