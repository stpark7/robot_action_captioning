import h5py
import json
import numpy as np
import os
from typing import Optional, Iterator
import math

from .dataconfig import DataConfig
from .datatype import EpisodeData, EnvironmentData, FrameData, Sample

# TODO : hdf5파일 열고 닫는거 자주 하는거 개선
# __getitem__ 메서드를 통한 인덱스 접근?
# typehint 개선


class DataLoader:
    """
    Iterable DataLoader that loads data from HDF5 files based on DataConfig.
    
    Automatically calculates valid index range based on time offsets in DataConfig
    to prevent out-of-bounds access.
    
    Args:
        hdf5_path: Path to the HDF5 file.
        demo_id: Demo ID to load (e.g., "demo_0").
        data_config: DataConfig specifying which data to load.
    
    Usage:
        config = DataConfig(time_offsets=[TimeOffset(offset=0), TimeOffset(offset=30)])
        loader = DataLoader("data.hdf5", "demo_0", config)
        
        for sample in loader:
            # sample contains episode, environment, and frames data
            print(sample["frames"])
    """
    
    def __init__(self, hdf5_path: str, demo_id: str, data_config: DataConfig, step_size: int = 1):
        self.hdf5_path = hdf5_path
        self.demo_id = demo_id
        self.data_config = data_config
        self.step_size = step_size
        
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found at {hdf5_path}")

        self._start: int = 0
        self._end: int = 0
        self._total_frames: int = 0
        self._environment_data: Optional[EnvironmentData] = None
        self._episode_data: Optional[EpisodeData] = None

        self._load_metadata()

        self._calculate_valid_range()

    
    def _load_metadata(self) -> None:
        """Load episode and environment metadata from HDF5."""
        with h5py.File(self.hdf5_path, "r") as f:
            
            # Validate structure
            if "data" not in f:
                raise ValueError(f"Invalid HDF5 structure: missing 'data' group")
            if self.demo_id not in f["data"]:
                raise ValueError(f"Demo '{self.demo_id}' not found in HDF5 file")
            
            ep_grp = f[f"data/{self.demo_id}"]
            
            # Get total number of frames
            if "obs" not in ep_grp:
                raise ValueError(f"No 'obs' group in demo '{self.demo_id}'")
            
            obs_grp = ep_grp["obs"]
            # Use any existing key to determine frame count
            first_key = list(obs_grp.keys())[0]
            self._total_frames = obs_grp[first_key].shape[0]
            
            self._environment_data = EnvironmentData.load_from_hdf5(self.hdf5_path)
            self._episode_data = EpisodeData.load_from_hdf5(self.hdf5_path, self.demo_id)

    
    def _calculate_valid_range(self) -> None:
        """ Calculate valid base index range based on DataConfig's time offsets."""
        min_offset = self.data_config.get_min_offset()  # e.g., -10
        max_offset = self.data_config.get_max_offset()  # e.g., 30

        self._start = abs(min_offset) if min_offset < 0 else 0

        self._end = self._total_frames - max_offset
    
    def __iter__(self) -> Iterator[Sample]:
        """Yield samples using a generator."""
        for idx in range(self._start, self._end, self.step_size):
            yield self._load_sample(idx)
    
    def __len__(self) -> int:
        """Return number of valid samples."""
        if self.step_size == 1:
            return self._end - self._start
        return math.ceil((self._end - self._start) / self.step_size)
    
    def get_valid_index_range(self) -> tuple[int, int]:
        """Return the valid index range (start, end) as a tuple."""
        return (self._start, self._end)
    
    def _load_sample(self, base_idx: int) -> Sample:
        """
        Load a complete sample for the given base index.
        
        Args:
            base_idx: The base index (t) to load data around.
        
        Returns:
            Sample containing episode, environment, and frames data.
        """
        sample = Sample(
            episode=self._episode_data,
            environment=self._environment_data,
        )
        
        with h5py.File(self.hdf5_path, "r") as f:
            obs_grp = f[f"data/{self.demo_id}/obs"]
            actions = f[f"data/{self.demo_id}/actions"] if "actions" in f[f"data/{self.demo_id}"] else None
            
            for time_offset in self.data_config.time_offsets:
                frame_idx = base_idx + time_offset.offset
                
                # Setup data containers
                images = None
                robot_state = None
                action = None

                # Load images if requested
                if time_offset.include_image:
                    image_data = {}
                    for key in self.data_config.image_keys:
                        if key in obs_grp:
                            image_data[key] = obs_grp[key][frame_idx]
                    if image_data:
                        images = image_data
                
                # Load robot state if requested
                if time_offset.include_robot_state:
                    state_data = {}
                    for key in self.data_config.robot_state_keys:
                        if key in obs_grp:
                            state_data[key] = obs_grp[key][frame_idx]
                    if state_data:
                        robot_state = state_data
                
                # Load action if requested
                if time_offset.include_action and actions is not None:
                    action_data = {}
                    for key in self.data_config.action_keys:
                        if key == "actions":
                            action_data[key] = actions[frame_idx]
                    if action_data:
                        action = action_data
                
                frame_data = FrameData(
                    offset=time_offset.offset,
                    images=images,
                    robot_state=robot_state,
                    action=action
                )
                sample.frames.append(frame_data)
        
        return sample
