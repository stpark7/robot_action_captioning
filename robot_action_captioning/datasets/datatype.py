from typing import List, Dict, Any, Optional
import json
import h5py
from pydantic import BaseModel, ConfigDict, field_validator
import numpy as np


class FrameData(BaseModel):
    """단일 시점(time offset)의 데이터."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    offset: int
    images: Optional[Dict[str, np.ndarray]] = None
    robot_state: Optional[Dict[str, np.ndarray]] = None
    action: Optional[Dict[str, np.ndarray]] = None


class ObjectInfo(BaseModel):
    name: str
    object_type: Optional[str] = None # e.g. "type": "microwave"
    category: Optional[str] = None
    fixture: Optional[str] = None # placement fixture

    @classmethod
    def _load_from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

class EnvironmentData(BaseModel):
    env_name: str # ex. PnPCountertoCab
    robot: str 
    camera_names: List[str]
    camera_height: int 
    camera_width: int 

    @classmethod
    def _load_from_json(cls, json_str: str):
        data = json.loads(json_str)
        env_kwargs = data.get('env_kwargs', {})
        return cls(
            env_name=data['env_name'],
            robot=env_kwargs.get('robots', [None])[0],
            camera_names=env_kwargs.get('camera_names', []),
            camera_height=env_kwargs.get('camera_heights', 0),
            camera_width=env_kwargs.get('camera_widths', 0),
        )

    @classmethod
    def load_from_hdf5(cls, hdf5_file_path: str):
        """Load EnvironmentData from HDF5 file path."""
        with h5py.File(hdf5_file_path, 'r') as f:
            env_args_str = f['data'].attrs.get('env_args')
            if env_args_str is None:
                raise ValueError(f"No 'env_args' attribute found in {hdf5_file_path}")
            return cls._load_from_json(env_args_str)

class EpisodeData(BaseModel):
    episode_id: str
    lang: str
    objects_info: List[ObjectInfo]

    @classmethod
    def _load_from_json(cls, json_str: str):
        data = json.loads(json_str)
        # Map object_cfgs to objects_info
        if 'object_cfgs' in data:
            data['objects_info'] = data['object_cfgs']

        return cls(**data)

    @classmethod
    def load_from_hdf5(cls, hdf5_file_path: str, episode_id: str = "demo_0"):
        """Load EpisodeData from HDF5 file path.
        
        Args:
            hdf5_file_path: Path to the HDF5 file.
            episode_id: Episode ID to load (default: "demo_0").
        """
        with h5py.File(hdf5_file_path, 'r') as f:
            if episode_id not in f['data']:
                raise ValueError(f"Episode '{episode_id}' not found in {hdf5_file_path}")

            ep_grp = f['data'][episode_id]
            ep_meta_str = ep_grp.attrs.get('ep_meta')
            if ep_meta_str is None:
                raise ValueError(f"No 'ep_meta' attribute found for {episode_id}")

            ep_meta = json.loads(ep_meta_str)
            ep_meta['episode_id'] = episode_id

            return cls._load_from_json(json.dumps(ep_meta))


class Sample(BaseModel):
    """DataLoader가 yield하는 하나의 샘플."""
    episode: Optional[EpisodeData] = None
    environment: Optional[EnvironmentData] = None
    frames: List[FrameData] = []


class RobotData(BaseModel):
    """Robot states and observations for an entire episode.
    
    All arrays have shape (num_frames, ...) where the first dimension is time.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Robot States & Observations (shape: [num_frames, ...])
    robot0_base_pos: np.ndarray
    robot0_base_quat: np.ndarray
    robot0_base_to_eef_pos: np.ndarray
    robot0_base_to_eef_quat: np.ndarray
    robot0_eef_pos: np.ndarray
    robot0_eef_quat: np.ndarray
    robot0_gripper_qpos: np.ndarray
    robot0_gripper_qvel: np.ndarray
    robot0_joint_pos: np.ndarray
    robot0_joint_vel: np.ndarray
    robot0_joint_pos_cos: np.ndarray
    robot0_joint_pos_sin: np.ndarray

    # Images (shape: [num_frames, H, W, C])
    robot0_agentview_left_image: Optional[np.ndarray] = None
    robot0_agentview_right_image: Optional[np.ndarray] = None
    robot0_eye_in_hand_image: Optional[np.ndarray] = None

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the episode."""
        return len(self.robot0_base_pos)

    @classmethod
    def load_from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def load_from_hdf5(cls, hdf5_file_path: str, episode_id: str = "demo_0"):
        """Load RobotData for entire episode from HDF5 file path.
        
        Args:
            hdf5_file_path: Path to the HDF5 file.
            episode_id: Episode ID to load (default: "demo_0").
        
        Returns:
            RobotData with all frames. Arrays have shape (num_frames, ...).
        """
        with h5py.File(hdf5_file_path, 'r') as f:
            if episode_id not in f['data']:
                raise ValueError(f"Episode '{episode_id}' not found in {hdf5_file_path}")

            obs_grp = f['data'][episode_id]['obs']

            data = {
                # Robot States & Observations - load all frames
                'robot0_base_pos': obs_grp['robot0_base_pos'][:],
                'robot0_base_quat': obs_grp['robot0_base_quat'][:],
                'robot0_base_to_eef_pos': obs_grp['robot0_base_to_eef_pos'][:],
                'robot0_base_to_eef_quat': obs_grp['robot0_base_to_eef_quat'][:],
                # 'robot0_eef_pos': obs_grp['robot0_eef_pos'][:],
                # 'robot0_eef_quat': obs_grp['robot0_eef_quat'][:],
                'robot0_gripper_qpos': obs_grp['robot0_gripper_qpos'][:],
                # 'robot0_gripper_qvel': obs_grp['robot0_gripper_qvel'][:],
                # 'robot0_joint_pos': obs_grp['robot0_joint_pos'][:],
                # 'robot0_joint_vel': obs_grp['robot0_joint_vel'][:],
                # 'robot0_joint_pos_cos': obs_grp['robot0_joint_pos_cos'][:],
                # 'robot0_joint_pos_sin': obs_grp['robot0_joint_pos_sin'][:],
            }

            # Images (optional) - load all frames
            if 'robot0_agentview_left_image' in obs_grp:
                data['robot0_agentview_left_image'] = obs_grp['robot0_agentview_left_image'][:]
            if 'robot0_agentview_right_image' in obs_grp:
                data['robot0_agentview_right_image'] = obs_grp['robot0_agentview_right_image'][:]
            if 'robot0_eye_in_hand_image' in obs_grp:
                data['robot0_eye_in_hand_image'] = obs_grp['robot0_eye_in_hand_image'][:]

            return cls(**data)
