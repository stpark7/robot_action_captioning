from typing import List, Dict, Any, Optional
import json
import h5py
from pydantic import BaseModel, ConfigDict, field_validator
import numpy as np

class ObjectInfo(BaseModel):
    name: str
    type: str # e.g. "type": "microwave"
    category: Optional[str] = None
    fixture: Optional[str] = None # placement fixture

    @classmethod
    def load_from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

class EnvironmentData(BaseModel):
    env_name: str # ex. PnPCountertoCab
    robot: str
    camera_names: List[str]
    camera_height: int
    camera_width: int

    @classmethod
    def load_from_json(cls, json_str: str):
        data = json.loads(json_str)
        env_kwargs = data.get('env_kwargs', {})

        # Flatten env_kwargs into the main dict for initialization
        # Handle both 'robot' and 'robots' (HDF5 uses 'robots')
        robot = env_kwargs.get('robot') or env_kwargs.get('robots')
        data['robot'] = robot[0] if isinstance(robot, list) else robot

        data['camera_names'] = env_kwargs.get('camera_names')

        # Handle singular and plural (HDF5 uses 'camera_heights'/'camera_widths')
        camera_height = env_kwargs.get('camera_height') or env_kwargs.get('camera_heights')
        camera_width = env_kwargs.get('camera_width') or env_kwargs.get('camera_widths')
        data['camera_height'] = camera_height[0] if isinstance(camera_height, list) else camera_height
        data['camera_width'] = camera_width[0] if isinstance(camera_width, list) else camera_width

        return cls(**data)

    @classmethod
    def load_from_hdf5(cls, hdf5_file_path: str):
        """Load EnvironmentData from HDF5 file path."""
        with h5py.File(hdf5_file_path, 'r') as f:
            env_args_str = f['data'].attrs.get('env_args')
            if env_args_str is None:
                raise ValueError(f"No 'env_args' attribute found in {hdf5_file_path}")
            return cls.load_from_json(env_args_str)

class EpisodeData(BaseModel):
    episode_id: str
    lang: str
    objects_info: List[ObjectInfo]

    @classmethod
    def load_from_json(cls, json_str: str):
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

            return cls.load_from_json(json.dumps(ep_meta))

class RobotData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Robot States & Observations
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

    # Images
    robot0_agentview_left_image: Optional[np.ndarray] = None
    robot0_agentview_right_image: Optional[np.ndarray] = None
    robot0_eye_in_hand_image: Optional[np.ndarray] = None

    @classmethod
    def load_from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def load_from_hdf5(cls, hdf5_file_path: str, episode_id: str = "demo_0", frame_idx: int = 0):
        """Load RobotData from HDF5 file path.
        
        Args:
            hdf5_file_path: Path to the HDF5 file.
            episode_id: Episode ID to load (default: "demo_0").
            frame_idx: Frame index to load (default: 0).
        """
        with h5py.File(hdf5_file_path, 'r') as f:
            if episode_id not in f['data']:
                raise ValueError(f"Episode '{episode_id}' not found in {hdf5_file_path}")

            obs_grp = f['data'][episode_id]['obs']

            data = {
                # Robot States & Observations
                'robot0_base_pos': obs_grp['robot0_base_pos'][frame_idx],
                'robot0_base_quat': obs_grp['robot0_base_quat'][frame_idx],
                'robot0_base_to_eef_pos': obs_grp['robot0_base_to_eef_pos'][frame_idx],
                'robot0_base_to_eef_quat': obs_grp['robot0_base_to_eef_quat'][frame_idx],
                'robot0_eef_pos': obs_grp['robot0_eef_pos'][frame_idx],
                'robot0_eef_quat': obs_grp['robot0_eef_quat'][frame_idx],
                'robot0_gripper_qpos': obs_grp['robot0_gripper_qpos'][frame_idx],
                'robot0_gripper_qvel': obs_grp['robot0_gripper_qvel'][frame_idx],
                'robot0_joint_pos': obs_grp['robot0_joint_pos'][frame_idx],
                'robot0_joint_vel': obs_grp['robot0_joint_vel'][frame_idx],
                'robot0_joint_pos_cos': obs_grp['robot0_joint_pos_cos'][frame_idx],
                'robot0_joint_pos_sin': obs_grp['robot0_joint_pos_sin'][frame_idx],
            }

            # Images (optional)
            if 'robot0_agentview_left_image' in obs_grp:
                data['robot0_agentview_left_image'] = obs_grp['robot0_agentview_left_image'][frame_idx]
            if 'robot0_agentview_right_image' in obs_grp:
                data['robot0_agentview_right_image'] = obs_grp['robot0_agentview_right_image'][frame_idx]
            if 'robot0_eye_in_hand_image' in obs_grp:
                data['robot0_eye_in_hand_image'] = obs_grp['robot0_eye_in_hand_image'][frame_idx]

            return cls(**data)

    # @classmethod
    # def validate_numpy_array(cls, v: Any) -> Any:
    #     if v is None:
    #         return None
    #     if isinstance(v, list):
    #         return np.array(v)
    #     return v