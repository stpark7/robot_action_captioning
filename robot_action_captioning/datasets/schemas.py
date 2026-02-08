from typing import List, Dict, Any, Optional
import json
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
        data['robot'] = env_kwargs.get('robot')
        data['camera_names'] = env_kwargs.get('camera_names')
        data['camera_height'] = env_kwargs.get('camera_height')
        data['camera_width'] = env_kwargs.get('camera_width')
        
        return cls(**data)

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
    
    # Sensor/Other
    object: Optional[np.ndarray] = None

    @classmethod
    def load_from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(**data)

    # @classmethod
    # def validate_numpy_array(cls, v: Any) -> Any:
    #     if v is None:
    #         return None
    #     if isinstance(v, list):
    #         return np.array(v)
    #     return v