import h5py
import json
import random
import numpy as np
import os
from typing import Type, TypeVar, Optional, List
from pydantic import BaseModel

from .schemas import EpisodeData, EnvironmentData, RobotData
from .datatypes import Type1

T = TypeVar("T", bound=BaseDataType)

class DataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        self.environment_data = EnvironmentData().load()
        self.episode_data = EpisodeData().load()
        self.robot_data = RobotData().load()

    def load(self, episode_id: Optional[int], data_type: Type[T]) -> T:
        """
        Loads data from HDF5 and populates the provided data_type model.
        """
        with h5py.File(self.dataset_path, "r") as f:
            # 1. Environment Metadata
            if "data" not in f or "env_args" not in f["data"].attrs:
                 raise ValueError("Invalid HDF5 structure: missing env_args")
            
            env_meta = json.loads(f["data"].attrs["env_args"])
            env_name = env_meta.get("env_name")
            
            # 2. Episode Selection
            all_demos = sorted(list(f["data"].keys()))
            if episode_id is None:
                selected_demo = random.choice(all_demos)
            else:
                selected_demo = f"demo_{episode_id}"
                if selected_demo not in all_demos:
                    raise ValueError(f"Episode {selected_demo} not found.")
            
            ep_grp = f[f"data/{selected_demo}"]
            ep_meta = json.loads(ep_grp.attrs["ep_meta"])
            
            # 3. Time Sampling (t, t+H)
            if "obs" not in ep_grp:
                raise ValueError("No 'obs' group in episode.")
            
            obs_grp = ep_grp["obs"]
            num_steps = obs_grp["robot0_base_to_eef_pos"].shape[0]
            
            if num_steps <= self.horizon:
                 # Fallback or error? specific error for shortage
                 raise ValueError(f"Episode length {num_steps} <= horizon {self.horizon}")
                 
            t = random.randint(0, num_steps - self.horizon - 1)
            t_next = t + self.horizon
            
            # 4. Dynamic Object Population
            # We instantiate the requested Pydantic model by gathering necessary data
            # based on the expected fields in data_type.
            
            field_values = {}
            model_fields = data_type.model_fields
            
            for field_name, field_info in model_fields.items():
                field_type = field_info.annotation
                
                # Check if the field expects EpisodeData
                if field_type == EpisodeData:
                    field_values[field_name] = EpisodeData(
                        episode_id=selected_demo,
                        lang=ep_meta.get("lang", "")
                    )
                
                # Check if the field expects EnvironmentData
                elif field_type == EnvironmentData:
                    # Parse objects info
                    object_configs = ep_meta.get("object_cfgs", [])
                    objects_info = []
                    for cfg in object_configs:
                        objects_info.append({
                            "name": cfg.get("name"),
                            "type": cfg.get("type"),
                            "category": cfg.get("info", {}).get("cat"),
                            "fixture": cfg.get("placement", {}).get("fixture")
                        })
                    
                    field_values[field_name] = EnvironmentData(
                        env_name=env_name,
                        objects_info=objects_info
                    )
                
                # Check if the field expects RobotData
                # Here we need a convention. 
                # If field_name is 'current_robot', use index t.
                # If field_name is 'next_robot', use index t_next.
                # If we want a generic 'robot', maybe use full sequence?
                # For this refactoring, let's support specific keys.
                elif field_type == RobotData:
                    idx = t
                    if "next" in field_name:
                        idx = t_next
                    
                    # Extract necessary robot fields
                    # We assume standard keys exist or we error/fill None.
                    # Commonly used keys in this dataset:
                    
                    def get_obs(key, index):
                        if key in obs_grp:
                            return obs_grp[key][index]
                        return np.array([]) # Or None

                    r_data = RobotData(
                        robot0_base_to_eef_pos=get_obs("robot0_base_to_eef_pos", idx),
                        robot0_base_to_eef_quat=get_obs("robot0_base_to_eef_quat", idx),
                        robot0_gripper_qpos=get_obs("robot0_gripper_qpos", idx),
                        robot0_agentview_left_image=get_obs("robot0_agentview_left_image", idx),
                        robot0_agentview_right_image=get_obs("robot0_agentview_right_image", idx),
                        robot0_eye_in_hand_image=get_obs("robot0_eye_in_hand_image", idx)
                    )
                    field_values[field_name] = r_data
            
            return data_type(**field_values)
