import json
import numpy as np
import pytest
from robot_action_captioning.datasets.schemas import ObjectInfo, EnvironmentData, EpisodeData, RobotData

'''
HDF5파일에서 schemas.py의 클래스를 사용할 때 
데이터가 잘 들어가는지 테스트하는 코드입니다.
'''

def test_object_info_load_from_json():
    data = {
        "name": "plate",
        "type": "plate",
        "category": "kitchen",
        "fixture": "counter"
    }
    json_str = json.dumps(data)
    obj = ObjectInfo.load_from_json(json_str)
    assert obj.name == "plate"
    assert obj.type == "plate"
    assert obj.category == "kitchen"
    assert obj.fixture == "counter"

def test_environment_data_load_from_json():
    data = {
        "env_name": "TestEnv",
        "objects_info": [
            {"name": "obj1", "type": "type1"},
            {"name": "obj2", "type": "type2"}
        ]
    }
    json_str = json.dumps(data)
    env = EnvironmentData.load_from_json(json_str)
    assert env.env_name == "TestEnv"
    assert len(env.objects_info) == 2
    assert env.objects_info[0].name == "obj1"

def test_episode_data_load_from_json():
    data = {
        "episode_id": "ep_001",
        "lang": "Take the apple"
    }
    json_str = json.dumps(data)
    ep = EpisodeData.load_from_json(json_str)
    assert ep.episode_id == "ep_001"
    assert ep.lang == "Take the apple"

def test_robot_data_load_from_json():
    # Helper to simulate numpy array serialization (usually they are serialized as lists)
    data = {
        "robot0_base_to_eef_pos": [0.1, 0.2, 0.3],
        "robot0_base_to_eef_quat": [0.0, 0.0, 0.0, 1.0],
        "robot0_gripper_qpos": [0.05, 0.05],
        "robot0_agentview_left_image": None
    }
    json_str = json.dumps(data)
    
    # We expect pydantic to handle conversion from list to np.ndarray if configured correctly?
    # Wait, pydantic doesn't automatically convert list to numpy array unless validatory is set.
    # The existing ConfigDict(arbitrary_types_allowed=True) allows np.ndarray type, but doesn't guarantee auto conversion from list.
    # Let's see if it works or if we need a validator.
    
    try:
        robot_data = RobotData.load_from_json(json_str)
        assert isinstance(robot_data.robot0_base_to_eef_pos, np.ndarray) or isinstance(robot_data.robot0_base_to_eef_pos, list)
        # If it stays as list, we might want to check if that's acceptable. 
        # Usually for Pydantic v2 we might need a custom validator or use `pydantic.BeforeValidator` or similar if strict type is needed.
        # But let's check what happens first.
        assert np.allclose(robot_data.robot0_base_to_eef_pos, [0.1, 0.2, 0.3])
    except Exception as e:
        pytest.fail(f"Failed to load RobotData from json: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
