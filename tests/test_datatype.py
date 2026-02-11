import h5py
import json # Also needed for data extraction later
import numpy as np
from pathlib import Path
import pytest

from robot_action_captioning.utils.utils import get_hdf5_files, get_demo_ids
from robot_action_captioning.datasets.datatype import EnvironmentData, EpisodeData, RobotData

HDF5_FILE_PATH = Path("~/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5").expanduser()

@pytest.fixture(scope="module")
def hdf5_data():
    """Get HDF5 file data from the configured path."""
    with h5py.File(HDF5_FILE_PATH, 'r') as f:
        yield f['data']

def test_verify_environment_data(hdf5_data):
    """Verify EnvironmentData matches raw HDF5 data."""
    print(f"Verifying EnvironmentData with {HDF5_FILE_PATH}")
    
    # Load via class method
    env_data = EnvironmentData.load_from_hdf5(HDF5_FILE_PATH)
    
    # Load raw data
    raw_env_args = json.loads(hdf5_data.attrs['env_args'])
    
    # Verify fields
    assert env_data.env_name == raw_env_args['env_name']
    
    raw_kwargs = raw_env_args.get('env_kwargs', {})
    assert env_data.robot == raw_kwargs.get('robots')
    assert env_data.camera_names == raw_kwargs.get('camera_names')
    assert env_data.camera_height == raw_kwargs.get('camera_heights')
    assert env_data.camera_width == raw_kwargs.get('camera_widths')


def test_verify_episode_data(hdf5_data):
    """Verify EpisodeData matches raw HDF5 data."""
    print(f"Verifying EpisodeData with {HDF5_FILE_PATH}")
    
    demo_ids = sorted([key for key in hdf5_data.keys() if key.startswith("demo_")])
    assert len(demo_ids) > 0, "No demos found"
    
    # Test with the first demo
    demo_id = demo_ids[0]
    
    # Load via class method
    episode = EpisodeData.load_from_hdf5(HDF5_FILE_PATH, demo_id)
    
    # Load raw data
    ep_grp = hdf5_data[demo_id]
    raw_ep_meta = json.loads(ep_grp.attrs['ep_meta'])
    
    # Verify fields
    assert episode.episode_id == demo_id
    assert episode.lang == raw_ep_meta['lang']
    
    # Verify Objects Info
    raw_objects = raw_ep_meta.get('object_cfgs', [])
    assert len(episode.objects_info) == len(raw_objects)
    
    for i, obj_info in enumerate(episode.objects_info):
        raw_obj = raw_objects[i]
        
        # Verify name
        assert obj_info.name == raw_obj['name']
        
        # Verify category (mapped from info['cat'])
        raw_cat = raw_obj['info']['cat']
        assert obj_info.category == raw_cat
        
        print(f"Verified Object: {obj_info.name}, Category: {obj_info.category}")


def test_verify_robot_data(hdf5_data):
    """Verify RobotData matches raw HDF5 data."""
    print(f"Verifying RobotData with {HDF5_FILE_PATH}")
    
    demo_ids = sorted([key for key in hdf5_data.keys() if key.startswith("demo_")])
    if not demo_ids:
        return
        
    demo_id = demo_ids[0]
    
    # Load via class method
    robot_data = RobotData.load_from_hdf5(str(HDF5_FILE_PATH), demo_id)
    
    # Load raw data
    obs_grp = hdf5_data[demo_id]['obs']
    
    # Verify shapes and values
    np.testing.assert_array_equal(robot_data.robot0_base_pos, obs_grp['robot0_base_pos'][:])
    np.testing.assert_array_equal(robot_data.robot0_base_quat, obs_grp['robot0_base_quat'][:])
    np.testing.assert_array_equal(robot_data.robot0_base_to_eef_pos, obs_grp['robot0_base_to_eef_pos'][:])
    np.testing.assert_array_equal(robot_data.robot0_base_to_eef_quat, obs_grp['robot0_base_to_eef_quat'][:])
    np.testing.assert_array_equal(robot_data.robot0_gripper_qpos, obs_grp['robot0_gripper_qpos'][:])
    
    print(f"Verified RobotData shapes: {robot_data.robot0_base_pos.shape}")


if __name__ == "__main__":
    print("Running tests manually...")
    with h5py.File(HDF5_FILE_PATH, 'r') as f:
        hdf5_data = f['data']
        
        try:
            test_verify_environment_data(hdf5_data)
            print("test_verify_environment_data PASSED")
        except Exception as e:
            print(f"test_verify_environment_data FAILED: {e}")
            import traceback
            traceback.print_exc()

        try:
            test_verify_episode_data(hdf5_data)
            print("test_verify_episode_data PASSED")
        except Exception as e:
            print(f"test_verify_episode_data FAILED: {e}")
            import traceback
            traceback.print_exc()

        try:
            test_verify_robot_data(hdf5_data)
            print("test_verify_robot_data PASSED")
        except Exception as e:
            print(f"test_verify_robot_data FAILED: {e}")
            import traceback
            traceback.print_exc()


