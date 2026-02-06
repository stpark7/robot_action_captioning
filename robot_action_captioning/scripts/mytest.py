import h5py
import json
import argparse
import os
import numpy as np

class HDF5DataInspector:
    """
    A class to inspect HDF5 files for Robot Action Captioning.
    It checks fields for Environment Data, Episode Data, and Robot Data.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
    def inspect(self, episode_id=None):
        """
        Main method to inspect the HDF5 file structure and content types.
        """
        if not os.path.exists(self.file_path):
            print(f"Error: File not found at {self.file_path}")
            return

        try:
            with h5py.File(self.file_path, 'r') as f:
                print(f"=== Inspecting HDF5 File: {os.path.basename(self.file_path)} ===\n")
                
                # 1. Inspect Environment Data
                self._inspect_environment(f)
                print("-" * 50)
                
                # 2. Select an Episode to Inspect
                demos = list(f['data'].keys())
                if not demos:
                    print("No demos (episodes) found in 'data' group.")
                    return
                
                selected_demo = demos[0]
                if episode_id is not None:
                    demo_name = f"demo_{episode_id}"
                    if demo_name in demos:
                        selected_demo = demo_name
                    else:
                        print(f"Warning: Episode {episode_id} not found. Using {selected_demo} instead.\n")
                
                print(f"\n[Inspecting Episode: {selected_demo}]")
                demo_group = f['data'][selected_demo]
                
                # 3. Inspect Episode Data
                self._inspect_episode(demo_group)
                print("-" * 50)
                
                # 4. Inspect Robot Data
                self._inspect_robot(demo_group)
                print("=" * 50)
                
        except Exception as e:
            print(f"An error occurred while inspecting the file: {e}")

    def _inspect_environment(self, f):
        print("\n[1. Environment Data]")
        print("Description: Contains global configuration for the environment.")
        
        if 'data' in f and 'env_args' in f['data'].attrs:
            env_args_str = f['data'].attrs['env_args']
            print(f"  Source: f['data'].attrs['env_args'] (HDF5 Attribute)")
            print(f"  Raw Type: JSON String")
            
            try:
                env_args = json.loads(env_args_str)
                print(f"  Parsed Structure (dict):")
                for key, value in env_args.items():
                    print(f"    - {key}: {type(value).__name__} (Value: {value})")
                    if key == 'env_name':
                        print(f"      -> Corresponds to 'EnvironmentData.env_name'")
            except json.JSONDecodeError:
                print("    Error: Could not decode JSON string.")
        else:
            print("  Missing 'data' group or 'env_args' attribute.")

    def _inspect_episode(self, demo_group):
        print("\n[2. Episode Data]")
        print("Description: Contains metadata specific to a single episode (demo).")
        
        if 'ep_meta' in demo_group.attrs:
            ep_meta_str = demo_group.attrs['ep_meta']
            print(f"  Source: demo_group.attrs['ep_meta'] (HDF5 Attribute)")
            print(f"  Raw Type: JSON String")
            
            try:
                ep_meta = json.loads(ep_meta_str)
                print(f"  Parsed Structure (dict):")
                for key, value in ep_meta.items():
                    if isinstance(value, list):
                        print(f"    - {key}: List[{type(value[0]).__name__ if value else 'Any'}] (Length: {len(value)})")
                        if key == 'object_cfgs':
                            print(f"      -> Maps to 'EnvironmentData.objects_info'")
                    else:
                        print(f"    - {key}: {type(value).__name__} (Value: {value})")
                        if key == 'lang':
                             print(f"      -> Corresponds to 'EpisodeData.lang'")
            except json.JSONDecodeError:
                print("    Error: Could not decode JSON string.")
        else:
            print("  Missing 'ep_meta' attribute.")

    def _inspect_robot(self, demo_group):
        print("\n[3. Robot Data]")
        print("Description: Contains time-series data of robot states and observations.")
        
        if 'obs' in demo_group:
            obs_group = demo_group['obs']
            print(f"  Source: demo_group['obs'] (HDF5 Group)")
            print(f"  Fields (Datasets found):")
            
            key_width = 30
            print(f"    {'Field Name':<{key_width}} | {'Shape':<15} | {'Dtype':<10} | {'Description'}")
            print(f"    {'-'*key_width}-+-{'-'*15}-+-{'-'*10}-+-{'-'*20}")
            
            for key in sorted(obs_group.keys()):
                item = obs_group[key]
                if isinstance(item, h5py.Dataset):
                    desc = ""
                    if "image" in key:
                        desc = "Visual observation (Image)"
                    elif "pos" in key:
                        desc = "Position vector (XYZ)"
                    elif "quat" in key:
                        desc = "Orientation quaternion (XYZW)"
                    elif "qpos" in key:
                        desc = "Joint positions"
                    else:
                        desc = "Robot/Sensor Data"
                        
                    print(f"    {key:<{key_width}} | {str(item.shape):<15} | {str(item.dtype):<10} | {desc}")
        else:
            print("  Missing 'obs' group.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDF5 Data Inspector for Robot Action Captioning")
    parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument("--episode_id", type=int, default=None, help="Specific episode ID to inspect (optional)")
    # "home\park\robocasa\datasets\v0.1\single_stage\kitchen_pnp\PnPCounterToCab\2024-04-24\demo_gentex_im128_randcams.hdf5"
    args = parser.parse_args()
    
    inspector = HDF5DataInspector(args.file)
    inspector.inspect(episode_id=args.episode_id)
