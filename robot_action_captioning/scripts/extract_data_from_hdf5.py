import argparse
import sys
import os

# Add parent directory to path to allow imports if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now we can import from the package
from datasets.dataloader import DataLoader
from datasets.datatypes import CaptioningDataType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--episode_id",
        type=int,
        default=None,
        help="ID of the episode to extract",
    )
    args = parser.parse_args()

    try:
        # Initialize DataLoader
        loader = DataLoader(dataset_path=args.dataset, horizon=30)
        
        # Load Data using the CaptioningDataType definition
        print(f"Loading data from {args.dataset}...")
        data = loader.load(episode_id=args.episode_id, data_type=CaptioningDataType)

        # Output Summary
        print(f"\n[Extraction Success]")
        print(f"Environment: {data.environment.env_name}")
        print(f"Language Instruction: {data.episode.lang}")
        print(f"Objects Identified: {len(data.environment.objects_info)}")
        
        print("\n[Robot Data]")
        print(f"Current EEF Pos (t): {data.current_robot.robot0_base_to_eef_pos}")
        print(f"Next EEF Pos (t+H): {data.next_robot.robot0_base_to_eef_pos}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

