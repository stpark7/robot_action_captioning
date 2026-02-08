import os
from pathlib import Path
from typing import List

def get_hdf5_files(path: str) -> List[str]:
    """
    주어진 경로에서 모든 HDF5 파일을 재귀적으로 찾아 리스트로 반환합니다.
    
    Args:
        path: HDF5 파일을 검색할 디렉토리 경로
        
    Returns:
        HDF5 파일 경로들의 리스트
    """
    hdf5_files = []
    root_path = Path(path)
    
    if not root_path.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path}")
    
    if not root_path.is_dir():
        raise NotADirectoryError(f"디렉토리가 아닙니다: {path}")
    
    # 재귀적으로 모든 .hdf5 및 .h5 파일 검색
    for file_path in root_path.rglob("*.hdf5"):
        hdf5_files.append(str(file_path))
    
    return sorted(hdf5_files)
