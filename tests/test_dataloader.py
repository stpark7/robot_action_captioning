"""
DataLoader 테스트 - 3가지 DataConfig 시나리오 검증
"""
import pytest
from robot_action_captioning.datasets.dataloader import DataLoader
from robot_action_captioning.datasets.dataconfig import DataConfig, TimeOffset
from robot_action_captioning.utils.utils import get_hdf5_files


class TestDataLoader:
    """DataLoader 테스트 - singleframe, 3프레임, negative offset"""

    @pytest.fixture
    def hdf5_info(self):
        """테스트용 HDF5 파일 경로와 demo_id 반환"""
        import h5py
        
        data_path = "/home/park/robocasa/datasets"
        try:
            hdf5_files = get_hdf5_files(data_path)
        except (FileNotFoundError, NotADirectoryError):
            pytest.skip(f"Data directory not found: {data_path}")
        if not hdf5_files:
            pytest.skip("No HDF5 files found")
        
        hdf5_path = hdf5_files[0]
        
        with h5py.File(hdf5_path, "r") as f:
            demo_ids = sorted(list(f["data"].keys()))
            if not demo_ids:
                pytest.skip("No demos found in HDF5 file")
            demo_id = demo_ids[0]
        
        return hdf5_path, demo_id
    
    @pytest.fixture
    def config_singleframe(self):
        """단일 프레임 (offset=0)"""
        return DataConfig(
            time_offsets=[TimeOffset(offset=0)]
        )
    
    @pytest.fixture
    def config_three_frames(self):
        """3개 프레임 (offset=0, 15, 30)"""
        return DataConfig(
            time_offsets=[
                TimeOffset(offset=0),
                TimeOffset(offset=15),
                TimeOffset(offset=30),
            ]
        )
    
    @pytest.fixture
    def config_negative_offset(self):
        """Negative offset 포함 (offset=-10, 0, 10)"""
        return DataConfig(
            time_offsets=[
                TimeOffset(offset=-10),
                TimeOffset(offset=0),
                TimeOffset(offset=10),
            ]
        )

    def test_singleframe(self, hdf5_info, config_singleframe):
        """단일 프레임 DataConfig 테스트"""
        hdf5_path, demo_id = hdf5_info
        loader = DataLoader(hdf5_path, demo_id, config_singleframe)
        
        # 생성 확인
        assert loader is not None
        assert len(loader) > 0
        
        # 유효 인덱스 범위: start=0, end=total_frames
        start, end = loader.get_valid_index_range()
        assert start == 0
        assert end == loader._total_frames
        
        # 스냅샷 구조 확인
        sample = next(iter(loader))
        assert sample.episode is not None
        assert sample.environment is not None
        assert len(sample.frames) == 1
        assert sample.frames[0].offset == 0

    def test_three_frames(self, hdf5_info, config_three_frames):
        """3개 프레임 DataConfig 테스트"""
        hdf5_path, demo_id = hdf5_info
        loader = DataLoader(hdf5_path, demo_id, config_three_frames)
        
        # 생성 확인
        assert loader is not None
        assert len(loader) > 0
        
        # 유효 인덱스 범위: start=0, end=total_frames-30
        start, end = loader.get_valid_index_range()
        assert start == 0
        assert end == loader._total_frames - 30
        
        # 스냅샷 구조 확인
        sample = next(iter(loader))
        assert len(sample.frames) == 3
        offsets = [f.offset for f in sample.frames]
        assert offsets == [0, 15, 30]

    def test_negative_offset(self, hdf5_info, config_negative_offset):
        """Negative offset DataConfig 테스트"""
        hdf5_path, demo_id = hdf5_info
        loader = DataLoader(hdf5_path, demo_id, config_negative_offset)
        
        # 생성 확인
        assert loader is not None
        assert len(loader) > 0
        
        # 유효 인덱스 범위: start=10 (|-10|), end=total_frames-10
        start, end = loader.get_valid_index_range()
        assert start == 10
        assert end == loader._total_frames - 10
        
        # 스냅샷 구조 확인
        sample = next(iter(loader))
        assert len(sample.frames) == 3
        offsets = [f.offset for f in sample.frames]
        assert offsets == [-10, 0, 10]