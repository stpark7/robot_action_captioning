from typing import List, Optional
from pydantic import BaseModel, Field

# =============================================================================
# Time Offset 정의
# =============================================================================

class TimeOffset(BaseModel):

    """
    t 시점 기준 오프셋 정의
    
    Args:
        offset: t 기준 오프셋 (음수=과거, 양수=미래, 0=현재)
        include_image: 해당 시점의 이미지 포함 여부
        include_robot_state: 해당 시점의 로봇 state(관측값) 포함 여부
        include_action: 해당 시점의 action 포함 여부
    
    Examples:
        # 현재 시점, 모든 데이터 포함
        TimeOffset(offset=0)
        
        # t+30 시점, 이미지만 포함
        TimeOffset(offset=30, include_image=True, include_robot_state=False, include_action=False)
        
        # t-10 시점 (과거), 로봇 상태만
        TimeOffset(offset=-10, include_image=False, include_robot_state=True, include_action=False)
    """

    offset: int = 0
    include_image: bool = True
    include_robot_state: bool = True
    include_action: bool = True


# =============================================================================
# DataType 설정 클래스
# =============================================================================

class DataConfig(BaseModel):
    """
    어떤 데이터를 가져올지 정의하는 설정 클래스
    
    Args:
        time_offsets: 시간 오프셋 리스트 (각 시점별로 어떤 데이터를 가져올지 정의)
        image_keys: 가져올 이미지 키 리스트
        robot_state_keys: 가져올 로봇 상태 키 리스트
        action_keys: 가져올 액션 키 리스트
    
    Returns (from DataLoader):
        Sample:
            - episode: EpisodeData or None
            - environment: EnvironmentData or None
            - frames: List[FrameData]
                - offset: int
                - images: Dict[str, np.ndarray] or None
                - robot_state: Dict[str, np.ndarray] or None
                - action: Dict[str, np.ndarray] or None
    """

    # 시간 오프셋 리스트
    time_offsets: List[TimeOffset] = Field(default_factory=list)

    # 어떤 이미지를 가져올지 (카메라 선택)
    image_keys: List[str] = Field(
        default_factory=lambda: [
            "robot0_agentview_left_image",
            "robot0_agentview_right_image",
            "robot0_eye_in_hand_image",
        ]
    )

    # 어떤 로봇 상태를 가져올지
    robot_state_keys: List[str] = Field(
        default_factory=lambda: [
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_base_to_eef_pos",
            "robot0_base_to_eef_quat",
            "robot0_gripper_qpos",
            # "robot0_gripper_qvel",
            # "robot0_joint_pos",
            # "robot0_joint_vel",
            # "robot0_joint_pos_cos",
            # "robot0_joint_pos_sin",
            # "robot0_eef_pos",
            # "robot0_eef_quat",
        ]
    )

    # 어떤 액션을 가져올지
    action_keys: List[str] = Field(
        default_factory=lambda: [
            #"actions",
        ]
    )

    def get_max_offset(self) -> int:
        if not self.time_offsets:
            return 0
        return max(t.offset for t in self.time_offsets)

    def get_min_offset(self) -> int:
        if not self.time_offsets:
            return 0
        return min(t.offset for t in self.time_offsets)

    def to_folder_name(self) -> str:
        """DataConfig의 time offset 값들을 폴더명으로 변환합니다.
        
        예: offsets [0, 30, 60] -> "[0,30,60]"
        """
        offsets = [str(t.offset) for t in self.time_offsets]
        return f"[{','.join(offsets)}]"


# =============================================================================
# 프리셋 DataType들 (자주 쓰는 조합)
# =============================================================================

# # 프리셋 1: t와 t+30 시점, 모든 데이터
# TwoFrameFullData = DataConfig(
#     time_offsets=[
#         TimeOffset(offset=0),
#         TimeOffset(offset=30),
#     ]
# )

# # 프리셋 2: t, t+10, t+20, t+30 이미지 + t, t+30 로봇/액션 데이터만
# FourFrameSelectiveData = DataConfig(
#     time_offsets=[
#         TimeOffset(offset=0, include_image=True, include_robot_state=True, include_action=True),
#         TimeOffset(offset=10, include_image=True, include_robot_state=False, include_action=False),
#         TimeOffset(offset=20, include_image=True, include_robot_state=False, include_action=False),
#         TimeOffset(offset=30, include_image=True, include_robot_state=True, include_action=True),
#     ]
# )

# # 프리셋 3: 과거 + 현재 + 미래 (컨텍스트 포함)
# PastPresentFuture = DataConfig(
#     time_offsets=[
#         TimeOffset(offset=-10, include_image=True, include_robot_state=False, include_action=False),
#         TimeOffset(offset=0, include_image=True, include_robot_state=True, include_action=True),
#         TimeOffset(offset=10, include_image=True, include_robot_state=False, include_action=False),
#     ]
# )