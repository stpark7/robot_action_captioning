from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from .schemas import EpisodeData, EnvironmentData, RobotData

'''
schemas.py에서 정의된 데이터 중 어떤 데이터를 불러올지 정의하는 클래스입니다.
'''

class Type1(BaseModel):
    """
    Example configuration for robot action captioning.
    Includes episode info, environment info, and robot states at t and t+H (conceptually, or full trajectory).
    Here we define what we WANT to extract.
    """
    episode: EpisodeData
    environment: EnvironmentData
    
    current_robot: RobotData # State at time t
    next_robot: RobotData    # State at time t+H

class Type2(BaseModel):
    pass

class Type3(BaseModel):
    pass
