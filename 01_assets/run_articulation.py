import argparse
# 명령줄 인수를 처리함
from isaaclab.app import AppLauncher
# isaac sim 에플리케이션 실행 관련 모듈

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# ArgumentParser객체를 생성함
AppLauncher.add_app_launcher_args(parser)
# isaac sim 관련 객체에 명령줄 인수를 추가함
args_cli = parser.parse_args()
# 명령줄 인수를 pytion형식으로 변환
app_launcher = AppLauncher(args_cli)
# isaac sim 애플리케이션 실행 설정
simulation_app = app_launcher.app
# 실행 

import torch
# GPU가속
import isaacsim.core.utils.prims as prim_utils
# Stage(Omniverse의 3D 장면)**에서 **Prim(기본 객체)**을 생성하거나 조작할 때 사용하는 유틸리티 모듈을 불러오는 것입니다.
# 사각형, 원 같은 도형을 직접 만드는 기능뿐 아니라, USD에서 "Prim"이라는 기본 객체(메시, 카메라, 라이트 등)를 생성 및 관리하는 데 쓰이는 범용 함수들을 포함
# Xform Prim → 변환(Translation/Rotation/Scale) 노드
# Mesh Prim → 삼각형 메쉬(예: 원, 사각형, 복잡한 형상)
# Light Prim → 조명
# Camera Prim → 카메라
import isaaclab.sim as sim_utils
# 물리 엔진, 카메라등 설정함
from isaaclab.assets import Articulation
# 이미 만들어진것을 불러옴
from isaaclab.sim import SimulationContext
# 시뮬레이션 실행의 중심 컨트롤러(컨텍스트) 역할을 하며, 물리 엔진과 렌더링 환경을 초기화/제어합니다.
# 시뮬레이션 시간 간격 설정 (time step)
# 카메라 시점 설정
# 환경 초기화 및 리셋
from isaaclab_assets import CARTPOLE_CFG
# isaaclab_assets 패키지에서 CARTPOLE_CFG를 불러옵니다.
# CARTPOLE_CFG는 카트폴(CartPole) 환경에 대한 설정(Config) 객체입니다.
# 🔑 CARTPOLE_CFG의 구성 요소:
# USD 모델 경로: 카트폴 로봇의 USD 파일 위치.
# 물리 속성: 중력, 마찰력, 강체/관절 설정 등.
# 초기 상태: 카트 위치, 폴의 각도 초기값.
# 제어 설정: 카트 이동 속도 제어, 폴의 각도 안정화 등.

def design_scene() -> tuple[dict, list[list[float]]]:
    # dict: 생성된 시뮬레이션 객체(여기서는 cartpole 로봇)
    # list[list[float]]: 각 환경의 origin 위치 좌표 리스트
    cfg = sim_utils.GroundPlaneCfg()
    # 지면을 생성
    cfg.func("/World/defaultGroundPlane", cfg)
    # 생성된 지면을 배치함
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    # 빛을 생성
    cfg.func("/World/Light", cfg)
    # 생성된 빛을 배치함

    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # origins: 각 환경(병렬 환경 포함)의 기준 위치. 
    # origins는 **각 로봇 또는 환경(Env)의 기준 위치(Reference Position)**를 나타내는 좌표 리스트입니다.
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # /World/Origin1 → 첫 번째 환경의 기준점.
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    # /World/Origin2 → 두 번째 환경의 기준점.

    cartpole_cfg = CARTPOLE_CFG.copy()
    # 자산에서 기존 카트플 설정을 복사함
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    # 정규표현식(Origin.*)을 사용 → Origin1, Origin2에 각각 카트폴 로봇 생성.
    cartpole = Articulation(cfg=cartpole_cfg)
    # 다관절 로봇(articulated robot) 클래스를 생성.
    # 카트폴은 cart(base) + pole(joint)로 구성된 다관절 시스템.
    scene_entities = {"cartpole": cartpole}
    # scene_entities: 시뮬레이션에서 관리할 엔티티(여기서는 카트폴 로봇).
    # origins: 환경 배치를 위한 위치 정보.
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
# sim: Isaac Lab의 물리 시뮬레이션 엔진 컨텍스트(SimulationContext) 객체.
# entities: {"cartpole": Articulation 객체} 형태의 로봇 엔티티 딕셔너리.
# origins: 각 환경의 기준 좌표(torch.Tensor 형태).
    robot = entities["cartpole"]
    # robot: 카트폴 로봇을 변수에 할당.
    sim_dt = sim.get_physics_dt()
    # sim_dt: 시뮬레이션 한 스텝(프레임)당 시간 간격 Δt.
    count = 0
    # count: 시뮬레이션 루프 카운터.
    while simulation_app.is_running():
    # 시뮬레이션 애플리케이션이 실행 중일 때 반복.
        if count % 500 == 0:
        # 500프레임마다 로봇 상태를 초기화.
            count = 0
            root_state = robot.data.default_root_state.clone()
            # 각 로봇 인스턴스의 초기 루트 상태 ([pos(3), quat(4), lin_vel(3), ang_vel(3)]).
            root_state[:, :3] += origins
            # root_state[:, :3] += origins: 각 병렬 환경의 origin 좌표만큼 이동시켜 환경 간 위치 분리.
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # 포즈(위치+회전)와 속도(선속도+각속도)를 시뮬레이터에 적용.
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # 각 관절 초기 위치(joint_pos)와 속도를 복사.
            robot.reset()
            print("[INFO]: Resetting robot state...")
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # torch.randn_like → 평균 0, 표준편차 1의 랜덤 값을 생성.
        robot.set_joint_effort_target(efforts)
        # 각 관절에 ±5.0의 랜덤 토크를 인가 → 카트폴을 무작위로 움직이게 함.
        robot.write_data_to_sim()
        # write_data_to_sim(): 제어 명령과 상태를 물리 시뮬레이터에 반영.
        sim.step()
        # sim.step(): 물리 시뮬레이션 1 스텝 진행.
        count += 1
        robot.update(sim_dt)
        # robot.update(sim_dt): 로봇 내부 데이터(센서, 상태 버퍼) 갱신.

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
