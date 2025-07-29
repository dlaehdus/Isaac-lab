import argparse
# 명령줄 인수를 처리함
from isaaclab.app import AppLauncher
# isaac sim 애플ㄹ리케이션을 실행할때 사용

parser = argparse.ArgumentParser(description="This script demonstrates adding a custom robot to an Isaac Lab environment.")
# Argmentparser객체를 생성함
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# Argmentparser에 뭘 추가함 
# 옵션 명 --num_envs
# 자료형: int (정수형)
# 기본값: 1.0
AppLauncher.add_app_launcher_args(parser)
# 생성된 객체에 명령줄 인수를 추가함
args_cli = parser.parse_args()
# 명령줄 인수를 pytion형식으로 변환
app_launcher = AppLauncher(args_cli)
# isaac sim 애플리케이션 실행설정
simulation_app = app_launcher.app
# 실행

import numpy as np
# 수치 연산을 위한 표준 라이브러리
import torch
# PyTorch는 딥러닝과 GPU 가속 연산을 위한 라이브러리.
import isaaclab.sim as sim_utils
# Isaac Lab의 시뮬레이션 유틸리티 모듈을 sim_utils라는 별칭으로 불러옴
# 물리 엔진 및 카메라 설정.
from isaaclab.actuators import ImplicitActuatorCfg
# 로봇의 구동기(Actuator) 설정을 위한 클래스.
# Implicit Actuator는 토크 기반 구동 모델에서 암시적 적분(implicit integration)을 사용하여 안정적인 계산을 수행.
# 예: 로봇 관절 모터 설정 시 토크 제한, 감쇠 계수, 제어 모드 등을 지정.
from isaaclab.assets import AssetBaseCfg
# 로봇, 오브젝트, 환경 요소 등 모든 시뮬레이션 개체의 공통 설정(Base Configuration)을 정의하는 클래스.
# 위치(translation), 회전(orientation), USD 경로 등의 공통 속성을 포함.
from isaaclab.assets.articulation import ArticulationCfg
# 로봇이나 다관절(articulated) 구조물의 설정 클래스.
# 예: 6자유도 로봇팔, 모바일 로봇과 같이 링크(Link)와 조인트(Joint)로 구성된 모델을 불러올 때 사용.
# 로봇 USD 파일 경로, 조인트 초기화 값, 구동기 설정과 같은 파라미터 포함.
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# InteractiveScene: 로봇과 오브젝트가 상호작용할 수 있는 시뮬레이션 장면(Scene) 클래스.
# InteractiveSceneCfg: 장면(Scene) 구성 설정.
# 환경 요소(조명, 지면, 오브젝트)
# 로봇(Articulations)
# 센서(Camera, LiDAR 등)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# Isaac Nucleus 서버 내 기본 자산 경로를 나타내는 상수.
# 모든 USD 파일은 Nucleus 서버(Omniverse Asset Server)에 저장되며, 이 상수를 기반으로 경로를 구성.


JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"),
    # Jetbot의 USD 모델을 스폰(배치)하기 위한 설정.
    # usd_path: Jetbot 로봇의 USD 파일 경로.
    # ISAAC_NUCLEUS_DIR는 Nucleus 서버 경로 상수.
    # 결과: .../Robots/Jetbot/jetbot.usd 경로의 Jetbot 모델을 불러옴.
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
    # Jetbot의 휠(바퀴) 액추에이터 설정.
    # 키 "wheel_acts": 액추에이터 그룹 이름.
    # ImplicitActuatorCfg:
    # joint_names_expr=[".*"]: 모든 조인트(.*는 정규표현식) 대상.
    # damping=None, stiffness=None: 감쇠 및 강성 제거 → 바퀴가 자유롭게 회전하도록 설정.
)

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
    # Dofbot 로봇 모델의 USD 파일을 로드.
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
        # rigid_props: 강체(Rigid Body) 물리 속성
            disable_gravity=False,
            # disable_gravity=False: 중력 활성화.
            max_depenetration_velocity=5.0,
            # max_depenetration_velocity=5.0: 충돌 후 관통 보정 속도 제한(5 m/s 이하).
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            # enabled_self_collisions=True: 자기 충돌(Self-collision) 활성화.
            # solver_position_iteration_count=8: 위치 기반 솔버 반복 횟수(정확도 향상).
            # solver_velocity_iteration_count=0: 속도 기반 솔버 반복 비활성화.
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        # 초기 조인트 위치(joint_pos):
        # 각 관절 joint1~joint4를 0 라디안으로 초기화.
        },
        pos=(0.25, -0.25, 0.0),
        # 초기 로봇 위치(pos):
        # (x=0.25, y=-0.25, z=0.0) 위치에 스폰.
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
            # "front_joints": joint1과 joint2에 적용.
            # effort_limit_sim=100.0: 토크 한계(100 N·m).
            # velocity_limit_sim=100.0: 속도 제한(100 rad/s).
            # stiffness=10000.0: 높은 강성(즉각적인 위치 제어 반응).
            # damping=100.0: 감쇠 설정(진동 억제).
        ),
        "joint3_act": ImplicitActuatorCfg(
        # "joint3_act": joint3에 개별 액추에이터 적용.
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
        # "joint4_act": joint4 전용 액추에이터.
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)


class NewRobotsSceneCfg(InteractiveSceneCfg):
# InteractiveSceneCfg를 상속받아 새로운 시뮬레이션 장면 설정 클래스를 정의.
# 이 클래스는 **지면, 조명, 로봇 등 장면을 구성하는 모든 자산(Assets)**을 정의.
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # ground: 시뮬레이션 환경의 지면을 생성하는 설정.
    # AssetBaseCfg:
    # prim_path: /World/defaultGroundPlane → USD Stage 상의 지면 프림 경로.
    # spawn: sim_utils.GroundPlaneCfg() → 기본 평면 지면 생성 유틸리티 사용.
    # 결과: 장면에 기본 평면 지면이 배치됨.
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        # dome_light: 환경광(Dome Light)을 추가.
        # prim_path: /World/Light → USD Stage 상의 조명 노드 경로.
        # spawn: sim_utils.DomeLightCfg(...)
        # intensity=3000.0 → 빛의 세기.
        # color=(0.75, 0.75, 0.75) → 중립 회색빛 환경광 설정.
    )
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    # 이전에 정의한 **JETBOT_CONFIG**를 기반으로 새로운 프림 경로를 지정.
    # replace(...):
    # 기존 JETBOT_CONFIG를 복사한 뒤 특정 속성(prim_path)만 변경.
    # prim_path="{ENV_REGEX_NS}/Jetbot":
    # **{ENV_REGEX_NS}**는 환경 인스턴스마다 고유 네임스페이스를 자동으로 부여.
    # 여러 환경(num_envs > 1)을 생성할 때 충돌 없이 Jetbot을 배치 가능.
    # 결과: 각 환경 인스턴스별 Jetbot 로봇 생성.
    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    # Jetbot과 동일한 방식으로 Dofbot 추가.
    # DOFBOT_CONFIG 기반으로 새로운 prim 경로를 지정.
    # 각 환경에서 Dofbot 로봇이 독립적으로 생성.

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
# sim: 시뮬레이션 컨텍스트 (SimulationContext) 객체. 물리 엔진 실행을 담당.
# scene: Jetbot, Dofbot 등 여러 로봇이 배치된 장면 (InteractiveScene) 객체.
    sim_dt = sim.get_physics_dt()
    # sim_dt: 물리 시뮬레이션의 시간 간격(Δt, time step)을 가져옴.
    sim_time = 0.0
    # sim_time: 누적 시뮬레이션 시간.
    count = 0
    # count: 시뮬레이션 루프 횟수 카운터.

    while simulation_app.is_running():
    # 시뮬레이션 앱이 실행 중인 동안 반복. (GUI 창 닫히면 종료됨)
        if count % 500 == 0:
        # 500프레임마다 Jetbot, Dofbot의 위치/속도/조인트 상태를 초기화.
        # 에피소드 리셋(Episode Reset) 또는 강화학습 초기화와 같은 개념.
            count = 0
            root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            # Jetbot의 초기 루트 상태(x, y, z, quaternion, lin vel, ang vel)를 복사.
            root_jetbot_state[:, :3] += scene.env_origins
            # 각 환경의 중심 위치(env_origins)에 따라 위치 오프셋 적용.
            root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
            # Dofbot도 동일하게 복제 및 위치 설정.
            root_dofbot_state[:, :3] += scene.env_origins

            scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            # pose (위치 + quaternion 회전) 7개 요소 → 시뮬레이터에 적용.
            scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
            # velocity (선속도, 각속도) → 시뮬레이터에 적용.
            scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            # Dofbot도 동일하게 적용.
            scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

            joint_pos, joint_vel = (
            # 조인트 위치 및 속도를 복사해서 시뮬레이터에 설정.
                scene["Jetbot"].data.default_joint_pos.clone(),
                scene["Jetbot"].data.default_joint_vel.clone(),
            )
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            joint_pos, joint_vel = (
            # 조인트 위치 및 속도를 복사해서 시뮬레이터에 설정.
                scene["Dofbot"].data.default_joint_pos.clone(),
                scene["Dofbot"].data.default_joint_vel.clone(),
            )
            scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            # 모든 장면 내 시뮬레이션 객체를 리셋. (내부적으로 캐시 재적용 포함)
            print("[INFO]: Resetting Jetbot and Dofbot state...")
        if count % 100 < 75:
        # 100 프레임 단위로 반복되는 제어 패턴 적용:
            action = torch.Tensor([[10.0, 10.0]])
            # 처음 75프레임 동안 양 바퀴 10 rad/s → 직진.
        else:
            action = torch.Tensor([[5.0, -5.0]])
            # 이후 25프레임은 한쪽 바퀴만 반전 → 회전 동작.
        scene["Jetbot"].set_joint_velocity_target(action)
        # Jetbot에 속도 명령(joint velocity control) 적용.
        wave_action = scene["Dofbot"].data.default_joint_pos
        # joint1~joint4에 대해 시간에 따라 변하는 사인파 기반 목표 위치 생성.
        wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        # 0.5 Hz의 주기로 진동, 진폭은 0.25 rad.
        scene["Dofbot"].set_joint_position_target(wave_action)
        # 조인트 위치 제어(Position Control) 명령 적용.
        scene.write_data_to_sim()
        # 버퍼에 기록된 모든 제어 명령/상태를 시뮬레이터에 반영.
        sim.step()
        # 1프레임 시뮬레이션 실행.
        sim_time += sim_dt
        # sim_time: 총 경과 시간 누적.
        count += 1
        # count: 프레임 카운터 증가.
        scene.update(sim_dt)
        # scene.update(): 내부 센서/관측값 갱신 등 수행.

def main():=
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # 시뮬레이션 물리 엔진 및 실행 환경 설정 객체.
    sim = sim_utils.SimulationContext(sim_cfg)
    # SimulationContext 객체를 생성.
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # 시점 정의
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    # NewRobotsSceneCfg: 이전에 정의한 Jetbot+Dofbot+지면+조명 환경 구성 클래스.
    # args_cli.num_envs: CLI 인자로부터 가져온 환경 개수.
    # num_envs > 1이면 Jetbot/Dofbot을 여러 인스턴스로 복제하여 병렬 환경 생성.
    # scene_cfg = NewRobotsSceneCfg(num_envs=10, env_spacing=2.0) 이걸 번경하면 됌
    # env_spacing=2.0: 각 환경 간격(거리) 설정.
    # 환경들이 서로 충돌하지 않도록 배치 간격 지정.
    scene = InteractiveScene(scene_cfg)
    # SceneCfg 기반으로 실제 USD Stage에 오브젝트(지면, 조명, 로봇) 스폰.
    # 로봇 상태/액추에이터 관리 및 시뮬레이션 데이터 동기화.
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
