import argparse
# Python의 표준 라이브러리로, 명령줄 인자를 처리하기 위한 모듈입니다. 사용자 입력(예: --num_envs)을 파싱
from isaaclab.app import AppLauncher
# Isaac Sim 기반의 시뮬레이션 애플리케이션을 초기화하고 실행합니다.

parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
# argparse.ArgumentParser를 사용해 명령줄 인자 파서를 생성합니다.
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# --num_envs 인자를 추가합니다.
# type=int: 정수형 입력.
# default=16: 기본값은 16개의 환경.
# help: 도움말 메시지로, 환경 수를 지정한다고 설명.
AppLauncher.add_app_launcher_args(parser)
# AppLauncher가 제공하는 추가 인자(예: --device, --headless)를 파서에 추가합니다. 이는 GPU 장치, 렌더링 모드 등을 설정하는 데 사용됩니다.
args_cli = parser.parse_args()
# 명령줄 인자를 파싱하여 args_cli 객체에 저장합니다. 예: --num_envs 32를 입력하면 args_cli.num_envs == 32.
app_launcher = AppLauncher(args_cli)
# AppLauncher 객체를 생성하여 시뮬레이션 애플리케이션을 초기화
simulation_app = app_launcher.app
# 초기화된 Isaac Sim 애플리케이션을 simulation_app에 저장합니다. 이는 시뮬레이션 실행 및 종료를 관리합니다.

import math
# 수학 연산(예: math.pi)을 위해 사용.
import torch
# PyTorch로 텐서 연산 및 GPU 가속 처리.
import isaaclab.envs.mdp as mdp
# 관찰 및 동작을 계산하는 함수
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
# 환경과 그 설정을 정의하는 클래스.
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
# EventTermCfg, ObservationGroupCfg, ObservationTermCfg: 이벤트, 관찰 그룹, 관찰 항목을 설정하는 클래스.
from isaaclab.managers import SceneEntityCfg
# 장면 내 엔티티(예: 로봇의 특정 관절)를 지정하는 클래스.
from isaaclab.utils import configclass
# 설정 클래스를 정의하기 위한 데코레이터.
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg
# 이전에 정의된 Cartpole 장면 설정을 가져옴(지면, 조명, Cartpole 로봇 포함).

@configclass
# ActionsCfg를 설정 클래스로 정의.
class ActionsCfg:
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)
    # JointEffortActionCfg를 사용해 Cartpole의 카트에 적용할 힘을 정의.
    # asset_name="robot": Cartpole 로봇을 대상으로 지정.
    # joint_names=["slider_to_cart"]: 카트의 슬라이더 관절에 힘을 적용.
    # scale=5.0: 힘의 스케일을 5배로 조정(입력된 힘에 5를 곱함).

@configclass
class ObservationsCfg:
# 환경의 관찰을 정의.
    @configclass
    class PolicyCfg(ObsGroup):
    # ObsGroup을 상속받아 "policy"라는 관찰 그룹을 정의
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # 함수로 관절의 상대적 위치를 계산.
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # 함수로 관절의 상대적 속도를 계산.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            # 관찰에 노이즈를 추가하지 않음.
            self.concatenate_terms = True
            # 관찰 항목(위치, 속도)을 하나의 텐서로 결합.
    policy: PolicyCfg = PolicyCfg()
    # "policy" 그룹을 정의. 이는 강화학습 에이전트가 사용할 관찰 공간.

@configclass
class EventCfg:
# 시뮬레이션 이벤트(예: 초기화, 리셋)를 정의.
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        # 폴의 질량을 무작위로 변경.
        mode="startup",
        # 시뮬레이션 시작 시 1회 실행.
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            # Cartpole의 "pole" 바디를 대상으로 지정.
            "mass_distribution_params": (0.1, 0.5),
            # 질량을 0.1~0.5kg 범위에서 추가.
            "operation": "add",
            # 기존 질량에 추가.
        },
    )
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        # 카트의 관절 위치와 속도를 리셋.
        mode="reset",
        # 환경 리셋 시 실행.
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            # "slider_to_cart" 관절 대상.
            "position_range": (-1.0, 1.0),
            # 카트 위치를 ±1.0m 범위에서 무작위 설정.
            "velocity_range": (-0.1, 0.1),
            # 카트 속도를 ±0.1m/s 범위에서 설정.
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        # 폴의 관절 위치와 속도를 리셋.
        mode="reset",
        # 환경 리셋 시 실행.
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            # "cart_to_pole" 관절 대상.
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            # 폴 각도를 ±7.16도(0.125π 라디안) 범위에서 설정.
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
            # 폴 각속도를 ±0.0314 rad/s 범위에서 설정.
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
# ManagerBasedEnvCfg를 상속받아 Cartpole 환경을 정의.
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # CartpoleSceneCfg를 사용하며, num_envs=1024로 1024개의 환경을 설정, env_spacing=2.5로 환경 간 거리를 2.5m로 지정.
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    # observations, actions, events: 위에서 정의한 관찰, 동작, 이벤트 설정을 사용.

    def __post_init__(self):
        self.viewer.eye = [4.5, 0.0, 6.0]
        # 카메라 위치 설정.
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # 카메라가 바라보는 지점.
        self.decimation = 4
        # 시뮬레이션 스텝마다 환경을 4번 업데이트(200Hz / 4 = 50Hz).
        self.sim.dt = 0.005
        # 시뮬레이션 시간 간격을 5ms(200Hz)로 설정.

def main():
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # 명령줄에서 입력받은 num_envs로 환경 수를 재설정(기본값 16).
    env_cfg.sim.device = args_cli.device
    # GPU/CPU 장치를 설정(예: cuda 또는 cpu).
    env = ManagerBasedEnv(cfg=env_cfg)
    # ManagerBasedEnv 객체를 생성하여 환경 초기화.
    count = 0
    # 시뮬레이션 스텝 카운터 초기화.
    while simulation_app.is_running():
    # 시뮬레이션이 실행 중인 동안 루프.
        with torch.inference_mode():
        # PyTorch의 자동 미분(autograd)을 비활성화하여 성능 최적화.
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            joint_efforts = torch.randn_like(env.action_manager.action)
            obs, _ = env.step(joint_efforts)
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            count += 1
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
