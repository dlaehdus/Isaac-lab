import argparse
# Python의 명령줄 인자 파싱 모듈.
from isaaclab.app import AppLauncher
# 이는 시뮬레이션 애플리케이션(Isaac Sim 기반)을 초기화하고 실행하는 데 사용됨.

parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
# 명령줄 인자를 처리하기 위한 파서를 생성
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# --num_envs: 시뮬레이션 환경의 수를 지정하는 인자.
# type=int: 정수형.
# default=2: 기본값은 2개의 환경.
# help="Number of environments to spawn.": 도움말 메시지.
AppLauncher.add_app_launcher_args(parser)
# AppLauncher가 제공하는 추가 인자(예: GPU 장치, 렌더링 설정 등)를 파서에 추가.
args_cli = parser.parse_args()
# 명령줄 인자를 파싱하여 args_cli 객체에
app_launcher = AppLauncher(args_cli)
# AppLauncher 객체를 생성하여 시뮬레이션 환경을 초기화.
simulation_app = app_launcher.app
# 초기화된 Isaac Sim 애플리케이션을 가져옴.

import torch
# PyTorch로 텐서 연산 및 GPU 가속 처리.
import isaaclab.sim as sim_utils
# 물리 엔진, 렌더링, 카메라 설정 등 시뮬레이션 유틸리티.
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# ArticulationCfg(관절 객체 설정), AssetBaseCfg(기본 자산 설정).
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# InteractiveScene과 InteractiveSceneCfg로 장면 구성.
from isaaclab.sim import SimulationContext
# 시뮬레이션 이벤트 관리
from isaaclab.utils import configclass
# 데이터 클래스를 설정하기 위한 데코레이터.
from isaaclab_assets import CARTPOLE_CFG
# 데이터 클래스를 설정하기 위한 데코레이터.

@configclass
# Isaac Lab의 유틸리티로, 클래스를 설정 객체로 변환해 속성을 체계적으로 관리.
class CartpoleSceneCfg(InteractiveSceneCfg):
# InteractiveSceneCfg를 상속받아 시뮬레이션 장면을 정의.
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # ground: 지면을 /World/defaultGroundPlane에 생성. GroundPlaneCfg로 평평한 바닥 설정.
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    # 돔 조명을 생성. 세기 3000, 색상은 밝은 회색(0.75, 0.75, 0.75).
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # CARTPOLE_CFG를 기반으로 Cartpole 로봇을 {ENV_REGEX_NS}/Robot 경로에 배치.

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
# sim: sim_utils.SimulationContext: 시뮬레이션 환경을 제어하는 객체.
# scene: InteractiveScene: CartpoleSceneCfg로 정의된 장면(지면, 조명, Cartpole 로봇 포함).
    robot = scene["cartpole"]
    # InteractiveScene에서 "cartpole" 키로 Cartpole 로봇을 가져옴. 이는 CartpoleSceneCfg에서 정의된 /World/envs/env_{i}/Robot 경로의 로봇들.
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
