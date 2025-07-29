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
Mesh Prim → 삼각형 메쉬(예: 원, 사각형, 복잡한 형상)
# Light Prim → 조명
# Camera Prim → 카메라
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import CARTPOLE_CFG


def design_scene() -> tuple[dict, list[list[float]]]:
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    scene_entities = {"cartpole": cartpole}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    robot = entities["cartpole"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print("[INFO]: Resetting robot state...")
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


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
