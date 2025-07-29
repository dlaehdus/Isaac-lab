import argparse
# 명령줄 인자 처리용 라이브러리
from isaaclab.app import AppLauncher
# Isaac sim  실행 관련 라이브러리

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# ArgumentParser객체를 생성함
AppLauncher.add_app_launcher_args(parser)
# 생성된 객체에 isaaclab 명령줄 인수를 추가함
args_cli = parser.parse_args()
# 명령줄 인자를 pytion언어로 변화함
# args_cli : 명령줄 인자를 파싱한 객체.
app_launcher = AppLauncher(args_cli)
# isaacsim 애플리케이션 설정
simulation_app = app_launcher.app
# 실행

import torch
# GPU가속 연산을 위한 라이브러리
import isaacsim.core.utils.prims as prim_utils
# 3D 기본 도형 큐브, 구등을 나타내는 프림을 불러옴
import isaaclab.sim as sim_utils
# Isaac Lab 시뮬레이션용 유틸리티 모듈.
import isaaclab.utils.math as math_utils
# 수학기능을 사용함
from isaaclab.assets import RigidObject, RigidObjectCfg
# 강체 객체 생성 및 설정 클래스.
from isaaclab.sim import SimulationContext
# 시뮬레이션 실행 환경 클래스.

def design_scene():
    cfg = sim_utils.GroundPlaneCfg()
    # 지면을 생성함
    cfg.func("/World/defaultGroundPlane", cfg)
    # /World/defaultGroundPlane경로에 평면을 배치함
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    # 빛을 생성함
    cfg.func("/World/Light", cfg)
    # 빛을 배치함
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    # 원뿔 배치 기준 설정
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    cone_cfg = RigidObjectCfg(
    # 원뿔 강체를 생성함
        prim_path="/World/Origin.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            # 반지름
            height=0.2,
            # 높이 설정
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            # 강체 속성을 정의함
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            # 질량을 설정
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            # 녹색에 금속성 0.2인 시각적 재질
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
        # 초기 상태 설정
    )
    cone_object = RigidObject(cfg=cone_cfg)
    # USD stage 상에 프림 스폰 및 시뮬레이터에 등록.
    scene_entities = {"cone": cone_object}
    return scene_entities, origins
    # 생성된 원뿔 객체를 딕셔너리로 관리 후 반환, 그리고 원점 좌표 리스트도 반환.


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    cone_object = entities["cone"]
    sim_dt = sim.get_physics_dt()
    # 시뮬레이션 간격
    sim_time = 0.0
    # 기뮬레이션 누적 시간
    count = 0
    # 루프 카운터
    while simulation_app.is_running():
        if count % 250 == 0:
            sim_time = 0.0
            count = 0
            root_state = cone_object.data.default_root_state.clone()
            root_state[:, :3] += origins
            # USD 프림의 위치, 회전, 속도 등이 담긴 초기 상태 텐서
            # 각 인스턴스별로 위치를 origins 좌표에 맞게 오프셋 적용
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device
            )
            # 원뿔 위치에 반경 0.1m, 높이 0.25~0.5m 범위 내에서 난수 분포를 샘플링하여 약간씩 위치 분산.
            cone_object.write_root_pose_to_sim(root_state[:, :7])
            cone_object.write_root_velocity_to_sim(root_state[:, 7:])
            cone_object.reset()
            # 상태(위치, 회전)와 속도를 시뮬레이터에 적용 후 리셋.
            # 리셋으로 내부 상태, 충돌 등 초기화.
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        cone_object.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        cone_object.update(sim_dt)
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_pos_w}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
