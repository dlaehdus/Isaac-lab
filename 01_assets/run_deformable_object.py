import argparse
# 명령줄 인수를 처리함
from isaaclab.app import AppLauncher
# isaac sim 애플리케이션 실행

parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
# ArgumentParser객체를 생성함
AppLauncher.add_app_launcher_args(parser)
# isaac sim 명령줄 인수를 객체에 추가함
args_cli = parser.parse_args()
# pytion이 알아듣기 쉽게 명령줄 인수를 변환
app_launcher = AppLauncher(args_cli)
# isaac sim 애플리케이션 실행 설정을 구성함
simulation_app = app_launcher.app
# isaac sim 애플리케이션 실행

import torch
# gpu가속기 사용, 딥러닝
import isaacsim.core.utils.prims as prim_utils
# 시뮬레이션 환경에서 프림(Prim), 즉 3D 객체(메시, 조명, 카메라 등)를 생성하거나 관리하는 데 사용됩니다.
# 로봇, 물체, 환경 요소를 시뮬레이션 내에서 배치하거나 조작.
import isaaclab.sim as sim_utils
# VIDIA Isaac Lab의 시뮬레이션 유틸리티 모듈로, 물리 엔진 설정, 렌더링, 카메라 조작 등을 지원합니다
import isaaclab.utils.math as math_utils
# Isaac Lab의 수학 유틸리티 모듈로, 3D 변환(회전, 이동), 쿼터니언, 벡터 연산 등 시뮬레이션에 필요한 수학적 연산을 제공합니다.
from isaaclab.assets import DeformableObject, DeformableObjectCfg
# DeformableObject는 변형 가능한 객체(예: 천, 고무, 유체 같은 소프트 바디)를 시뮬레이션하기 위한 클래스입니다.
# DeformableObjectCfg는 해당 객체의 설정(예: 강성, 마찰, 질량)을 정의하는 구성 클래스입니다.
from isaaclab.sim import SimulationContext
# 물리적 스태핑, 렌더닝과 같은 시뮬레이션 이벤트를 제어하는 클래스


def design_scene():
    cfg = sim_utils.GroundPlaneCfg()
    # 지면을 생성함
    cfg.func("/World/defaultGroundPlane", cfg)
    # 지면을 배치함
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    # 빛을 생성함
    cfg.func("/World/Light", cfg)
    # 빛을 배치함
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    # origins는 3D 좌표로 이루어진 리스트
    # 각 좌표는 [x, y, z] 형식으로, 4개의 점을 정의합니다. 
    # 이 좌표들은 아마도 시뮬레이션 환경(예: Isaac Lab)에서 객체(예: DeformableObject)의 초기 위치를 설정하는 데 사용
    for i, origin in enumerate(origins):
    # origins: 이전에 정의된 좌표 리스트 [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]를 참조
    # enumerate(origins): origins 리스트를 반복하면서 각 좌표와 해당 인덱스(i)를 가져옵니다.
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
        # Isaac Sim의 유틸리티 함수로, 시뮬레이션 환경에 새로운 프림(3D 객체)을 생성합니다.
        # f"/World/Origin{i}": 프림의 경로. /World/Origin0, /World/Origin1, /World/Origin2, /World/Origin3과 같이 각 좌표에 대해 고유한 이름으로 프림이 생성됩니다.
    cfg = DeformableObjectCfg(
    # DeformableObjectCfg는 변형 가능한 객체의 설정을 정의하는 클래스입니다. 이 객체는 물리적으로 변형될 수 있는 소프트 바디(예: 고무, 천)를 시뮬레이션합니다.
        prim_path="/World/Origin.*/Cube",
        # 프림 경로로, 와일드카드(.*)를 사용해 /World/Origin0/Cube, /World/Origin1/Cube, /World/Origin2/Cube, /World/Origin3/Cube에 해당하는 모든 프림에 큐브를 생성합니다.
        # 이는 이전 코드에서 origins에 따라 생성된 /World/Origin{i} 프림들 아래에 큐브 객체를 배치하겠다는 의미입니다.
        spawn=sim_utils.MeshCuboidCfg(
        # 큐브의 기하학적 및 물리적 속성을 정의합니다.
            size=(0.2, 0.2, 0.2),
            # 큐브의 크기
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            # deformable_props: 변형 가능한 물체의 물리적 속성.
            # rest_offset=0.0: 객체의 휴지 상태에서의 접촉 거리.
            # contact_offset=0.001: 충돌 감지를 위한 접촉 거리.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            # diffuse_color=(0.5, 0.1, 0.0): 큐브의 색상(RGB로 주황빛 색상).
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            # physics_material: 물리적 재질.
            # poissons_ratio=0.4: 푸아송 비율(재질의 횡변형 정도).
            # youngs_modulus=1e5: 영률(재질의 강성, 100kPa).
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        # 큐브의 초기 위치를 (0.0, 0.0, 1.0)으로 설정. 이는 로컬 좌표로, /World/Origin{i} 프림의 좌표를 기준으로 (0, 0, 1)만큼 오프셋됩니다.
        # 따라서 실제 월드 좌표는 /World/Origin{i}의 위치(origins[i])에 (0, 0, 1)을 더한 값입니다.
        debug_vis=True,
        # 디버깅용 시각화를 활성화. 예를 들어, 객체의 변형 상태나 충돌 지점을 시각적으로 확인 가능.
    )
    cube_object = DeformableObject(cfg=cfg)
    # DeformableObject 클래스를 사용해 설정(cfg)을 기반으로 변형 가능한 큐브 객체를 생성합니다.
    # /World/Origin0/Cube, /World/Origin1/Cube, /World/Origin2/Cube, /World/Origin3/Cube에 각각 큐브가 생성되며, 각 큐브는 origins의 좌표에 (0, 0, 1)을 더한 위치에 배치됩니다.
    scene_entities = {"cube_object": cube_object}
    # 생성된 cube_object를 딕셔너리에 저장. 이는 시뮬레이션 환경에서 객체를 관리하거나 참조하기 위해 사용됩니다.
    # scene_entities는 다른 코드에서 객체를 참조하거나 제어(예: 물리적 상호작용, 위치 업데이트)할 때 유용합니다.
    return scene_entities, origins
    # 함수가 scene_entities(큐브 객체를 포함한 딕셔너리)와 origins(좌표 리스트)를 반환합니다.

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor):
# sim: sim_utils.SimulationContext: 시뮬레이션 환경을 제어하는 객체.
# entities: dict[str, DeformableObject]: DeformableObject를 포함한 딕셔너리. 여기서는 cube_object를 "cube_object" 키로 참조.
# origins: torch.Tensor: 이전에 정의된 좌표(예: [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]])의 텐서 형태.
    cube_object = entities["cube_object"]
    # entities 딕셔너리에서 cube_object를 가져옴. 이는 이전 코드에서 생성된 4개의 변형 가능한 큐브(/World/Origin{i}/Cube)를 나타냅니다.
    sim_dt = sim.get_physics_dt()
    # 시뮬레이션의 시간 간격(물리 스텝 크기).
    sim_time = 0.0
    # 시뮬레이션 누적 시간
    count = 0
    # 루프 카운터
    nodal_kinematic_target = cube_object.data.nodal_kinematic_target.clone()
    # 큐브의 노드(격자점) 위치를 제어하기 위한 kinematic target을 복사. 이는 변형 가능한 객체의 각 노드의 목표 위치/속도를 정의.
    while simulation_app.is_running():
    # 시뮬레이션 애플리케이션이 실행 중인 동안 루프를 반복.
        if count % 250 == 0:
        # 250카운터 후 객체 상태를 리셋
            sim_time = 0.0
            # 누적시간을 0 으로
            count = 0
            # 루프 카운터를 0 으로
            nodal_state = cube_object.data.default_nodal_state_w.clone()
            # 기본 노드 상태를 복사
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
            # 각 큐브 위치를 0.1을 더해 업데이트
            # cube_object.num_instances는 큐브 수로, origins의 길이(4)에 의해 결정됨. 즉, 4개의 큐브.
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            # 랜덤 회전을 생성
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
            # 노드 위치를 새로운 위치(pos_w)와 회전(quat_w)으로 변환.
            cube_object.write_nodal_state_to_sim(nodal_state)
            # 변환된 노드 상태를 시뮬레이션에 적용.
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            # 목표 위치를 업데이트.
            nodal_kinematic_target[..., 3] = 1.0
            # 목표 속도(또는 활성화 플래그)를 설정.
            cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
            # 목표 상태를 시뮬레이션에 적용.
            cube_object.reset()
            # 객체 상태를 리셋.
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        nodal_kinematic_target[[0, 3], 0, 2] += 0.001
        # 첫 번째와 네 번째 큐브의 첫 번째 노드의 z좌표를 매 스텝마다 0.001만큼 증가. 이는 큐브의 일부 노드가 위로 움직이도록 만듦.
        nodal_kinematic_target[[0, 3], 0, 3] = 0.0
        # 해당 노드의 속도(또는 활성화 플래그)를 0으로 설정.
        cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
        # 업데이트된 목표를 적용.
        cube_object.write_data_to_sim()
        # 큐브 데이터를 시뮬레이션에 반영.
        sim.step()
        # 시뮬레이션을 한 스텝 진행.
        sim_time += sim_dt
        count += 1
        cube_object.update(sim_dt)
        # 큐브 상태를 업데이트.
        if count % 50 == 0:
            print(f"Root position (in world): {cube_object.data.root_pos_w[:, :3]}")
            # count % 50 == 0: 50 스텝마다 큐브의 루트 위치(root_pos_w)를 출력.


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
