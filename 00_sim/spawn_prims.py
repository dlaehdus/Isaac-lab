import argparse
# 터미널에서 pytion 스트립트를 실행할 떄 스트립트 이름 뒤에 추가로 입력하는 옵션이나 값을 처리함
from isaaclab.app import AppLauncher
# isaac sim 애플리케이션을 실행할때 실행 및 구성을 위한 기능을 제공

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# argparse를 사용해서 ArgumentParser객체를 생성함 이것은 명령줄 인수를 처리하는데 사용함
AppLauncher.add_app_launcher_args(parser)
# 앞에서 생성된 객체 parser에 isaaclab 명령줄 인수를 추가함
args_cli = parser.parse_args()
# 터미널에 입력한 명령줄 인수를 처리할때 pytion이 알아들을수 있도록 argparse.Namespace 객체로 반환해줌
app_launcher = AppLauncher(args_cli)
# pytion이 알아들을수 있는 코드로 작성된 args_cli을 받아 isaac sim 애플리케이션의 실행을 설정함
simulation_app = app_launcher.app
# 이 단계에서 isaac sim 애플리케이션이 실제로 실행이 됨

import isaacsim.core.utils.prims as prim_utils
# https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.core.utils/docs/index.html
# isaacsim.core.utils.prims를 가져와서 prim_utils라는 별칭으로 사용하는 코드
# 이 모듈은 USD(Universal Scene Description) 스테이지에서 **프림(prim)**을 다루기 위한 유틸리티 함수들을 제공합니다.
# 프림은 USD에서 3D 장면의 기본 구성 요소로, 객체(예: 큐브, 구), 카메라, 조명, 변환 등을 나타냅니다.
# USD 스테이지에서 프림을 생성, 삭제, 속성 설정/조회, 경로 관리, 관계 설정 등을 처리하는 유틸리티 함수를 제공합니다.
# **프림(Prim)**은 **USD(Universal Scene Description)**에서 3D 장면을 구성하는 기본 단위입니다. USD는 NVIDIA Omniverse와 Isaac Sim에서 사용되는 오픈소스 포맷으로, 3D 장면의 객체, 속성, 관계 등을 기술합니다.

import isaaclab.sim as sim_utils
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html
# NVIDIA의 Isaac Lab 프레임워크에서 제공하는 isaaclab.sim 모듈을 가져와 sim_utils라는 별칭으로 사용하는 코드
# 이 모듈은 Isaac Sim 기반의 시뮬레이션 환경을 구성하고 관리하기 위한 고수준 기능을 제공하며, 특히 로봇 학습(예: 강화 학습, 모방 학습)과 시뮬레이션 작업을 간소화하는 데 초점을 둡니다.
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.assets

def design_scene():
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    prim_utils.create_prim("/World/Objects", "Xform")
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    design_scene()

    sim.reset()
    print("[INFO]: Setup complete...")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
    # isaac sim 애플리케이션을 정상적으로 종료함.
