import argparse
# 터미널에 입력한 명령줄 인수를 처리하는데 사용함
from isaaclab.app import AppLauncher
# Isaac sim  애플리케이션을 실행하고 구동하는데 필요함

parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
# ArgumentParser 객체를 생성함 설명은 description으로 설명함
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# 큐보이드 크기 인자
# 옵션명: --size
# 자료형: float (실수형)
# 기본값: 1.0
# 용도: 변형 가능한 큐보이드(또는 특정 오브젝트)의 한 변의 길이 지정.
# 큐보이드는 6개의 사각형 면을 가진 **직각 평행육면체 각 면은 직사각형이며, 인접한 면은 직각으로 만납니다.
parser.add_argument("--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280")
# **뷰포트(Viewport)**는 그래픽 환경(시뮬레이션, CAD, 게임 엔진 등)에서 3D 장면을 표시하는 화면 영역을 의미합니다.
# **3D 가상 장면을 2D 화면에 투영하여 표시하는 "시각 창"**입니다.
# 카메라 시점(Camera View)과 렌더링 해상도를 반영하여 사용자가 시뮬레이션이나 모델을 시각적으로 확인할 수 있도록 합니다.
# 뷰포트 가로 해상도 인자
# 옵션명: --width
# 자료형: int (정수형)
# 기본값: 1280 (픽셀 단위)
# 용도: 시뮬레이션 뷰포트(Viewport) 또는 캡처 이미지의 가로 해상도 지정.
parser.add_argument("--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720")
# 뷰포트 세로 해상도 인자
# 옵션명: --height
# 자료형: int (정수형)
# 기본값: 720 (픽셀 단위)
# 용도: 뷰포트 및 이미지 캡처의 세로 해상도 지정.
# 터미널에 다음과 같이 실행하면
# python sim_env.py --size 0.8 --width 1920 --height 1080
# 큐보이드 한 변 길이를 0.8m로 하고, 1920×1080 해상도로 시뮬레이션을 실행합니다.
AppLauncher.add_app_launcher_args(parser)
# 앞에서 생성된 객체 parser에 isaaclab 명령줄 인수를 추가함
args_cli = parser.parse_args()
# 명령줄 인수를 pytion형식으로 변환함
app_launcher = AppLauncher(args_cli)
# isaac sim 애플리케이션의 실행을 설정함
simulation_app = app_launcher.app
# isaac sim 애플리케이션의 실행

import isaaclab.sim as sim_utils
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html
# NVIDIA의 Isaac Lab 프레임워크에서 제공하는 isaaclab.sim 모듈을 가져와 sim_utils라는 별칭으로 사용하는 코드

def design_scene():
    cfg_ground = sim_utils.GroundPlaneCfg()
    # 이 클래스는 지면 프림(Ground Plane Primitive)의 속성(크기, 색상 등)을 설정하기 위한 구성 객체(Config Object).
    # 여기서는 기본값을 사용(size=None, color=None).
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    # GroundPlaneCfg.func 메서드를 호출하여 USD 스테이지 경로 /World/defaultGroundPlane에 지면 프림을 생성.
    # cfg_ground 객체에 지정된 속성을 이용해 USD 내에 Ground Plane을 추가.
    cfg_light_distant = sim_utils.DistantLightCfg(
    # 원거리 조명(Directional Light) 구성 객체 생성
        intensity=3000.0,
        # 밝기
        color=(0.75, 0.75, 0.75),
        # 색
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    # func 메서드를 통해 /World/lightDistant 경로에 원거리 조명을 생성.
    # 위치(translation) (1, 0, 10)에 배치하여 장면 전체를 균일하게 조명.
    
    cfg_cuboid = sim_utils.CuboidCfg(
    큐보이드(직육면체) 생성 설정 클래스.

        size=[args_cli.size] * 3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    )
    cfg_cuboid.func("/World/Object", cfg_cuboid, translation=(0.0, 0.0, args_cli.size / 2))


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
