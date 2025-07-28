# 독립형 Python 스크립트로 작업할 때 첫 번째 단계는 시뮬레이션 애플리케이션을 실행하는 것입니다. 
# Isaac Sim의 다양한 종속성 모듈은 시뮬레이션 앱이 실행된 후에만 사용할 수 있으므로, 이 작업은 처음에 필수적입니다.

import argparse
# argparse는 **명령줄 인수(Command-Line Arguments)**를 처리하는 데 사용됩니다. 
# 즉, 사용자가 터미널에서 스크립트를 실행할 때 추가적인 설정(옵션)을 입력할 수 있게 해줍니다.
# 터미널에서 python script.py --num_envs 10 --headless를 입력하면, argparse가 이 입력을 파싱(parse)해서 스크립트에서 사용할 수 있는 데이터로 변환해요.
from isaaclab.app import AppLauncher
# Isaac Lab 프레임워크에서 제공하는 AppLauncher 클래스를 가져옵니다.
# --headless: GUI 없이 실행, 서버 환경에서 리소스 절약.
# --livestream: 시뮬레이션을 원격으로 스트리밍.
# --offscreen_render: 화면 렌더링 없이 물리 시뮬레이션만 수행.
# --width, --height: 뷰포트 크기 설정.

parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim import SimulationCfg, SimulationContext

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    sim.reset()
    print("[INFO]: Setup complete...")
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
