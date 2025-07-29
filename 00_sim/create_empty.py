# 독립형 Python 스크립트로 작업할 때 첫 번째 단계는 시뮬레이션 애플리케이션을 실행하는 것입니다. 
# Isaac Sim의 다양한 종속성 모듈은 시뮬레이션 앱이 실행된 후에만 사용할 수 있으므로, 이 작업은 처음에 필수적입니다.

import argparse
# argparse는 **명령줄 인수(Command-Line Arguments)**를 처리하는 데 사용됩니다. 
# 즉, 사용자가 터미널에서 스크립트를 실행할 때 추가적인 설정(옵션)을 입력할 수 있게 해줍니다.
# 터미널에서 python script.py --num_envs 10 --headless를 입력하면, argparse가 이 입력을 파싱(parse)해서 스크립트에서 사용할 수 있는 데이터로 변환해요.
from isaaclab.app import AppLauncher
# isaaclab.app 서브패키지는 Isaac Sim 애플리케이션의 실행 및 구성을 위한 앱별 기능을 제공합니다. 주요 기능은 다음과 같습니다:
# 다양한 구성으로 시뮬레이션 앱을 실행하는 기능
# 시뮬레이션 앱으로 테스트를 실행하는 기능
# AppLauncher는 명령줄 인수와 환경 변수를 기반으로 Isaac Sim 애플리케이션을 실행하는 유틸리티 클래스입니다.
# 이 클래스는 환경 변수, 명령줄 인수, 또는 키워드 인수를 통해 설정된 시뮬레이션 앱 설정을 해석하고, 이를 바탕으로 시뮬레이션 앱을 실행하며, 실행 후 필요한 확장 프로그램을 구성합니다.
# AppLauncher 클래스는 환경 변수를 통해 시뮬레이션 앱의 동작을 제어합니다. 아래는 관련 환경 변수와 그 동작에 대한 자세한 설명입니다.

# 헤드리스 모드 (Headless Mode)
# 환경 변수: HEADLESS=1
# 동작: HEADLESS=1로 설정하면 시뮬레이션 앱이 헤드리스 모드(그래픽 사용자 인터페이스(GUI) 없이 실행)로 시작됩니다.
# 참고: LIVESTREAM={1,2}가 설정된 경우, 이 설정은 HEADLESS 환경 변수를 덮어쓰고 헤드리스 모드를 강제로 활성화합니다.

# 라이브스트리밍 (Livestreaming)
# 환경 변수: LIVESTREAM={1,2}
# 동작: LIVESTREAM 값이 {1,2} 중 하나로 설정되면 라이브스트리밍이 활성화되며, 앱은 헤드리스 모드로 실행됩니다.
# LIVESTREAM=1: (Deprecated) Isaac Native Livestream 확장을 통해 스트리밍을 활성화합니다. 사용자는 Omniverse Streaming Client를 통해 연결할 수 있습니다. 이 방법은 Isaac Sim 4.5부터 더 이상 지원되지 않으며, WebRTC 라이브스트리밍을 사용하는 것이 권장됩니다.
# LIVESTREAM=2: WebRTC Livestream 확장을 통해 스트리밍을 활성화합니다. 사용자는 WebRTC 프로토콜을 사용하는 WebRTC 클라이언트를 통해 연결할 수 있습니다.
# 주의: 각 Isaac Sim 인스턴스는 하나의 스트리밍 클라이언트에만 연결할 수 있습니다. 이미 스트리밍 클라이언트에 연결된 Isaac Sim 인스턴스에 두 번째 사용자가 연결하려고 하면 오류가 발생합니다.
# 공인 IP 주소 (Public IP Address):
# LIVESTREAM={1,2}가 설정된 경우, PUBLIC_IP 환경 변수를 설정하여 원격 라이브스트리밍을 위한 공인 IP 주소 엔드포인트를 지정할 수 있습니다.

# 카메라 활성화 (Enable Cameras)
# 환경 변수: ENABLE_CAMERAS=1
# 동작: ENABLE_CAMERAS=1로 설정하면 카메라가 활성화됩니다. 이는 GUI 없이 시뮬레이터를 실행하면서 뷰포트와 카메라 이미지를 렌더링하는 데 유용합니다.
# 세부 동작: ENABLE_CAMERAS=1은 오프스크린 렌더링 파이프라인을 활성화하여 GUI 없이 장면을 렌더링할 수 있도록 합니다.
# 주의: 오프스크린 렌더링 파이프라인은 isaaclab.sim.SimulationContext 클래스와 함께 사용해야만 작동합니다. 이는 오프스크린 렌더링 파이프라인이 SimulationContext 클래스에서 내부적으로 사용하는 플래그를 활성화하기 때문입니다.

parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
AppLauncher.add_app_launcher_args(parser)
# 기존 argparse.ArgumentParser 객체에 AppLauncher 관련 인수를 추가하는 유틸리티 함수입니다. 이 함수는 SimulationApp과 관련된 커맨드라인 인수를 추가하여 환경 변수를 오버라이드할 수 있도록 합니다.
# headless (bool): True일 경우 앱이 헤드리스(노-GUI) 모드로 실행됩니다. HEADLESS 환경 변수와 동일한 값을 가집니다. False일 경우, 헤드리스 모드는 HEADLESS 환경 변수에 따라 결정됩니다.
# livestream (int): {1, 2} 중 하나일 경우, 라이브스트리밍과 헤드리스 모드가 활성화됩니다. 값은 LIVESTREAM 환경 변수와 동일하게 매핑됩니다. -1일 경우, 라이브스트리밍은 LIVESTREAM 환경 변수에 따라 결정됩니다. 유효한 옵션은 다음과 같습니다:
# 0: 비활성화
# 1: Native (Deprecated)
# 2: WebRTC

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
