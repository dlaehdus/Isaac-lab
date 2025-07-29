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
# 이 줄은 Python의 argparse 모듈을 사용하여 새로운 ArgumentParser 객체를 생성합니다.
# argparse.ArgumentParser는 커맨드라인에서 입력된 인수를 파싱(parse)하는 데 사용되는 객체입니다. 이를 통해 사용자가 스크립트 실행 시 커맨드라인에서 전달한 인수(예: --headless, --livestream 2)를 처리할 수 있습니다
# **파싱(parsing)**은 프로그래밍에서 데이터를 분석하고 구조화된 형태로 변환하는 과정을 의미합니다. 주로 문자열이나 파일과 같은 입력 데이터를 읽어서, 프로그램이 이해하고 처리할 수 있는 구조(예: 객체, 리스트, 딕셔너리 등)로 나누는 작업을 말합니다.
# description 매개변수는 ArgumentParser 객체의 설명을 정의합니다. description 매개변수는 ArgumentParser 객체의 설명을 정의합니다. 이 설명은 사용자가 스크립트를 실행할 때 --help 또는 -h 플래그를 사용하면 표시됩니다.
# 이 설명은 사용자가 스크립트를 실행할 때 --help 또는 -h 플래그를 사용하면 표시됩니다.
# 터미널에서 python script.py --help를 실행
AppLauncher.add_app_launcher_args(parser)
# 이 줄은 isaaclab.app.AppLauncher 클래스의 정적 메서드인 add_app_launcher_args를 호출하여, 앞서 생성한 parser 객체에 Isaac Sim 애플리케이션 실행과 관련된 커맨드라인 인수를 추가합니다.
# add_app_launcher_args는 ArgumentParser 객체에 AppLauncher가 사용하는 특정 인수들을 추가하여, 사용자가 커맨드라인에서 Isaac Sim의 실행 설정(예: 헤드리스 모드, 라이브스트리밍, 카메라 렌더링 등)을 제어할 수 있도록 합니다.
# 기존 argparse.ArgumentParser 객체에 AppLauncher 관련 인수를 추가하는 유틸리티 함수입니다. 이 함수는 SimulationApp과 관련된 커맨드라인 인수를 추가하여 환경 변수를 오버라이드할 수 있도록 합니다.
# headless (bool): True일 경우 앱이 헤드리스(노-GUI) 모드로 실행됩니다. HEADLESS 환경 변수와 동일한 값을 가집니다. False일 경우, 헤드리스 모드는 HEADLESS 환경 변수에 따라 결정됩니다.
# livestream (int): {1, 2} 중 하나일 경우, 라이브스트리밍과 헤드리스 모드가 활성화됩니다. 값은 LIVESTREAM 환경 변수와 동일하게 매핑됩니다. -1일 경우, 라이브스트리밍은 LIVESTREAM 환경 변수에 따라 결정됩니다. 유효한 옵션은 다음과 같습니다:
#     0: 비활성화
#     1: Native (Deprecated)
#     2: WebRTC Livestream 확장을 통해 스트리밍. WebRTC 프로토콜을 사용하는 WebRTC 클라이언트를 통해 연결.
#     -1: 라이브스트리밍 설정은 LIVESTREAM 환경 변수에 따라 결정됩니다.
#     예시: --livestream 2를 사용하여 WebRTC 스트리밍을 활성화할 수 있습니다.
# enable_cameras (bool) 설명: 헤드리스 모드에서도 카메라 센서를 활성화하고 뷰포트 및 카메라 이미지를 렌더링할지 결정합니다.
# device (str)
#     설명: 시뮬레이션을 실행할 하드웨어 장치를 지정합니다.
#     cpu: CPU 사용.
#     cuda: GPU 사용 (디바이스 ID 0).
#     cuda:N: GPU 사용, N은 디바이스 ID (예: cuda:0).
# experience (str)
#     설명: Isaac Sim 실행 시 로드할 경험 파일(experience file)을 지정합니다.
# parser 객체에 AppLauncher 관련 커맨드라인 옵션을 추가하여, 사용자가 Isaac Sim의 실행 방식을 커맨드라인에서 설정할 수 있도록 합니다.
args_cli = parser.parse_args()
# 이 줄은 parser 객체의 parse_args() 메서드를 호출하여 커맨드라인에서 전달된 인수를 파싱(parse)하고, 이를 argparse.Namespace 객체로 반환합니다.
# parse_args()는 스크립트 실행 시 커맨드라인에 입력된 인수들을 읽어들여, parser에 정의된 인수들에 따라 값을 매핑합니다.
# 반환된 args_cli 객체는 각 인수의 이름과 값을 속성으로 가지는 객체입니다. 예를 들어, --headless가 커맨드라인에 지정되었다면, args_cli.headless는 True가 됩니다.
# 커맨드라인에서 입력된 인수를 파싱하여 Python에서 사용할 수 있는 객체로 변환합니다.
# 파싱된 인수는 이후 AppLauncher에 전달되어 Isaac Sim 실행 설정을 결정합니다.
# 만약 터미널에 입력을 ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless --livestream 2이라고 하면
# args_cli = Namespace(headless=True, livestream=2, enable_cameras=False, device=None, experience=None, kit_args=None)
app_launcher = AppLauncher(args_cli)
# 이 줄은 isaaclab.app.AppLauncher 클래스의 인스턴스를 생성합니다.
# AppLauncher 클래스는 args_cli 객체(파싱된 커맨드라인 인수)를 받아, 이를 기반으로 Isaac Sim 애플리케이션의 실행 설정을 구성합니다.
# AppLauncher는 환경 변수(HEADLESS, LIVESTREAM, ENABLE_CAMERAS 등)와 args_cli에 포함된 인수를 조합하여 최종 실행 설정을 결정합니다. 이때, args_cli에 명시된 인수는 환경 변수보다 우선순위를 가집니다.
simulation_app = app_launcher.app
# 이 줄은 AppLauncher 객체의 app 속성을 통해 실행된 SimulationApp 인스턴스를 가져옵니다.
# 이 단계에서 Isaac Sim 애플리케이션이 실제로 실행되며, 설정된 모드(헤드리스, 라이브스트리밍 등)에 따라 동작합니다.
# AppLauncher가 초기화한 SimulationApp 인스턴스를 가져와, 이후 시뮬레이션 작업에 사용할 수 있도록 합니다.
# 이 객체를 통해 Isaac Sim의 기능을 호출하거나, 시뮬레이션 환경을 조작할 수 있습니다.

from isaaclab.sim import SimulationCfg, SimulationContext
# SimulationContext 물리적 스테핑 및 렌더링과 같은 시뮬레이션 관련 이벤트를 제어하는 클래스입니다.
# 이 클래스는 물리 시뮬레이션 스테핑(physics stepping), 렌더링, 콜백 추가 등 다양한 시뮬레이션 동작을 관리합니다.
# 또한, SimulationCfg, PhysxCfg, RenderCfg와 같은 설정 클래스를 사용하여 시뮬레이션 환경을 구성합니다.
# PyTorch 백엔드: PyTorch 백엔드를 기본으로 사용하며, 모든 데이터 구조는 torch.Tensor 객체로 처리됩니다.
# 시뮬레이터 설정: SimulationCfg 객체를 통해 물리 시간 단계, 서브스텝 수, 물리 솔버 파라미터 등을 설정합니다.
# 시뮬레이션 제어: 시뮬레이션의 재생, 일시정지, 스테핑, 중지 기능을 제공합니다.
# 콜백 관리: 물리 스테핑, 렌더링 등의 이벤트에 콜백을 추가하거나 제거할 수 있습니다.
# 버전 호환성 확인: Isaac Sim 버전 간 호환성을 확인합니다.

# SimulationCfg 시뮬레이션 물리 설정을 정의하는 클래스입니다. 물리 시간 단계, 중력, 장치, Fabric 사용 여부 등을 설정합니다.


def main():
# 이 함수는 프로그램의 진입점(entry point) 역할을 하며, 시뮬레이션 설정과 실행 로직을 포함합니다.
    sim_cfg = SimulationCfg(dt=0.01)
    # 시뮬레이션 설정을 정의합니다.
    # dt=0.01은 물리 시뮬레이션의 시간 단계(time-step)를 초 단위로 설정합니다(즉, 0.01초, 약 100Hz).
    # SimulationCfg의 다른 속성(예: device, gravity, use_fabric)은 기본값으로 설정됩니다. 예를 들어:
    # device="cuda:0" (GPU 디바이스 0번).
    # gravity=(0.0, 0.0, -9.81) (중력 벡터, m/s²).
    # use_fabric=True (Fabric 인터페이스 활성화).
    # render_interval=1 (렌더링 스텝당 물리 스텝 1개).
    # 예시 코드
    # sim_cfg = SimulationCfg(
        # dt=0.01,
        # device="cuda:0",
        # gravity=(0.0, 0.0, -9.81),
        # use_fabric=True,
        # render_interval=1,
        # ...
    # )
    sim = SimulationContext(sim_cfg)
    # 시뮬레이션의 전반적인 제어를 담당하는 클래스입니다. 물리 스테핑, 렌더링, 이벤트 콜백 등을 관리합니다.
    # sim_cfg 객체를 매개변수로 받아 시뮬레이션 환경을 설정합니다. 이 경우, dt=0.01과 같은 설정이 적용됩니다.
    # SimulationContext는 싱글톤(singleton) 객체로, 프로그램 내에서 단일 인스턴스만 존재합니다. 따라서 sim은 시뮬레이션 환경을 제어하는 유일한 객체입니다.
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # set_camera_view 메서드는 뷰포트 카메라의 위치(eye)와 바라보는 타겟 위치(target)를 설정합니다.
    # eye=[2.5, 2.5, 2.5]: 카메라의 위치(x=2.5, y=2.5, z=2.5, 단위는 미터).
    # target=[0.0, 0.0, 0.0]: 카메라가 바라보는 타겟 위치(월드 좌표계의 원점).
    # 뷰포트에서 사용자가 장면을 볼 수 있도록 카메라의 위치와 방향을 설정합니다. GUI가 활성화된 경우, 이 설정에 따라 뷰포트에 표시되는 장면이 결정됩니다.
    sim.reset()
    # reset() 메서드는 시뮬레이션 상태를 초기 상태로 되돌립니다. 이는 물리 엔진, 객체 상태, 시간 등을 리셋하여 시뮬레이션이 시작점에서 다시 시작할 수 있도록 합니다.
    # 시뮬레이션의 모든 액터(예: 리지드 바디, 관절 등)의 초기 위치, 속도, 힘 등을 설정값으로 되돌립니다.
    # 이 메서드는 시뮬레이션을 시작하기 전에 호출되어, 모든 설정(예: sim_cfg의 dt, gravity 등)이 올바르게 적용되었는지 확인합니다.
    print("[INFO]: Setup complete...")
    while simulation_app.is_running():
    # simulation_app은 AppLauncher를 통해 초기화된 isaacsim.SimulationApp 인스턴스로, Isaac Sim 애플리케이션의 실행 상태를 나타냅니다.
    # is_running() 메서드는 Isaac Sim이 현재 실행 중인지 확인합니다. 예를 들어, GUI에서 창을 닫거나 프로그램이 종료되면 False를 반환하여 루프가 종료됩니다.
    # 이 while 루프는 시뮬레이션이 계속 실행되도록 하며, 내부에서 시뮬레이션 스텝을 반복적으로 호출합니다.
        sim.step()
        # 설명: 시뮬레이션을 한 단계 진행합니다.
        # SimulationContext의 step(render=True) 메서드를 호출하여 물리 시뮬레이션을 한 단계(즉, dt=0.01초) 진행합니다.
        # 기본적으로 render=True로 설정되어 있어, 물리 시뮬레이션 후 장면을 렌더링합니다. 이는 뷰포트, 카메라, UI 요소 등을 업데이트합니다.
        # 물리 시뮬레이션은 설정된 dt(0.01초)에 따라 물리 상태(예: 객체 위치, 속도, 충돌)를 업데이트하며, 렌더링은 설정된 카메라 뷰(예: eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])를 기준으로 장면을 표시합니다.

if __name__ == "__main__":
# 이 조건문은 Python 스크립트가 직접 실행되는지, 아니면 모듈로 임포트되는지 확인하는 관용적인 구문입니다.
# __name__은 Python의 내장 변수로, 스크립트가 실행되는 방식에 따라 값을 가집니다:
# 스크립트가 직접 실행될 때(예: python script.py), __name__은 "__main__"으로 설정됩니다.
# 스크립트가 다른 파일에서 모듈로 임포트될 때(예: import script), __name__은 모듈 이름(예: script)으로 설정됩니다.
    main()
    # main() 함수를 호출하여 시뮬레이션의 주요 로직을 실행합니다.
    # sim_cfg = SimulationCfg(dt=0.01): 시뮬레이션 설정(시간 단계 0.01초)을 구성.
    # sim = SimulationContext(sim_cfg): 시뮬레이션 컨텍스트를 초기화.
    # sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0]): 카메라 뷰를 설정.
    # sim.reset(): 시뮬레이션 상태를 초기화.
    # print("[INFO]: Setup complete..."): 설정 완료 메시지 출력.
    # while simulation_app.is_running(): sim.step(): 시뮬레이션을 반복적으로 진행.
    simulation_app.close()
    # close() 메서드는 Isaac Sim 애플리케이션을 정상적으로 종료합니다. 이는 시뮬레이션 리소스를 정리하고, 열린 창(GUI 모드인 경우)이나 백그라운드 프로세스를 닫으며, 메모리와 시스템 자원을 해제합니다.
