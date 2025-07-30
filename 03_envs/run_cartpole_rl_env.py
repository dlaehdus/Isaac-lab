import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
# PyTorch 라이브러리를 가져옵니다. 텐서 연산과 GPU 가속을 지원하며, RL 환경에서 관찰, 동작, 보상을 처리하는 데 사용됩니다.
from isaaclab.envs import ManagerBasedRLEnv
# RL 작업을 위한 환경 클래스. ManagerBasedEnv를 확장하여 보상, 종료 조건, 커리큘럼, 명령 등을 지원합니다.
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
# Cartpole 환경의 설정 클래스. 장면(CartpoleSceneCfg), 동작, 관찰, 이벤트, 보상, 종료 조건 등을 정의합니다.

def main():
    env_cfg = CartpoleEnvCfg()
    # CartpoleEnvCfg 객체를 생성하여 RL 환경 설정을 초기화합니다. 
    # 이 설정은 이전 코드에서 정의된 CartpoleSceneCfg, ActionsCfg, ObservationsCfg, EventCfg, RewardsCfg, TerminationsCfg를 포함합니다.
    # scene: 4096개 환경, 환경 간 거리 4.0m.
    # actions: 카트에 힘 적용(scale=100.0).
    # observations: 카트와 폴의 위치/속도.
    # events: 리셋 시 카트와 폴 상태 무작위화.
    # rewards: 생존, 종료, 폴 각도, 카트/폴 속도.
    # terminations: 시간 초과, 카트 범위 벗어남.
    env_cfg.scene.num_envs = args_cli.num_envs
    # env_cfg.scene.num_envs를 명령줄 인자 args_cli.num_envs로 재설정합니다(기본값 16). 
    # 이는 병렬로 실행할 Cartpole 로봇 수를 결정합니다.
    env_cfg.sim.device = args_cli.device
    # 시뮬레이션 장치를 args_cli.device로 설정합니다(예: cuda 또는 cpu). 
    # 이는 PyTorch 연산에 사용할 장치를 지정합니다.
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # ManagerBasedRLEnv 객체를 생성하여 RL 환경을 초기화합니다. 
    # env_cfg에 정의된 설정을 기반으로 장면, 동작, 관찰, 보상, 종료 조건 등을 구성합니다.

    count = 0
    # 시뮬레이션 스텝 카운터를 초기화합니다.
    while simulation_app.is_running():
    # 시뮬레이션 애플리케이션이 실행 중인 동안 루프를 반복합니다. 사용자가 창을 닫거나 Ctrl+C를 누르면 종료됩니다.
        with torch.inference_mode():
        # PyTorch의 자동 미분(autograd)을 비활성화하여 성능을 최적화합니다. RL 환경에서 계산된 보상 및 관찰에 대한 기울기 계산을 방지합니다.
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            joint_efforts = torch.randn_like(env.action_manager.action)
            # env.action_manager.action의 형태와 동일한 랜덤 힘(정규분포)을 생성합니다. 이는 Cartpole의 slider_to_cart 관절에 적용할 힘입니다
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # 설명: 환경을 한 스텝 진행합니다.
            # joint_efforts: 입력된 랜덤 힘을 적용.
            # obs: 관찰(ObservationsCfg의 joint_pos_rel, joint_vel_rel로 정의된 카트 위치/속도, 폴 각도/각속도).
            # rew: 보상(RewardsCfg의 alive, terminating, pole_pos, cart_vel, pole_vel의 합계).
            # terminated: 종료 조건(TerminationsCfg의 cart_out_of_bounds) 충족 여부.
            # truncated: 시간 초과(time_out)로 인한 중단 여부.
            # info: 추가 정보(보상 항목별 기여도, 종료 상태 등).
            # 역할: RL 환경의 스텝 진행 및 결과 반환.
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            count += 1
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
