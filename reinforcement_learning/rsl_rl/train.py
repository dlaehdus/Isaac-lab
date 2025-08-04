import argparse
# argparse는 명령줄 인수(CLI arguments)를 파싱하는 Python 표준 라이브러리이다.
# 이 모듈을 사용해 스크립트가 사용자 입력(예: --task, --seed)을 처리할 수 있게 한다.
# 영향으로는 후속 코드에서 parser 객체를 생성하여 학습 파라미터를 커스터마이징한다.
# RL 훈련 연계로는 task 이름이나 seed 같은 인수를 통해 환경 재현성과 학습 설정을 제어하여, 실험의 일관성을 유지한다.
import sys
# sys는 시스템 관련 변수와 함수를 제공하는 Python 표준 라이브러리이다.
# 여기서는 sys.argv를 조작하여 Hydra(설정 관리 도구)와의 호환성을 확보한다.
# 영향으로는 CLI 인수 처리를 간소화하며, Hydra가 제대로 작동하도록 한다.
# RL 훈련 연계로는 스크립트 실행 시 사용자 인수를 환경 config에 반영하여, 다중 GPU 학습이나 비디오 녹화 같은 옵션을 활성화한다.
from isaaclab.app import AppLauncher
# AppLauncher는 Isaac Sim 시뮬레이터를 CLI 인수로起動하는 클래스이다.
# 이 임포트로 시뮬레이터를 백그라운드에서 실행할 수 있게 한다.
# 영향으로는 simulation_app 객체를 생성하여 PhysX 기반 시뮬레이션을 초기화한다.
# RL 훈련 연계로는 시뮬레이터가 RL 환경의 기반이 되어, 로봇 동작과 물리 계산을 가능케 하며, 학습 루프의 필수 전제 조건이다.
import cli_args
# cli_args는 RSL-RL 전용 CLI 인수를 추가하는 로컬 모듈이다
# 이 임포트로 parser에 RSL-RL 관련 인수(예: 학습률)를 추가할 수 있다
# 영향으로는 스크립트의 CLI 인터페이스를 확장하여 RSL-RL 설정을 사용자 정의한다.
# RL 훈련 연계로는 RSL-RL 알고리즘(PPO 등)의 하이퍼파라미터를 CLI로 조정하여, 학습 성능과 안정성을 최적화한다.


parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# ArgumentParser는 CLI 인수를 관리하는 객체로, description은 도움말 출력 시 표시된다. 
# 이 줄로 parser를 초기화하여 인수 추가를 준비한다. 영향으로는 후속 add_argument() 호출의 기반이 된다
# RL 훈련 연계로는 스크립트의 사용자 인터페이스를 정의하여, task(예: Lift-Cube)나 distributed 옵션을 쉽게 지정할 수 있게 한다.
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# 이 인수는 학습 중 비디오 녹화를 활성화하는 플래그이다.
# action="store_true"로 인수가 주어지면 True로 설정된다.
# help는 도움말 설명. 영향으로는 args_cli.video가 True 시 카메라 렌더링을 활성화한다.
# RL 훈련 연계로는 학습 과정을 시각화하여 디버깅(예: 로봇 동작 오류 확인)에 유용하며, 비디오 길이/간격과 연계된다.
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# 비디오 녹화 길이(스텝 수)를 지정한다
# type=int로 입력을 정수로 변환. 영향으로는 비디오 래퍼에서 사용되어 녹화 지속 시간을 제어한다
# RL 훈련 연계로는 긴 학습에서 특정 스텝 수만큼 녹화하여, 정책 성능을 분석할 때 효율적이다.
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# 비디오 녹화 간격(스텝 수)을 지정한다.
# 영향으로는 녹화 트리거(lambda step: step % interval == 0)에 사용되어 자원 낭비를 방지한다.
# L 훈련 연계로는 주기적 녹화로 학습 진행을 모니터링하며, 과도한 저장 공간 사용을 피한다.
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# 병렬 시뮬레이션 환경 수를 지정한다. 기본 None 시 config의 기본값 사용.
# env_cfg.scene.num_envs를 오버라이드하여 학습 속도를 조절한다.
# RL 훈련 연계로는 더 많은 환경으로 배치 크기를 늘려 정책 업데이트를 안정화하고, 학습 속도를 높인다 
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# 학습 task 이름(예: Isaac-Lift-Cube-Franka-v0)을 지정한다. 
# 영향으로는 gym.make()에서 사용되어 특정 환경을 로드한다.
# RL 훈련 연계로는 task를 선택하여 Lift나 Push 같은 매니퓰레이션 학습을 지정하며, config(env_cfg)와 연계된다.
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# 환경 랜덤 시드를 지정한다
# 영향으로는 재현성을 보장하며, 분산 학습 시 랭크별 조정.
# RL 훈련 연계로는 랜덤화(예: 초기 위치)를 제어하여 실험 재현성을 높이고, 정책 학습의 공정성을 유지한다.
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# 학습 반복 횟수를 지정한다. 영향으로는 runner.learn()의 num_learning_iterations에 사용되어 훈련 길이를 결정
# 출처는 argparse. RL 훈련 연계로는 총 학습 스텝을 제어하여 과적합 방지와 충분한 훈련을 균형.
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
# 다중 GPU/노드 학습을 활성화한다
# 영향으로는 device 설정과 버전 확인을 트리거한다. 
# RL 훈련 연계로는 대규모 데이터 처리로 학습 속도를 가속, 고차원 task(예: 시각 입력)에 필수.
cli_args.add_rsl_rl_args(parser)
# RSL-RL 전용 인수(예: learning_rate, gamma)를 parser에 추가한다.
# 영향으로는 agent_cfg를 업데이트하여 알고리즘 파라미터를 사용자 정의. 
# RL 훈련 연계로는 PPO 하이퍼파라미터를 조정하여 보상 최대화와 안정적 학습을 달성.
AppLauncher.add_app_launcher_args(parser)
# 시뮬레이터 관련 인수(예: --device)를 추가한다. 
# 영향으로는 app_launcher 초기화에 사용. 
# RL 훈련 연계로는 GPU device 지정으로 시뮬레이션 성능을 최적화.
args_cli, hydra_args = parser.parse_known_args()
# CLI 인수를 파싱하며, Hydra 인수를 분리한다.
# 영향으로는 args_cli를 사용해 config 오버라이드.
# RL 훈련 연계로는 사용자 입력을 config에 반영하여 유연한 실험 설정.
if args_cli.video:
    args_cli.enable_cameras = True
    # 비디오 녹화 시 카메라를 활성화한다. 영향으로는 환경 렌더링을 켜서 RGB 배열 생성.
    # RL 훈련 연계로는 학습 시각화로 정책 진단.
sys.argv = [sys.argv[0]] + hydra_args
# Hydra가 CLI 인수를 제대로 처리하도록 함. 
# 영향으로는 Hydra 초기화 방해 방지. 출처는 sys 모듈. RL 훈련 연계로는 config 로딩을 안정화.
app_launcher = AppLauncher(args_cli)
# 시뮬레이터를 CLI 인수로 설정. 영향으로는 simulation_app 생성. 출처는 isaaclab.app. RL 훈련 연계로는 PhysX 엔진 시작.
simulation_app = app_launcher.app
# 시뮬레이터 핸들을 저장. 영향으로는 후속 close()에 사용. 출처는 AppLauncher. RL 훈련 연계로는 시뮬레이션 관리.


import importlib.metadata as metadata
# importlib.metadata는 설치된 패키지의 메타데이터(예: 버전 정보)를 조회하는 Python 표준 라이브러리이다.
# 이 모듈을 임포트함으로써 설치된 RSL-RL 라이브러리의 버전을 확인할 수 있다. 
# 영향으로는 후속 코드에서 metadata.version()을 호출하여 설치된 버전을 가져오는 데 사용된다.
# RL 훈련 연계로는 라이브러리 호환성을 확인하여 분산 학습이 안정적으로 실행되도록 보장한다
# 호환되지 않는 버전은 학습 오류나 성능 저하를 초래할 수 있기 때문에 이 단계는 중요하다.
import platform
# platform은 운영 체제(OS) 정보를 제공하는 Python 표준 라이브러리이다.
# 이 모듈을 통해 현재 시스템이 Windows인지 Linux인지 등을 확인할 수 있다.
# 영향으로는 설치 명령을 OS에 맞게 조정하는 데 사용된다(예: Windows에서는 .bat 파일, Linux에서는 .sh 파일). 
# RL 훈련 연계로는 크로스-플랫폼 지원을 가능하게 하여, 다양한 환경에서 학습 스크립트를 실행할 수 있도록 돕는다.
from packaging import version
# packaging.version은 버전 문자열을 파싱하고 비교하는 기능을 제공하는 외부 라이브러리이다(보통 pip로 설치됨).
# 이를 통해 버전 번호(예: "2.3.1")를 비교 가능한 객체로 변환하여 설치된 버전과 요구 버전을 비교한다
# 영향으로는 정확한 버전 비교를 가능하게 하여, 요구 버전 미달 여부를 판단한다.
# RL 훈련 연계로는 RSL-RL의 최소 버전 요구 사항을 충족하는지 확인하여, 분산 학습 시 호환성 문제를 방지한다. 이는 학습 프레임워크의 안정성과 성능을 보장한다.



RSL_RL_VERSION = "2.3.1"
# 이 변수는 분산 학습에 필요한 RSL-RL 라이브러리의 최소 요구 버전을 정의한다.
# "2.3.1"은 코드 작성 시점에서 호환성이 검증된 버전으로 설정된 값이다. 
# 영향으로는 이 값이 비교 기준이 되어 설치된 버전이 이 값보다 낮으면 설치 안내가 트리거된다.
# RL 훈련 연계로는 분산 학습이 특정 버전 이상에서만 제대로 작동하므로, 이 값은 학습 환경의 안정성을 보장하는 기준점 역할을 한다.
installed_version = metadata.version("rsl-rl-lib")
# 이 줄은 현재 시스템에 설치된 RSL-RL 라이브러리의 버전 문자열(예: "2.2.0")을 가져온다. 
# rsl-rl-lib는 RSL-RL 프레임워크의 패키지 이름이다.
# 영향으로는 이 값이 요구 버전(RSL_RL_VERSION)과 비교되어 호환성 여부를 판단한다
# RL 훈련 연계로는 설치된 버전을 확인함으로써 분산 학습에 필요한 기능(예: 다중 GPU 지원)이 사용 가능한지 판단한다. 
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
# 이 조건문은 분산 학습(--distributed 옵션 활성화) 시에만 버전 검사를 수행하며, 설치된 버전이 최소 요구 버전("2.3.1")보다 낮은 경우 후속 코드(설치 안내 및 종료)를 실행한다.
# version.parse()는 버전 문자열을 비교 가능한 객체로 변환한다
# 영향으로는 호환되지 않는 버전 사용을 방지하고, 사용자에게 문제를 알린다. 
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    # 이 조건문은 OS에 따라 적절한 실행 스크립트를 선택하여 RSL-RL 라이브러리의 요구 버전을 설치하는 pip 명령을 생성한다.
    # Windows에서는 .bat 파일(배치 파일), Linux/Unix에서는 .sh 파일(셸 스크립트)을 사용한다.
    # cmd 리스트는 후속 print 문에서 사용자에게 제공될 설치 명령어이다.
    # 영향으로는 사용자에게 OS별 맞춤 설치 명령을 제공한다. 
    # RL 훈련 연계로는 모든 환경에서 동일한 버전을 설치하여 분산 학습의 호환성을 보장한다.
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)


import gymnasium as gym
# Gymnasium은 강화학습 환경의 표준 인터페이스를 제공하는 라이브러리이다.
# 이 임포트로 gym.make() 같은 함수를 사용하여 RL 환경을 생성할 수 있다.
# 영향으로는 후속 코드에서 Isaac Lab 환경을 Gym 스타일로 래핑하여 RSL-RL과 호환되게 한다
# RL 훈련 연계로는 환경을 표준화하여 관찰(observation), 액션(action), 보상(reward), 종료(done) 루프를 구현한다.
# 이는 Lift-Cube task 같은 환경을 학습 가능한 형태로 변환한다.
import os
# os는 운영 체제와 상호작용하는 Python 표준 라이브러리(예: 파일 경로 조작, 디렉토리 생성)이다. 
# 이 임포트로 로그 디렉토리 생성이나 파일 저장을 처리할 수 있다
# 영향으로는 log_dir 같은 경로를 안전하게 관리하여 실험 결과를 저장한다.
# RL 훈련 연계로는 학습 로그(체크포인트, config 파일)를 파일 시스템에 저장하여 실험 재현성과 분석을 가능하게 한다
import torch
# Torch는 PyTorch 라이브러리의 핵심 모듈로, 텐서 연산과 신경망을 지원한다.
# 이 임포트로 정책 네트워크 학습과 텐서 조작이 가능하다
# 영향으로는 후속 Torch 백엔드 설정(예: allow_tf32)과 RSL-RL 러너에서 사용된다
# RL 훈련 연계로는 RL 정책(Actor-Critic 네트워크)을 GPU에서 학습하여, 대규모 데이터 처리와 gradient descent를 효율적으로 수행한다
from datetime import datetime
# datetime은 날짜와 시간을 처리하는 Python 표준 라이브러리 클래스이다
# 임포트로 현재 시간을 문자열로 변환할 수 있다.
# 영향으로는 log_dir 생성 시 타임스탬프(예: "%Y-%m-%d_%H-%M-%S")를 추가하여 고유한 로그 폴더를 만든다
# RL 훈련 연계로는 각 학습 실행을 시간별로 구분하여, 여러 실험의 로그를 관리한다. 이는 실험 추적과 재현성을 높인다.
from rsl_rl.runners import OnPolicyRunner
# OnPolicyRunner는 RSL-RL 라이브러리의 학습 러너 클래스(On-Policy 알고리즘용)로, PPO 같은 방법을 지원한다.
# 이 임포트로 runner 객체를 생성하여 학습 루프를 실행할 수 있다. 영향으로는 환경과 에이전트를 연결하여 훈련을 관리한다.
# RL 훈련 연계로는 On-Policy 학습(현재 정책으로 데이터 수집 후 업데이트)을 구현하여, Lift task 같은 환경에서 보상을 최대화한다.
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# 이들은 Isaac Lab의 환경 클래스와 config(DirectMARLEnv: 다중 에이전트 환경, DirectRLEnvCfg: 단일 에이전트 config 등)이다.
# multi_agent_to_single_agent는 다중 에이전트 환경을 단일로 변환한다. 
# 영향으로는 env 생성 시 적합한 config를 선택한다
# RL 훈련 연계로는 Manager-Based 또는 Direct Workflow를 지원하여, 로봇팔 task를 단일/다중 에이전트로 학습한다.
from isaaclab.utils.dict import print_dict
# print_dict는 딕셔너리를 계층적으로 출력하는 유틸리티 함수이다
# 이 임포트로 config나 kwargs를 디버깅 출력할 수 있다.
# 영향으로는 비디오 kwargs 같은 딕셔너리를 nesting으로 예쁘게 출력한다.
# RL 훈련 연계로는 설정 값을 로그하여 실험 디버깅과 재현성을 돕는다.
from isaaclab.utils.io import dump_pickle, dump_yaml
# dump_pickle은 객체를 Pickle 형식으로 저장, dump_yaml은 YAML로 저장하는 함수이다.
# 이 임포트로 config를 파일로 덤프할 수 있다.
# 영향으로는 로그 디렉토리에 env/agent config를 저장한다.
# RL 훈련 연계로는 실험 config를 영속화하여 나중에 로드하거나 분석할 수 있게 한다.
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# RslRlOnPolicyRunnerCfg는 RSL-RL 러너의 config 클래스, RslRlVecEnvWrapper는 환경을 벡터화 래핑하는 클래스이다
# 이 임포트로 agent_cfg와 env 래퍼를 사용한다. 영향으로는 RSL-RL과 Isaac Lab의 통합을 가능케 한다.
# RL 훈련 연계로는 벡터 환경으로 병렬 학습을 지원하고, config로 러너를 설정한다.
import isaaclab_tasks
# isaaclab_tasks는 Isaac Lab의 등록된 task(예: Lift-Cube)를 포함하는 모듈이다.
# 이 임포트로 task를 gym.make()에서 사용할 수 있다.
# 영향으로는 환경 생성에 필요
from isaaclab_tasks.utils import get_checkpoint_path
# get_checkpoint_path는 로그 디렉토리에서 체크포인트 경로를 가져오는 함수이다.
# 이 임포트로 resume 시 이전 모델을 로드한다.
from isaaclab_tasks.utils.hydra import hydra_task_config


torch.backends.cuda.matmul.allow_tf32 = True
# TF32(TensorFloat-32)는 FP32의 정밀도를 유지하면서 FP16의 속도를 제공하는 형식으로, CUDA 연산을 가속화한다.
# 목적은 학습 속도를 높이는 데 있으며, 영향으로는 GPU 메모리 사용을 줄이고 throughput를 증가시킨다
# RL 훈련 연계로는 대규모 정책 네트워크 업데이트 시 계산 효율성을 높여, 다중 환경 학습에서 시간을 단축한다.
torch.backends.cudnn.allow_tf32 = True
# cuDNN은 CUDA 기반 딥러닝 연산 라이브러리로, TF32를 활성화하여 컨볼루션과 행렬 연산을 최적화한다.
# 목적은 Torch의 백엔드 성능을 향상시키는 데 있으며, 영향으로는 CNN 기반 정책 네트워크(예: 시각 입력 task)에서 속도 향상.
torch.backends.cudnn.deterministic = False
# deterministic=True 시 동일 입력에 항상 동일 결과를 보장하나 속도가 느려진다.
# False로 설정하면 최적화로 인해 약간의 변동이 있지만 속도가 빨라진다
torch.backends.cudnn.benchmark = False
# benchmark=True 시 입력 크기에 따라 최적 커널을 자동 선택하나, 초기 오버헤드가 있다. 
# False로 설정하면 고정 커널 사용으로 안정적이지만 약간 느릴 수 있다.

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
# @hydra_task_config는 Hydra가 task 이름(args_cli.task)으로 config를 로드하고, "rsl_rl_cfg_entry_point"는 entry point를 지정한다. 
# 목적은 config를 동적으로 관리하며, 영향으로는 env_cfg(환경 설정)와 agent_cfg(RSL-RL 러너 config)를 자동 제공.
# RL 훈련 연계로는 task별 config를 로드하여 Lift-Cube 같은 환경을 커스터마이징한다.
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # CLI 인수를 agent_cfg에 반영한다. 목적은 사용자 오버라이드이며, 영향으로는 seed나 device 같은 값 변경.
    # 출처는 cli_args 모듈. RL 훈련 연계로는 실험 변수를 동적으로 조정하여 최적 하이퍼파라미터 탐색을 돕는다.
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # 병렬 환경 수를 설정한다. 목적은 CLI로 규모 조정이며, 영향으로는 학습 배치 크기 결정.
    agent_cfg.max_iterations = (args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations)
    # 학습 반복 횟수를 설정한다. 목적은 훈련 길이 제어이며, 영향으로는 총 스텝 수 결정. 출처는 사용자 로직. RL 훈련 연계로는 과적합 방지와 충분한 학습 균형.
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
