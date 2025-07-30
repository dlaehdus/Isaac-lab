import math
# Python의 표준 math 모듈을 가져옵니다. 이 모듈은 수학 연산을 제공하며, 여기서는 math.pi를 사용하여 폴의 각도와 각속도 범위를 설정합니다.
import isaaclab.sim as sim_utils
# Isaac Lab의 시뮬레이션 유틸리티 모듈을 가져옵니다. sim_utils는 물리 엔진, 렌더링, 자산 생성(예: 지면, 조명)을 위한 도구를 제공합니다.
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# ArticulationCfg: 관절이 있는 로봇(예: Cartpole)의 설정을 정의.
# AssetBaseCfg: 기본 자산(예: 지면, 조명)의 설정을 정의.
from isaaclab.envs import ManagerBasedRLEnvCfg
# RL 환경을 위한 설정 클래스. ManagerBasedEnvCfg를 확장하여 보상, 종료 조건, 커리큘럼, 명령 등을 추가합니다.
from isaaclab.managers import EventTermCfg as EventTerm
# EventTermCfg: 이벤트(예: 리셋, 초기화) 설정.
from isaaclab.managers import ObservationGroupCfg as ObsGroup
# ObservationGroupCfg: 관찰 그룹(예: "policy") 설정.
from isaaclab.managers import ObservationTermCfg as ObsTerm
# ObservationTermCfg: 개별 관찰 항목(예: 관절 위치) 설정.
from isaaclab.managers import RewardTermCfg as RewTerm
# RewardTermCfg: 보상 항목 설정.
from isaaclab.managers import SceneEntityCfg
# SceneEntityCfg: 장면 내 엔티티(예: 로봇의 특정 관절) 지정.
from isaaclab.managers import TerminationTermCfg as DoneTerm
# TerminationTermCfg: 종료 조건 설정.
from isaaclab.scene import InteractiveSceneCfg
# 다중 환경을 지원하는 장면 설정 클래스. 환경의 물리적 구성(예: 로봇, 지면)을 정의합니다.
from isaaclab.utils import configclass
# 설정 클래스를 정의하기 위한 데코레이터. 데이터 클래스처럼 동작하며, 속성을 체계적으로 관리합니다.
import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
# Cartpole 작업에 특화된 관찰, 보상, 종료 함수를 포함한 모듈.
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
# Cartpole 로봇의 기본 설정. 로봇의 물리적 속성, 관절, 프림 경로 등을 정의.

# 장면 설정 (CartpoleSceneCfg)
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
# CartpoleSceneCfg를 설정 클래스로 정의하며, InteractiveSceneCfg를 상속받아 다중 환경을 지원하는 장면을 구성합니다.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

# 동작 설정 (ActionsCfg)
@configclass
class ActionsCfg:
# ActionsCfg: 환경의 동작을 정의.
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    # joint_effort: JointEffortActionCfg를 사용해 Cartpole의 카트에 적용할 힘을 설정.
    # asset_name="robot": Cartpole 로봇을 대상으로 지정.
    # joint_names=["slider_to_cart"]: 카트의 슬라이더 관절에 힘 적용.
    # scale=100.0: 입력된 힘에 100을 곱하여 스케일링.

# 관찰 설정 (ObservationsCfg)
@configclass
class ObservationsCfg:
# ObservationsCfg: 환경의 관찰 공간을 정의.
    @configclass
    class PolicyCfg(ObsGroup):
    # PolicyCfg: ObsGroup을 상속받아 "policy" 관찰 그룹을 정의
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # joint_pos_rel: mdp.joint_pos_rel로 관절의 상대적 위치(카트 위치, 폴 각도) 계산.
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # joint_vel_rel: mdp.joint_vel_rel로 관절의 상대적 속도(카트 속도, 폴 각속도) 계산.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            # enable_corruption=False: 관찰에 노이즈 추가 비활성화.
            self.concatenate_terms = True
            # concatenate_terms=True: 위치와 속도를 하나의 텐서로 결합.
    policy: PolicyCfg = PolicyCfg()
    # "policy" 그룹을 정의. RL 에이전트가 사용할 관찰 공간.

# 이벤트 설정 (EventCfg)
@configclass
class EventCfg:
    reset_cart_position = EventTerm(
    # reset_cart_position: 카트의 관절 위치와 속도를 리셋.
        func=mdp.reset_joints_by_offset,
        # func=mdp.reset_joints_by_offset: 관절 위치와 속도를 무작위로 설정.
        mode="reset",
        # mode="reset": 환경 리셋 시 실행.
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            # asset_cfg: "slider_to_cart" 관절 대상.
            "position_range": (-1.0, 1.0),
            # position_range=(-1.0, 1.0): 카트 위치를 ±1.0m 범위에서 무작위 설정.
            "velocity_range": (-0.5, 0.5),
            # velocity_range=(-0.5, 0.5): 카트 속도를 ±0.5m/s 범위에서 설정.
        },
    )

    reset_pole_position = EventTerm(
    # 폴의 관절 위치와 속도를 리셋.
        func=mdp.reset_joints_by_offset,
        # 관절 위치와 속도를 무작위로 설정.
        mode="reset",
        # 환경 리셋 시 실행.
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            # "cart_to_pole" 관절 대상.
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            # 폴 각도를 ±14.32도(0.25π 라디안) 범위에서 설정.
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
            # 폴 각속도를 ±0.785 rad/s 범위에서 설정.
        },
    )

# 보상 설정 (RewardsCfg)
@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # alive: 에이전트가 살아있는 동안 보상을 부여.
    # func=mdp.is_alive: 환경이 종료되지 않은 경우 참 반환.
    # weight=1.0: 생존 시 +1.0 보상.
    # 역할: 에이전트가 가능한 오래 폴을 균형 있게 유지하도록 유도.
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # terminating: 환경 종료 시 패널티 부여.
    # func=mdp.is_terminated: 종료 조건이 충족된 경우 참 반환.
    # weight=-2.0: 종료 시 -2.0 패널티.
    # 역할: 에이전트가 종료 조건(예: 카트가 범위를 벗어남)을 피하도록 유도.
    pole_pos = RewTerm(
    # pole_pos: 폴의 각도가 목표(0도, 수직)에서 벗어난 정도에 패널티.
        func=mdp.joint_pos_target_l2,
        # func=mdp.joint_pos_target_l2: 관절 각도와 목표 간 L2 오차 계산.
        weight=-1.0,
        # weight=-1.0: 오차에 -1.0 가중치 적용.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
        # params: "cart_to_pole" 관절, 목표 각도 0도.
    )
    cart_vel = RewTerm(
    # cart_vel: 카트 속도에 비례한 패널티.
        func=mdp.joint_vel_l1,
        # func=mdp.joint_vel_l1: 속도의 L1 노름 계산.
        weight=-0.01,
        # weight=-0.01: 속도에 -0.01 가중치.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
        # params: "slider_to_cart" 관절.
        # 역할: 카트의 이동 속도를 최소화.
    )
    pole_vel = RewTerm(
    # pole_vel: 폴 각속도에 비례한 패널티.
        func=mdp.joint_vel_l1,
        # func=mdp.joint_vel_l1: 각속도의 L1 노름 계산
        weight=-0.005,
        # weight=-0.005: 각속도에 -0.005 가중치.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
        # params: "cart_to_pole" 관절.
        # 역할: 폴의 회전 속도를 최소화.
    )

# 종료 조건 (TerminationsCfg)
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # time_out: 최대 에피소드 시간 초과 시 종료.
    # func=mdp.time_out: 에피소드 길이가 episode_length_s를 초과하면 참 반환.
    # time_out=True: 시간 초과는 중단(truncation)으로 간주.
    # 에피소드가 5초(episode_length_s)를 초과하면 종료.
    cart_out_of_bounds = DoneTerm(
    # cart_out_of_bounds: 카트 위치가 ±3.0m 범위를 벗어나면 종료.
        func=mdp.joint_pos_out_of_manual_limit,
        # 관절 위치가 지정된 범위를 벗어나는지 확인.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
        # "slider_to_cart" 관절, 범위 (-3.0, 3.0).
    )
    
# 환경 설정 (CartpoleEnvCfg)
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
# CartpoleEnvCfg: RL 환경을 정의하며, ManagerBasedRLEnvCfg를 상속.
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    # scene: CartpoleSceneCfg를 사용해 장면 설정.
    # num_envs=4096: 기본적으로 4096개의 병렬 환경.
    # env_spacing=4.0: 환경 간 거리 4.0m.
    # num_envs는 실행 시 args_cli.num_envs로 재설정 가능(기본값 16).
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # observations, actions, events, rewards, terminations: 위에서 정의한 설정 클래스를 사용.

    def __post_init__(self) -> None:
    # 설정 초기화 후 호출.
        self.decimation = 2
        # 시뮬레이션 스텝마다 환경을 2번 업데이트(120Hz / 2 = 60Hz).
        self.episode_length_s = 5
        # 최대 에피소드 길이 5초.
        self.viewer.eye = (8.0, 0.0, 5.0)
        # 카메라 위치.
        self.sim.dt = 1 / 120
        # 시뮬레이션 시간 간격 8.33ms(120Hz).
        self.sim.render_interval = self.decimation
        # 렌더링을 2 스텝마다 수행.
