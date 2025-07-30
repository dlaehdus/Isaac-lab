from __future__ import annotations
# Python의 미래 기능으로, 타입 힌트를 사용할 때 순환 참조를 허용합니다. 
# 예를 들어, 클래스 내에서 자신의 타입(CartpoleEnvCfg)을 참조할 수 있습니다.
import math
import torch
from collections.abc import Sequence
# Python의 표준 라이브러리로, 시퀀스 타입(리스트, 튜플 등)을 추상적으로 처리합니다.
# 용도: env_ids: Sequence[int] | None과 같은 타입 힌트에 사용.
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
# DirectRLEnv: RL 환경 클래스. 매니저를 사용하지 않고 직접 함수를 구현.
# DirectRLEnvCfg: DirectRLEnv를 위한 설정 클래스.
from isaaclab.scene import InteractiveSceneCfg
# 설명: 다중 환경을 지원하는 장면 설정 클래스.
# 용도: CartpoleEnvCfg에서 장면 구성.
from isaaclab.sim import SimulationCfg
# 설명: 시뮬레이션 설정 클래스(예: 시간 간격, 렌더링 주기).
# 용도: CartpoleEnvCfg에서 시뮬레이션 파라미터 설정.
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
# GroundPlaneCfg: 지면 설정 클래스.
# spawn_ground_plane: 지면을 시뮬레이션에 추가하는 함수.
from isaaclab.utils import configclass
# 설명: 설정 클래스를 정의하기 위한 데코레이터.
# 용도: CartpoleEnvCfg와 같은 설정 클래스를 정의.
from isaaclab.utils.math import sample_uniform
# 설명: 균일 분포에서 무작위 샘플을 생성하는 함수.
# 용도: 리셋 시 폴 각도를 무작위로 설정.

# 환경 설정 (CartpoleEnvCfg)
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
# 설명: CartpoleEnvCfg를 설정 클래스로 정의하며, DirectRLEnvCfg를 상속받아 RL 환경 설정을 구성.
# 역할: 시뮬레이션, 로봇, 장면, 리셋, 보상 스케일 등을 정의.
    decimation = 2
    # 설명: 시뮬레이션 스텝마다 환경을 2번 업데이트(120Hz / 2 = 60Hz).
    # 용도: 물리 스텝과 RL 스텝의 비율 조정.
    episode_length_s = 5.0
    # 설명: 최대 에피소드 길이를 5초로 설정.
    # 용도: 시간 초과 종료 조건(time_out)에 사용.
    action_scale = 100.0
    # 설명: 동작(힘)에 적용할 스케일링 인자. 입력된 동작에 100을 곱함.
    # 용도: 카트에 적용되는 힘의 크기 조정.
    action_space = 1
    # 설명: 동작 공간의 차원 수. Cartpole은 단일 동작(카트에 가해지는 힘)을 가짐.
    # 용도: RL 에이전트의 동작 차원 정의.
    observation_space = 4
    # 설명: 관찰 공간의 차원 수. Cartpole은 폴 각도, 폴 각속도, 카트 위치, 카트 속도(4차원)를 관찰.
    # 용도: RL 에이전트의 관찰 차원 정의.
    state_space = 0
    # 설명: 상태 공간의 차원 수. 비대칭 정책(actor-critic)에서 critic이 추가 상태를 사용할 수 있지만, 여기서는 사용 안 함.
    # 용도: 상태 공간 정의(현재는 비활성화).
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # sim: 시뮬레이션 설정.
    # dt=1/120: 시간 간격 8.33ms(120Hz).
    # render_interval=decimation: 렌더링을 2 스텝마다 수행(60Hz).
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # 설명: Cartpole 로봇 설정. CARTPOLE_CFG를 가져와 프림 경로를 {ENV_REGEX_NS}/Robot으로 재설정.
    # 용도: 다중 환경에서 로봇 배치(예: /World/envs/env_0/Robot).
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # 설명: 카트와 폴의 관절 이름을 정의.
    # 용도: 동작과 관찰에서 특정 관절을 참조.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    # scene: 장면 설정.
    # num_envs=4096: 기본적으로 4096개 병렬 환경.
    # env_spacing=4.0: 환경 간 거리 4.0m.
    # replicate_physics=True: 각 환경의 물리 시뮬레이션 독립.
    max_cart_pos = 3.0
    # 설명: 카트의 최대 위치 제한(±3.0m). 이를 초과하면 환경 종료.
    # 용도: 종료 조건(out_of_bounds)에 사용.
    initial_pole_angle_range = [-0.25, 0.25]
    # 설명: 리셋 시 폴의 초기 각도 범위(±0.25π 라디안, 약 ±14.32도).
    # 용도: 리셋 시 폴 각도를 무작위로 설정.
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # rew_scale_alive: 생존 보상(+1.0).
    # rew_scale_terminated: 종료 패널티(-2.0).
    # rew_scale_pole_pos: 폴 각도 오차 패널티(-1.0).
    # rew_scale_cart_vel: 카트 속도 패널티(-0.01).
    # rew_scale_pole_vel: 폴 각속도 패널티(-0.005).

# 환경 클래스 (CartpoleEnv)
class CartpoleEnv(DirectRLEnv):
# 설명: CartpoleEnv는 DirectRLEnv를 상속받아 RL 환경을 구현. cfg는 환경 설정을 저장.
# 역할: 장면 설정, 동작 적용, 관찰, 보상, 종료 조건, 리셋을 직접 구현.
    cfg: CartpoleEnvCfg
    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 설명: 초기화 함수. DirectRLEnv의 생성자를 호출하여 기본 환경을 설정.
        # cfg: 환경 설정.
        # render_mode: 렌더링 모드(예: "human", None).
        # **kwargs: 추가 인자.
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        # 설명: Cartpole 로봇의 관절 인덱스를 가져옴.
        # self._cart_dof_idx: "slider_to_cart" 관절의 인덱스.
        # self._pole_dof_idx: "cart_to_pole" 관절의 인덱스.
        self.action_scale = self.cfg.action_scale
        # 설명: 동작 스케일링 인자(100.0)를 클래스 변수로 저장.
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel
        # 설명: Cartpole의 관절 위치와 속도를 클래스 변수로 저장.
        # 용도: 관찰, 보상, 종료 조건 계산에서 사용.

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # 설명: Cartpole 로봇을 Articulation 객체로 생성.
        # 용도: 시뮬레이션에 로봇 추가.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 설명: 지면을 /World/ground에 추가.
        self.scene.clone_environments(copy_from_source=False)
        # 환경을 복제하여 num_envs개의 병렬 환경 생성. copy_from_source=False는 빈 환경을 생성.
        if self.device == "cpu":
        # 설명: CPU 시뮬레이션 시 환경 간 충돌을 방지.
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["cartpole"] = self.cartpole
        # 설명: Cartpole 로봇을 장면에 추가.
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # 설명: 조명을 /World/Light에 추가(세기 2000, 색상 회색).

    # 동작 처리 (_pre_physics_step, _apply_action)
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        # 설명: RL 스텝 전에 동작을 처리. 입력 동작을 복사하고 action_scale(100.0)을 곱함.
        # 용도: 동작을 스케일링하여 저장.
    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
        # 설명: 물리 스텝마다 스케일링된 동작을 카트 관절에 적용.
        # 용도: 카트에 힘 적용.

    # 관찰 계산 (_get_observations)
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    # 설명: 관찰을 계산하여 반환.
    # torch.cat: 폴 각도, 폴 각속도, 카트 위치, 카트 속도를 4차원 텐서로 결합.
    # observations = {"policy": obs}: 관찰을 "policy" 키로 딕셔너리에 저장.
    # 용도: RL 에이전트가 사용할 관찰 제공.

    # 보상 계산 (_get_rewards, compute_rewards)
    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    # 종료 조건 (_get_dones)
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    # 설명: 최신 관절 위치와 속도를 가져옴.
    # 용도: 종료 조건 계산에 사용.
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    # 리셋 (_reset_idx)
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
