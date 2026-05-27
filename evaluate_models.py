import mujoco
import numpy as np
import random
import torch
import torch.nn as nn
import csv

# ==========================================
# 1. 평가용 환경 (Env) - 타겟 고정 기능 추가
# ==========================================
class MujocoEvalEnv:
    def __init__(self):
        xml = """
        <mujoco model="eval_arm">
          <compiler angle="radian"/>
          <option gravity="0 0 -9.81" timestep="0.01"/>
          <worldbody>
            <light pos="0 0 3" dir="0 0 -1" directional="true"/>
            <geom name="floor" type="plane" size="1.5 1.5 0.1" rgba="0.8 0.9 0.8 1"/>
            <body name="base" pos="0 0 0.1">
              <geom type="cylinder" size="0.1 0.05" rgba="0.2 0.2 0.2 1"/>
              <body name="link1" pos="0 0 0.05">
                <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0" damping="1.5" range="-1.5 1.5"/>
                <geom type="capsule" size="0.05 0.2" pos="0 0 0.2" rgba="0.1 0.5 0.8 1"/>
                <body name="link2" pos="0 0 0.4">
                  <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0" damping="1.5" range="-2.0 2.0"/>
                  <geom type="capsule" size="0.04 0.2" pos="0 0 0.2" rgba="0.8 0.5 0.1 1"/>
                  <body name="link3" pos="0 0 0.4">
                    <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0" damping="1.5" range="0.0 2.5"/>
                    <geom name="link3_geom" type="capsule" size="0.03 0.13" pos="0 0 0.13" rgba="0.1 0.8 0.5 1"/>
                    <geom name="end_effector" type="sphere" size="0.04" pos="0 0 0.28" rgba="1 0.8 0.2 1"/>
                    <site name="ee_site" pos="0 0 0.28" size="0.02" rgba="1 0 0 0"/> 
                  </body>
                </body>
              </body>
            </body>
            <body name="target" mocap="true" pos="0 0 -1">
              <geom name="target_geom" type="sphere" size="0.02" rgba="1 0.3 0.3 1"/>
            </body>
          </worldbody>
          <actuator>
            <position joint="joint1" name="ctrl_j1" kp="150" ctrlrange="-1.5 1.5"/>
            <position joint="joint2" name="ctrl_j2" kp="150" ctrlrange="-2.0 2.0"/>
            <position joint="joint3" name="ctrl_j3" kp="150" ctrlrange="0.0 2.5"/>
          </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.target_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")]
        self.max_steps = 300

    def reset(self, fixed_target=None):
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        if fixed_target is not None:
            self.target_pos = fixed_target
        else:
            r = random.uniform(0.3, 0.65)
            theta = random.uniform(-1.0, 1.0)
            self.target_pos = np.array([r * np.cos(theta), r * np.sin(theta), random.uniform(0.15, 0.5)])
            
        self.data.mocap_pos[self.target_mocap_id] = self.target_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos[:3], self.data.site_xpos[self.ee_site_id], self.target_pos]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)
        max_step_size = 0.05 
        target_ctrl = self.data.ctrl[:3] + (action * max_step_size)
        self.data.ctrl[0] = np.clip(target_ctrl[0], -1.5, 1.5)
        self.data.ctrl[1] = np.clip(target_ctrl[1], -2.0, 2.0)
        self.data.ctrl[2] = np.clip(target_ctrl[2], 0.0, 2.5)
        mujoco.mj_step(self.model, self.data)
        
        action_penalty = 0.01 * np.sum(np.square(action))
        reward = -0.05 - action_penalty 
        done = False
        
        if self._check_contact():
            reward += 100.0 
            done = True
        elif self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

    def _check_contact(self):
        id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "end_effector")
        id2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        for i in range(self.data.ncon):
            if (self.data.contact[i].geom1 == id1 and self.data.contact[i].geom2 == id2) or \
               (self.data.contact[i].geom1 == id2 and self.data.contact[i].geom2 == id1):
                return True
        return False

# ==========================================
# 2. 신경망 및 IK 함수 정의
# ==========================================
class SB3ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(9, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return torch.tanh(self.actor(x))

def get_ik_action(env):
    error = env.target_pos - env.data.site_xpos[env.ee_site_id]
    error_norm = np.linalg.norm(error)
    if error_norm > 0.05:
        error = (error / error_norm) * 0.05
    jacp = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, None, env.ee_site_id)
    inv_term = np.linalg.inv(jacp @ jacp.T + (0.1 ** 2) * np.eye(3))
    delta_q = jacp.T @ inv_term @ error * 0.5 
    action = ((env.data.qpos[:3] + delta_q) - env.data.ctrl[:3]) / 0.05
    return np.clip(action, -1.0, 1.0)

# ==========================================
# 3. 100회 에피소드 평가 및 CSV 저장
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MujocoEvalEnv()
    
    # 가중치 불러오기
    model_pure = SB3ActorCritic().to(device)
    model_pure.load_state_dict(torch.load("pure_ppo_final.pth", map_location=device), strict=False)
    model_pure.eval()
    
    model_hybrid = SB3ActorCritic().to(device)
    model_hybrid.load_state_dict(torch.load("hybrid_bc_ppo_final.pth", map_location=device), strict=False)
    model_hybrid.eval()

    models_to_test = ["1_Pure_PPO", "2_Heuristic_IK", "3_Hybrid_PPO_BC"]
    results = []

    print("📊 100개 에피소드에 대한 정량적 평가를 시작합니다...")
    print("-" * 75)
    print(f"{'Ep':^5} | {'Target Pos (x,y,z)':^25} | {'Model':^15} | {'Success':^7} | {'Reward':^7} | {'Steps':^5}")
    print("-" * 75)

    for ep in range(1, 101):
        # 3가지 모델이 '완벽하게 똑같은 타겟'을 목표로 하도록 고정
        r = random.uniform(0.3, 0.65)
        theta = random.uniform(-1.0, 1.0)
        fixed_target = np.array([r * np.cos(theta), r * np.sin(theta), random.uniform(0.15, 0.5)])
        
        for mode in models_to_test:
            obs = env.reset(fixed_target=fixed_target)
            ep_reward = 0
            done = False
            
            while not done:
                if mode == "1_Pure_PPO":
                    with torch.no_grad():
                        action = model_pure(torch.FloatTensor(obs).unsqueeze(0).to(device)).cpu().numpy()[0]
                elif mode == "2_Heuristic_IK":
                    action = get_ik_action(env)
                elif mode == "3_Hybrid_PPO_BC":
                    with torch.no_grad():
                        action = model_hybrid(torch.FloatTensor(obs).unsqueeze(0).to(device)).cpu().numpy()[0]

                obs, reward, done = env.step(action)
                ep_reward += reward
            
            # 보상이 50 이상이면 타겟 성공으로 간주
            success = "Yes" if ep_reward > 50 else "No"
            
            results.append({
                "Episode": ep,
                "Target_X": round(fixed_target[0], 2),
                "Target_Y": round(fixed_target[1], 2),
                "Target_Z": round(fixed_target[2], 2),
                "Model": mode,
                "Success": success,
                "Reward": round(ep_reward, 1),
                "Steps": env.current_step
            })
            
            if ep % 10 == 0 or ep == 1:
                target_str = f"{fixed_target[0]:.2f}, {fixed_target[1]:.2f}, {fixed_target[2]:.2f}"
                print(f"{ep:^5} | {target_str:^25} | {mode:^15} | {success:^7} | {ep_reward:>7.1f} | {env.current_step:^5}")

    # CSV 파일 저장
    csv_filename = "evaluation_results.csv"
    keys = results[0].keys()
    with open(csv_filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    # 요약 통계 출력
    print("-" * 75)
    print("📈 실험 요약 (100 에피소드)")
    for mode in models_to_test:
        mode_results = [r for r in results if r["Model"] == mode]
        success_count = sum(1 for r in mode_results if r["Success"] == "Yes")
        avg_reward = sum(r["Reward"] for r in mode_results) / 100
        print(f"[{mode:^15}] 성공률: {success_count}% | 평균 보상: {avg_reward:.1f}")
    
    print(f"\n💾 전체 실험 데이터가 '{csv_filename}'에 저장되었습니다. 엑셀로 열어 논문 그래프에 활용하세요!")