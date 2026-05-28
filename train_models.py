import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mujoco
import mujoco.viewer
import random
import cv2

# ==========================================
# 1. 강화학습 환경 (진척도 보상 & 시작점 노이즈)
# ==========================================
class PandaPlanarEnv:
    def __init__(self):
        xml = """
        <mujoco>
          <compiler angle="radian"/>
          <include file="scene.xml"/>
          <worldbody>
            <body name="target" pos="0.5 0 0.025">
              <freejoint name="target_joint"/>
              <geom name="target_geom" type="box" size="0.025 0.025 0.025" rgba="0.2 0.8 0.2 1" mass="0.05" friction="5 0.5 0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
        original_cwd = os.getcwd()
        try:
            os.chdir("franka_emika_panda")
            self.model = mujoco.MjModel.from_xml_string(xml)
        finally:
            os.chdir(original_cwd)

        self.data = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.nv = self.model.nv
        self.nu = self.model.nu
        
        self.max_steps = 600
        self.current_step = 0
        
        self.gripper_min = self.model.actuator_ctrlrange[7][0]
        self.gripper_max = self.model.actuator_ctrlrange[7][1]
        
        self.q_home = np.zeros(self.nv)
        self.q_home[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        
        mujoco.mj_forward(self.model, self.data)
        self.fixed_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.fixed_quat, self.data.xmat[self.ee_id])
        
        self.prev_dist = 0.0 # 진척도 보상을 위한 이전 거리 저장 변수
        self.reset()
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # 💡 [해결책 1] 시작 관절에 미세한 노이즈를 주어 맹목적 암기(Overfitting) 차단!
        noise = np.random.uniform(-0.05, 0.05, self.nv)
        noise[7:] = 0.0 # 그리퍼는 정자세 유지
        
        self.data.qpos[:self.nv] = self.q_home.copy() + noise
        self.data.ctrl[:self.nu] = self.q_home[:self.nu].copy() + noise[:self.nu]
        
        # 박스를 랜덤 배치
        tx = random.uniform(0.4, 0.6)
        ty = random.uniform(-0.15, 0.15)
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
        qpos_adr = self.model.jnt_qposadr[jnt_id]
        
        self.data.qpos[qpos_adr:qpos_adr+3] = [tx, ty, 0.025]
        self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
        self.data.ctrl[7:] = self.gripper_max 
        
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        
        # 목표와의 초기 거리 설정
        ee_pos = self.data.xpos[self.ee_id]
        box_pos = self.data.xpos[self.target_body_id]
        target_grasp_pos = np.array([box_pos[0], box_pos[1], box_pos[2] + 0.105])
        self.prev_dist = np.linalg.norm(ee_pos - target_grasp_pos)
        
        return self._get_obs()

    def _get_obs(self):
        q = [self.data.qpos[0], self.data.qpos[1], self.data.qpos[3], self.data.qpos[5]]
        ee_pos = self.data.xpos[self.ee_id]
        box_pos = self.data.xpos[self.target_body_id]
        gripper_pos = self.data.qpos[7]
        
        # 💡 [해결책 2] 상대 거리를 10배 증폭하여 신경망이 절대 무시하지 못하게 강제 주입!
        rel_pos = (box_pos - ee_pos) * 10.0
        return np.concatenate([q, [gripper_pos], ee_pos, box_pos, rel_pos]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)
        
        self.data.ctrl[0] += action[0] * 0.02
        self.data.ctrl[1] += action[1] * 0.02
        self.data.ctrl[3] += action[2] * 0.02
        
        self.data.ctrl[0] = np.clip(self.data.ctrl[0], -2.8973, 2.8973) 
        self.data.ctrl[1] = np.clip(self.data.ctrl[1], -1.0, 1.7628)    
        self.data.ctrl[3] = np.clip(self.data.ctrl[3], -3.0718, -0.07)  
        
        if action[3] < 0.2:
            self.data.ctrl[7:] = self.gripper_min
        else:
            self.data.ctrl[7:] = self.gripper_max

        self.data.ctrl[2] = 0.0
        self.data.ctrl[4] = 0.0
        self.data.ctrl[6] = 0.785
        
        for _ in range(5):
            curr_z_axis = self.data.xmat[self.ee_id].reshape(3, 3)[:, 2]
            target_z_axis = np.array([0.0, 0.0, -1.0])
            err_rot = np.cross(curr_z_axis, target_z_axis)
            jacr = np.zeros((3, self.nv))
            mujoco.mj_jacBody(self.model, self.data, None, jacr, self.ee_id)
            j6_axis = jacr[:, 5]
            delta_q6 = np.dot(j6_axis, err_rot) / (np.dot(j6_axis, j6_axis) + 1e-6)
            self.data.ctrl[5] = self.data.qpos[5] + delta_q6 * 2.0
            mujoco.mj_step(self.model, self.data)

        ee_pos = self.data.xpos[self.ee_id]
        box_pos = self.data.xpos[self.target_body_id]
        target_grasp_pos = np.array([box_pos[0], box_pos[1], box_pos[2] + 0.105])
        dist = np.linalg.norm(ee_pos - target_grasp_pos)
        
        # ==========================================
        # 💡 [해결책 3] 진척도(Progress) 보상 설계
        # ==========================================
        # 이전 스텝보다 가까워지면 플러스 점수, 멀어지면 마이너스 점수!
        self.prev_dist = dist
        
        reward = 0.0
        is_gripping = action[3] < 0.2            
        # 2. 리프팅 극대화 보상
        if is_gripping and box_pos[2] > 0.026:
            reward += (box_pos[2] - 0.025) * 2000.0

        # 3. 잭팟 보상
        done = False
        is_success = False
        if box_pos[2] > 0.05: 
            reward += 100.0
            done = True
            is_success = True
        elif self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, is_success

# ==========================================
# 2. 마르코프 전문가 정책
# ==========================================
def get_expert_action(env):
    ee_pos = env.data.xpos[env.ee_id]
    box_pos = env.data.xpos[env.target_body_id]
    grasp_z = box_pos[2] + 0.105 
    
    dist_xy = np.linalg.norm(ee_pos[:2] - box_pos[:2])
    gripper_pos = env.data.qpos[7] 
    
    gripper_action = 1.0 
    
    if dist_xy > 0.02:
        ik_target = np.array([box_pos[0], box_pos[1], max(ee_pos[2], grasp_z + 0.1)])
    elif ee_pos[2] > grasp_z + 0.01:
        ik_target = np.array([box_pos[0], box_pos[1], grasp_z])
    elif gripper_pos > 0.028:
        ik_target = np.array([box_pos[0], box_pos[1], grasp_z])
        gripper_action = -1.0 
    else:
        ik_target = np.array([box_pos[0], box_pos[1], grasp_z + 0.2])
        gripper_action = -1.0 

    err_pos = ik_target - ee_pos
    if np.linalg.norm(err_pos) > 0.05: 
        err_pos = (err_pos / np.linalg.norm(err_pos)) * 0.05

    jacp = np.zeros((3, env.nv))
    mujoco.mj_jacBody(env.model, env.data, jacp, None, env.ee_id)
    J_pos = jacp.copy()
    J_pos[:, 2] = J_pos[:, 4] = J_pos[:, 5] = J_pos[:, 6] = 0.0 

    lambda_sq = 0.01
    J_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_sq * np.eye(3))
    delta_q = J_pinv @ err_pos * 0.2 
    
    action_raw = delta_q / 0.02
    expert_action = np.array([action_raw[0], action_raw[1], action_raw[3], gripper_action])
    return np.clip(expert_action, -1.0, 1.0)

# ==========================================
# 3. PPO 에이전트 및 확정형 하이브리드 학습
# ==========================================
class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) - 1.0)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def get_action_value(self, state, action=None):
        action_mean = self.actor(state)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        dist = torch.distributions.Normal(action_mean, action_std)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(state), action_mean

def train():
    print("🚀 [Hybrid PPO+DAgger] 랜덤 초기화와 진척도(Progress) 보상으로 완전한 인지 능력을 학습합니다!")
    env = PandaPlanarEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = PPOAgent(14, 4).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    num_iterations = 300 
    steps_per_iter = 1200 
    gamma = 0.99
    
    for iteration in range(1, num_iterations + 1):
        states, actions, logprobs, rewards, dones, values, experts = [], [], [], [], [], [], []
        state = env.reset()
        ep_reward = 0
        ep_rewards_list = []
        success_count = 0
        ep_count = 0
        
        # 확실한 지식을 위해 50회까지 전문가가 직접 운전(Teacher Forcing)
        is_pretrain_phase = iteration <= 50 
        
        # PPO로 넘어간 후에도 서서히 줄어드는 모방학습 가중치
        beta_bc = 5.0 * (0.95 ** max(0, iteration - 50))
        
        for _ in range(steps_per_iter):
            expert_act = get_expert_action(env)
            experts.append(expert_act)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_value(state_tensor)
            
            if is_pretrain_phase:
                step_action = expert_act
                actions.append(expert_act)
            else:
                step_action = action.cpu().numpy()[0]
                actions.append(step_action)
                
            next_state, reward, done, is_success = env.step(step_action)
            
            states.append(state)
            logprobs.append(logprob.item()); rewards.append(reward)
            dones.append(done); values.append(value.item())
            
            state = next_state; ep_reward += reward
            
            if done:
                ep_rewards_list.append(ep_reward)
                if is_success: success_count += 1
                ep_count += 1
                ep_reward = 0
                state = env.reset()
                
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.FloatTensor(np.array(actions)).to(device)
        old_logprobs_t = torch.FloatTensor(np.array(logprobs)).to(device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(device)
        values_t = torch.FloatTensor(np.array(values)).to(device)
        expert_targets_t = torch.FloatTensor(np.array(experts)).to(device)
        
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done: discounted_sum = 0
            discounted_sum = reward + (gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(10): 
            _, newlogprobs, entropy, newvalues, action_mean = agent.get_action_value(states_t, actions_t)
            ratio = torch.exp(newlogprobs - old_logprobs_t)
            
            pg_loss = torch.max(-advantages * ratio, -advantages * torch.clamp(ratio, 0.8, 1.2)).mean()
            v_loss = 0.5 * ((newvalues.squeeze() - returns_t) ** 2).mean()
            mse_per_sample = F.mse_loss(action_mean, expert_targets_t, reduction='none').mean(dim=1)
            
            if is_pretrain_phase:
                # 완벽한 전문가의 기억을 뇌에 강제 주입
                loss = mse_per_sample.mean() * 10.0 + v_loss
            else:
                # PPO 스스로 탐험하며 성장
                bc_loss = mse_per_sample.mean() * beta_bc
                loss = pg_loss + v_loss - 0.01 * entropy.mean() + bc_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if iteration % 5 == 0 or iteration == 1:
            phase_name = "🎓 시범학습(Pure BC)" if is_pretrain_phase else "🤖 자율주행(PPO+BC)"
            avg_rew = np.mean(ep_rewards_list) if ep_rewards_list else 0
            print(f"Iter {iteration:3d}/{num_iterations} [{phase_name}] | Avg Rew: {avg_rew:6.1f} | 성공: {success_count}/{ep_count}")

    torch.save(agent.state_dict(), "final_hybrid_ppo.pth")
    print("✅ 학습 완료!")

# ==========================================
# 4. 평가 렌더링 및 비디오 녹화 함수
# ==========================================
def evaluate(use_expert=False, record_video=False, num_episodes=5):
    mode_name = "수학 전문가(Expert)" if use_expert else "PPO 인공지능"
    print(f"\n🎬 평가 모드 시작... (조종자: {mode_name})")
    
    env = PandaPlanarEnv()
    device = torch.device("cpu")
    
    if not use_expert:
        agent = PPOAgent(14, 4).to(device)
        try:
            agent.load_state_dict(torch.load("final_hybrid_ppo.pth", map_location=device))
        except:
            print("⚠️ 저장된 모델이 없습니다.")
        agent.eval()

    # 💡 [비디오 녹화 모드]
    if record_video:
        print(f"📹 총 {num_episodes}개의 에피소드 연속 비디오 녹화를 시작합니다...")
        
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.azimuth = 140
        camera.elevation = -20
        camera.distance = 2.0
        camera.lookat[:] = [0.4, 0, 0.2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_filename = f"result_{'expert' if use_expert else 'ppo'}.mp4"
        video_writer = cv2.VideoWriter(video_filename, fourcc, 50.0, (640, 480))
        
        # 💡 지정된 횟수만큼 에피소드를 반복하며 프레임을 이어서 녹화합니다.
        for ep in range(1, num_episodes + 1):
            state = env.reset()
            print(f"▶️ 에피소드 {ep}/{num_episodes} 녹화 중...")
            
            while env.current_step < env.max_steps:
                if use_expert:
                    action = get_expert_action(env)
                else:
                    with torch.no_grad():
                        action = agent.actor(torch.FloatTensor(state).unsqueeze(0).to(device)).cpu().numpy()[0]
                    
                state, reward, done, is_success = env.step(action)
                
                renderer.update_scene(env.data, camera=camera)
                pixels = renderer.render()
                pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                video_writer.write(pixels_bgr)
                
                if done:
                    if is_success: 
                        print(f"  🎉 성공!")
                        # 성공 후 상자를 들고 있는 모습을 1초(50프레임) 정도 더 녹화하여 여운을 줍니다.
                        for _ in range(50):
                            renderer.update_scene(env.data, camera=camera)
                            video_writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
                    else: 
                        print(f"  ❌ 실패...")
                    break # 현재 에피소드를 종료하고 다음 에피소드로 넘어감
                
        video_writer.release()
        renderer.close()
        print(f"✅ 비디오가 '{video_filename}'로 성공적으로 저장되었습니다!")

    # 💡 [일반 화면 팝업 보기 모드]
    else:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = [0.4, 0, 0.2]
            
            # 뷰어 모드에서도 지정된 횟수만큼 반복하도록 수정
            for ep in range(1, num_episodes + 1):
                state = env.reset()
                print(f"▶️ 에피소드 {ep}/{num_episodes} 실행 중...")
                
                while viewer.is_running():
                    step_start = time.time()
                    
                    if use_expert:
                        action = get_expert_action(env)
                    else:
                        with torch.no_grad():
                            action = agent.actor(torch.FloatTensor(state).unsqueeze(0).to(device)).cpu().numpy()[0]
                        
                    state, reward, done, is_success = env.step(action)
                    viewer.sync()
                    
                    if done:
                        if is_success: 
                            print(f"  🎉 성공!")
                        else: 
                            print(f"  ❌ 실패...")
                        time.sleep(1.0) # 성공/실패 후 잠시 멈춤
                        break # 다음 에피소드로 넘어감
                        
                    time_until_next_step = env.model.opt.timestep * 5 - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                
                if not viewer.is_running():
                    break # 사용자가 창을 닫으면 완전히 종료

# ==========================================
# 실행 제어부
# ==========================================
if __name__ == "__main__":
    TRAIN_MODE = False           # True면 학습 시작
    EXPERT_MODE = False          # True면 전문가 테스트, False면 학습된 PPO 테스트
    RECORD_VIDEO = True          # True면 화면 팝업 대신 MP4 파일 저장
    
    # 💡 여기서 원하는 녹화(또는 시청) 에피소드 횟수를 자유롭게 설정하세요!
    RECORD_EPISODES = 1          
    
    if TRAIN_MODE:
        train()
        evaluate(use_expert=False, record_video=False, num_episodes=RECORD_EPISODES)
    else:
        evaluate(use_expert=EXPERT_MODE, record_video=RECORD_VIDEO, num_episodes=RECORD_EPISODES)