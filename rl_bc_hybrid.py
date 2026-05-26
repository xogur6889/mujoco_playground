import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==========================================
# 1. 환경 (Env) - Delta Control 
# ==========================================
class MujocoHybridPPOEnv:
    def __init__(self):
        xml = """
        <mujoco model="hybrid_bc_ppo_arm">
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
              <geom name="target_geom" type="sphere" size="0.06" rgba="1 0.3 0.3 1"/>
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
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.max_steps = 300
        self.current_step = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self.model.geom_rgba[self.target_geom_id] = [1.0, 0.3, 0.3, 1.0] 
        
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
        
        # Delta Control (상대적 제어)
        action = np.clip(action, -1.0, 1.0)
        max_step_size = 0.1
        target_ctrl = self.data.ctrl[:3] + (action * max_step_size)
        
        self.data.ctrl[0] = np.clip(target_ctrl[0], -1.5, 1.5)
        self.data.ctrl[1] = np.clip(target_ctrl[1], -2.0, 2.0)
        self.data.ctrl[2] = np.clip(target_ctrl[2], 0.0, 2.5)
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        distance = np.linalg.norm(obs[3:6] - obs[6:9])
        
        reward = -(distance * 2.0) 
        done = False
        
        if self._check_contact():
            reward += 50.0 
            done = True
            
        if self.current_step >= self.max_steps:
            done = True
            
        return obs, reward, done

    def _check_contact(self):
        id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "end_effector")
        id2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        for i in range(self.data.ncon):
            if (self.data.contact[i].geom1 == id1 and self.data.contact[i].geom2 == id2) or \
               (self.data.contact[i].geom1 == id2 and self.data.contact[i].geom2 == id1):
                return True
        return False

# ==========================================
# 2. 수학 전문가 (Expert IK) - 행동 복제용 정답지 생성기
# ==========================================
def get_ik_action(env):
    error = env.target_pos - env.data.site_xpos[env.ee_site_id]
    
    # 에러 캡핑: 자코비안 붕괴 방지
    error_norm = np.linalg.norm(error)
    max_step_dist = 0.05 
    if error_norm > max_step_dist:
        error = (error / error_norm) * max_step_dist
        
    jacp = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, None, env.ee_site_id)
    
    damping = 0.1
    inv_term = np.linalg.inv(jacp @ jacp.T + (damping ** 2) * np.eye(3))
    delta_q = jacp.T @ inv_term @ error * 0.5 
    
    # 현재 모터 명령을 기준으로 델타 제어 규격[-1, 1]에 맞춘 완벽한 정답 생성
    desired_q = env.data.qpos[:3] + delta_q
    action = (desired_q - env.data.ctrl[:3]) / 0.1
    
    return np.clip(action, -1.0, 1.0)

# ==========================================
# 3. SB3 PPO 아키텍처 완벽 이식 (신경망)
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SB3ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(9, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(9, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 3), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 3))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value_and_mean(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        # BC Loss 계산을 위해 action_mean을 추가로 반환합니다.
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean

# ==========================================
# 4. 논문 수식(1)이 결합된 하이브리드 PPO 학습 루프
# ==========================================
TRAIN_MODE = False

if __name__ == "__main__":
    env = MujocoHybridPPOEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SB3ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, eps=1e-5)
    
    # 하이퍼파라미터
    num_steps = 2048       
    batch_size = 64        
    n_epochs = 10          
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    beta_bc = 2.0  # [논문] 행동 복제(BC)의 영향력 가중치

    if TRAIN_MODE:
        print("🚀 [학습 모드] 전문가 IK 정답지를 결합한 'Behavioral-Cloning PPO'를 시작합니다!")
        start_time = time.time()
        
        # 💡 정답지가 있으므로 250번(50만 스텝)이 아닌 100번(20만 스텝)만에 PPO를 압도합니다!
        num_updates = 100 
        
        obs_buf = torch.zeros((num_steps, 9)).to(device)
        actions_buf = torch.zeros((num_steps, 3)).to(device)
        expert_actions_buf = torch.zeros((num_steps, 3)).to(device) # 전문가 정답지 버퍼 추가
        logprobs_buf = torch.zeros((num_steps)).to(device)
        rewards_buf = torch.zeros((num_steps)).to(device)
        dones_buf = torch.zeros((num_steps)).to(device)
        values_buf = torch.zeros((num_steps)).to(device)
        
        next_obs = torch.Tensor(env.reset()).to(device)
        next_done = torch.zeros(1).to(device)

        for update in range(1, num_updates + 1):
            ep_rewards = []
            current_ep_reward = 0
            
            # 1. Rollout 수집 
            for step in range(num_steps):
                obs_buf[step] = next_obs
                dones_buf[step] = next_done
                
                with torch.no_grad():
                    action, logprob, _, value, _ = model.get_action_and_value_and_mean(next_obs.unsqueeze(0))
                    values_buf[step] = value.flatten()
                    
                    # [핵심 수집] 만약 전문가(IK)였다면 이 순간에 어떤 행동을 내렸을까?
                    expert_act = get_ik_action(env)
                    expert_actions_buf[step] = torch.FloatTensor(expert_act).to(device)
                
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                
                action_np = action.cpu().numpy()[0]
                next_obs_np, reward, done = env.step(action_np)
                
                rewards_buf[step] = torch.tensor(reward).to(device)
                next_obs = torch.Tensor(next_obs_np).to(device)
                next_done = torch.Tensor([done]).to(device)
                
                current_ep_reward += reward
                if done:
                    ep_rewards.append(current_ep_reward)
                    current_ep_reward = 0
                    next_obs = torch.Tensor(env.reset()).to(device)

            # 2. GAE 계산
            with torch.no_grad():
                next_value = model.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
                advantages = torch.zeros_like(rewards_buf).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_buf[t + 1]
                        nextvalues = values_buf[t + 1]
                    delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values_buf

            # 3. PPO + BC 최적화 (논문 수식 반영)
            b_inds = np.arange(num_steps)
            for epoch in range(n_epochs):
                np.random.shuffle(b_inds) 
                for start in range(0, num_steps, batch_size):
                    end = start + batch_size
                    mb_inds = b_inds[start:end]
                    
                    # 신경망의 현재 출력(newmean)을 가져옴
                    _, newlogprob, entropy, newvalue, newmean = model.get_action_and_value_and_mean(obs_buf[mb_inds], actions_buf[mb_inds])
                    
                    # PPO Surrogate Loss
                    logratio = newlogprob - logprobs_buf[mb_inds]
                    ratio = logratio.exp()
                    mb_advantages = advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    
                    # === [논문 핵심: BC Loss] ===
                    # 인공지능이 전문가(IK)의 행동 궤적을 닮도록 강제 (MSE Loss 활용)
                    mb_expert_actions = expert_actions_buf[mb_inds]
                    bc_loss = F.mse_loss(newmean, mb_expert_actions)
                    
                    # [최종 Loss] PPO 기본 로직 + 행동 복제(BC)
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * 0.5 + (beta_bc * bc_loss)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            avg_reward = np.mean(ep_rewards) if len(ep_rewards) > 0 else 0
            if update % 5 == 0:
                print(f"Update {update:3d}/{num_updates} | Avg Reward: {avg_reward:7.1f} | BC Loss: {bc_loss.item():.4f}")

        torch.save(model.state_dict(), "hybrid_bc_ppo.pth")
        print(f"✅ 하이브리드(PPO+BC) 학습 완료! 소요 시간: {time.time() - start_time:.2f}초")

    else:
        print("🎮 [테스트 모드] 전문가의 부드러움과 RL의 유연성이 결합된 모델을 테스트합니다.")
        model.load_state_dict(torch.load("hybrid_bc_ppo.pth"))
        model.eval()
        
        state = env.reset()
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                state_ts = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    mean_action = model.actor(state_ts)
                
                state, reward, done = env.step(mean_action.cpu().numpy()[0])
                
                if done:
                    if reward > 0:
                        print("🎉 [하이브리드 AI] IK의 궤적과 RL의 문제해결력으로 타격 성공!")
                        env.model.geom_rgba[env.target_geom_id] = [0.2, 1.0, 0.2, 1.0]
                        viewer.sync()
                        time.sleep(1.0)
                    else:
                        print("⏰ 시간 초과! (재시도)")
                    
                    state = env.reset()
                    step_start = time.time()
                
                viewer.sync()
                time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)