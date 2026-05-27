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
# 1. 환경 (Env) - "바늘구멍 찾기" 난이도 극대화
# ==========================================
class MujocoHybridPPOEnv:
    def __init__(self):
        xml = """
        <mujoco model="hybrid_bc_ppo_arm_hard">
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
        
        action = np.clip(action, -1.0, 1.0)
        max_step_size = 0.1
        target_ctrl = self.data.ctrl[:3] + (action * max_step_size)
        
        self.data.ctrl[0] = np.clip(target_ctrl[0], -1.5, 1.5)
        self.data.ctrl[1] = np.clip(target_ctrl[1], -2.0, 2.0)
        self.data.ctrl[2] = np.clip(target_ctrl[2], 0.0, 2.5)
        
        mujoco.mj_step(self.model, self.data)
        
        # [핵심 수정 2] 가혹한 희소 보상 + 흔들림 페널티
        action_penalty = 0.01 * np.sum(np.square(action))
        reward = -0.05 - action_penalty 
        done = False
        
        if self._check_contact():
            reward += 100.0  # 타겟이 작아진 만큼 보상을 100점으로 상향 (극적인 그래프 도출)
            done = True
            
        if self.current_step >= self.max_steps:
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
# 2. 수학 전문가 (Expert IK)
# ==========================================
def get_ik_action(env):
    error = env.target_pos - env.data.site_xpos[env.ee_site_id]
    error_norm = np.linalg.norm(error)
    max_step_dist = 0.05 
    if error_norm > max_step_dist:
        error = (error / error_norm) * max_step_dist
        
    jacp = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, None, env.ee_site_id)
    
    damping = 0.1
    inv_term = np.linalg.inv(jacp @ jacp.T + (damping ** 2) * np.eye(3))
    delta_q = jacp.T @ inv_term @ error * 0.5 
    
    desired_q = env.data.qpos[:3] + delta_q
    action = (desired_q - env.data.ctrl[:3]) / 0.1
    return np.clip(action, -1.0, 1.0)

# ==========================================
# 3. 신경망 (SB3 PPO 구조)
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
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean

# ==========================================
# 4. 하이브리드 PPO 학습 루프
# ==========================================
TRAIN_MODE = True

if __name__ == "__main__":
    env = MujocoHybridPPOEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SB3ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, eps=1e-5)
    
    num_steps = 2048       
    batch_size = 64        
    n_epochs = 10          
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    
    initial_beta = 2.0 

    if TRAIN_MODE:
        print("🚀 [학습 모드] 탐험 난이도 극대화 + 지수 감쇠(Exponential Decay) 시작!")
        start_time = time.time()
        num_updates = 250
        
        obs_buf = torch.zeros((num_steps, 9)).to(device)
        actions_buf = torch.zeros((num_steps, 3)).to(device)
        expert_actions_buf = torch.zeros((num_steps, 3)).to(device)
        logprobs_buf = torch.zeros((num_steps)).to(device)
        rewards_buf = torch.zeros((num_steps)).to(device)
        dones_buf = torch.zeros((num_steps)).to(device)
        values_buf = torch.zeros((num_steps)).to(device)
        
        next_obs = torch.Tensor(env.reset()).to(device)
        next_done = torch.zeros(1).to(device)

        for update in range(1, num_updates + 1):
            # [핵심 수정 3] 지수 감쇠 (Exponential Decay) 적용
            # 부드럽게 2.0에서 시작하여 100 업데이트 즈음에 0에 수렴하도록 설계
            current_beta = initial_beta * (0.95 ** (update - 1))
            
            ep_rewards = []
            current_ep_reward = 0
            
            for step in range(num_steps):
                obs_buf[step] = next_obs
                dones_buf[step] = next_done
                
                with torch.no_grad():
                    action, logprob, _, value, _ = model.get_action_and_value_and_mean(next_obs.unsqueeze(0))
                    values_buf[step] = value.flatten()
                    
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

            b_inds = np.arange(num_steps)
            for epoch in range(n_epochs):
                np.random.shuffle(b_inds) 
                for start in range(0, num_steps, batch_size):
                    end = start + batch_size
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue, newmean = model.get_action_and_value_and_mean(obs_buf[mb_inds], actions_buf[mb_inds])
                    
                    logratio = newlogprob - logprobs_buf[mb_inds]
                    ratio = logratio.exp()
                    mb_advantages = advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    
                    mb_expert_actions = expert_actions_buf[mb_inds]
                    bc_loss = F.mse_loss(newmean, mb_expert_actions)
                    
                    # Beta Decay 가중치 적용
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * 0.5 + (current_beta * bc_loss)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            avg_reward = np.mean(ep_rewards) if len(ep_rewards) > 0 else 0
            if update % 5 == 0:
                print(f"Update {update:3d}/{num_updates} | Beta: {current_beta:.3f} | Avg Reward: {avg_reward:6.1f} | BC Loss: {bc_loss.item():.4f}")

        torch.save(model.state_dict(), "hybrid_bc_ppo_paper_proof.pth")
        print(f"✅ 논문 증명용 하이브리드 학습 완료! 소요 시간: {time.time() - start_time:.2f}초")

    else:
        print("🎮 [테스트 모드] 바늘구멍을 통과하는 완벽한 하이브리드 AI를 확인합니다.")
        model.load_state_dict(torch.load("hybrid_bc_ppo_paper_proof.pth"))
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
                    if reward > 50: 
                        print("🎉 [논문 입증 성공] 작아진 타겟을 흔들림 없이 터치!")
                        env.model.geom_rgba[env.target_geom_id] = [0.2, 1.0, 0.2, 1.0]
                        viewer.sync()
                        time.sleep(1.0)
                    else:
                        print("⏰ 아쉬운 종료! (재시도)")
                    
                    state = env.reset()
                    step_start = time.time()
                
                viewer.sync()
                time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)