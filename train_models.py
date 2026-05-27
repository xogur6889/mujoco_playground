import mujoco
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # <--- [수정됨] 이 줄이 추가되었습니다!

# ==========================================
# 1. 공통 환경 및 전문가 정의
# ==========================================
class MujocoTrainEnv:
    def __init__(self):
        xml = """
        <mujoco model="train_arm">
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

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
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
# 2. 신경망 정의
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
        self.actor_logstd = nn.Parameter(torch.zeros(1, 3) - 1.0)

    def get_value(self, x): return self.critic(x)
    def get_action_and_value_and_mean(self, x, action=None):
        action_mean = self.actor(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean

# ==========================================
# 3. 통합 학습 함수
# ==========================================
def train_model(mode_name, save_path, use_bc=False):
    print(f"\n🚀 [{mode_name}] 학습 시작...")
    env = MujocoTrainEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SB3ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, eps=1e-5)
    
    num_steps, batch_size, n_epochs = 2048, 64, 10
    gamma, gae_lambda, clip_coef, ent_coef = 0.99, 0.95, 0.2, 0.01
    
    num_updates = 200 
    initial_beta = 5.0 if use_bc else 0.0
    
    obs_buf = torch.zeros((num_steps, 9)).to(device)
    actions_buf = torch.zeros((num_steps, 3)).to(device)
    expert_actions_buf = torch.zeros((num_steps, 3)).to(device)
    logprobs_buf = torch.zeros((num_steps)).to(device)
    rewards_buf = torch.zeros((num_steps)).to(device)
    dones_buf = torch.zeros((num_steps)).to(device)
    values_buf = torch.zeros((num_steps)).to(device)
    
    next_obs = torch.Tensor(env.reset()).to(device)
    next_done = torch.zeros(1).to(device)
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # Beta를 빠르게 감쇠시켜 초반에만 강력하게 개입
        current_beta = initial_beta * (0.85 ** (update - 1)) if use_bc else 0.0
        
        ep_rewards = []
        current_ep_reward = 0
        
        for step in range(num_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value, _ = model.get_action_and_value_and_mean(next_obs.unsqueeze(0))
                values_buf[step] = value.flatten()
                if use_bc:
                    expert_actions_buf[step] = torch.FloatTensor(get_ik_action(env)).to(device)
            
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            
            next_obs_np, reward, done = env.step(action.cpu().numpy()[0])
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
                nextnonterminal = 1.0 - next_done if t == num_steps - 1 else 1.0 - dones_buf[t + 1]
                nextvalues = next_value if t == num_steps - 1 else values_buf[t + 1]
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
                ratio = (newlogprob - logprobs_buf[mb_inds]).exp()
                mb_advantages = (advantages[mb_inds] - advantages[mb_inds].mean()) / (advantages[mb_inds].std() + 1e-8)
                
                pg_loss = torch.max(-mb_advantages * ratio, -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                
                loss = pg_loss - ent_coef * entropy.mean() + v_loss * 0.5
                
                if use_bc:
                    bc_loss = F.mse_loss(newmean, expert_actions_buf[mb_inds])
                    loss += current_beta * bc_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        avg_reward = np.mean(ep_rewards) if len(ep_rewards) > 0 else 0
        if update % 10 == 0:
            print(f"Update {update:3d}/{num_updates} | Beta: {current_beta:.4f} | Avg Reward: {avg_reward:6.1f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ [{mode_name}] 모델 저장 완료! ({time.time() - start_time:.1f}초 소요)")

if __name__ == "__main__":
    train_model("1. Pure PPO (비교군)", "pure_ppo_final.pth", use_bc=False)
    train_model("3. Hybrid PPO+BC (실험군)", "hybrid_bc_ppo_final.pth", use_bc=True)