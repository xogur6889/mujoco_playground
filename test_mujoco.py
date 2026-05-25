import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class MujocoReachEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # [수정 1] 관통을 줄이기 위해 모터 힘(kp)을 60으로 낮추고, joint2의 범위를 늘림
        xml = """
        <mujoco model="rl_arm_humanoid_v2">
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
            <position joint="joint1" name="ctrl_j1" kp="60" ctrlrange="-1.5 1.5"/>
            <position joint="joint2" name="ctrl_j2" kp="60" ctrlrange="-2.0 2.0"/>
            <position joint="joint3" name="ctrl_j3" kp="60" ctrlrange="0.0 2.5"/>
          </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.target_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")]
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        # 스텝을 300으로 늘려 멀리 있는 물체에도 포기하지 않고 도달할 시간을 줌
        self.max_steps = 300
        self.current_step = 0

    def _check_contact(self, geom_name1, geom_name2):
        id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name1)
        id2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name2)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == id1 and contact.geom2 == id2) or \
               (contact.geom1 == id2 and contact.geom2 == id1):
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # 리셋 시 타겟 공을 다시 빨간색으로 되돌림
        self.model.geom_rgba[self.target_geom_id] = [1.0, 0.3, 0.3, 1.0]
        
        r = random.uniform(0.3, 0.65)
        theta = random.uniform(-1.0, 1.0)
        self.target_pos = np.array([r * np.cos(theta), r * np.sin(theta), random.uniform(0.15, 0.5)])
        self.data.mocap_pos[self.target_mocap_id] = self.target_pos
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        ee_pos = self.data.site_xpos[self.ee_site_id]
        qpos = self.data.qpos[:3]
        return np.concatenate([qpos, ee_pos, self.target_pos]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        safe_action = np.clip(action, -1.0, 1.0)
        
        # 넓어진 joint2 범위에 맞게 매핑 수정
        self.data.ctrl[0] = safe_action[0] * 1.5  
        self.data.ctrl[1] = safe_action[1] * 2.0  # -2.0 ~ 2.0
        self.data.ctrl[2] = (safe_action[2] + 1.0) * 1.25 
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        ee_pos = obs[3:6] 
        target_pos = obs[6:9]
        
        distance = np.linalg.norm(ee_pos - target_pos)
        
        # [수정 2] 거리에 대한 보상을 두 배로 키우고, 행동 페널티를 확 줄여서 로봇이 적극적으로 움직이게 만듦
        action_penalty = 0.01 * np.sum(np.square(action))
        reward = -(distance * 2.0) - action_penalty
        
        terminated = False
        truncated = False
        
        if self._check_contact("end_effector", "target_geom"):
            reward += 50.0  
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, {}


# ==========================================
# 실행 블록
# ==========================================
# ⚠️ 100만 번의 학습을 통해 인공지능이 완벽한 공간 지각 능력을 갖추게 합니다!
TRAIN_MODE = False

if __name__ == "__main__":
    env = MujocoReachEnv()
    model_path = "ppo_mujoco_arm_master" 

    if TRAIN_MODE:
        print("🚀 [학습 모드] 관통 버그를 수정하고 100만 번의 마스터 학습을 시작합니다! (약 15~20분 소요)")
        rl_model = PPO("MlpPolicy", env, verbose=1)
        start_time = time.time()
        
        # 1,000,000 스텝 학습!
        rl_model.learn(total_timesteps=1000000)
        
        print(f"✅ 학습 완료! 총 소요 시간: {time.time() - start_time:.2f}초")
        rl_model.save(model_path)
        print("💾 마스터 모델 저장 완료! TRAIN_MODE를 False로 변경하세요.")

    else:
        print("🎮 [테스트 모드] 시각적 피드백(초록색 공)이 추가된 뷰어를 실행합니다.")
        try:
            rl_model = PPO.load(model_path)
        except FileNotFoundError:
            print("❌ 모델 파일이 없습니다. TRAIN_MODE = True 로 먼저 학습해주세요.")
            exit()
            
        obs, _ = env.reset()
        
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                action, _states = rl_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    if terminated:
                        print("🎉 [정밀 타격 성공!] 타겟이 초록색으로 변합니다!")
                        # [시각 효과] 성공 시 타겟 공을 초록색으로 변경
                        env.model.geom_rgba[env.target_geom_id] = [0.2, 1.0, 0.2, 1.0]
                        viewer.sync()
                        time.sleep(1.0) # 1초간 초록불 감상
                    else:
                        print("⏰ 시간 초과! (Truncated)")
                        
                    obs, _ = env.reset()
                    step_start = time.time()
                
                viewer.sync()
                
                time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)