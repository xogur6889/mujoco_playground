import mujoco
import mujoco.viewer
import numpy as np
import time
import random

# ==========================================
# 1. 완벽하게 동일한 환경 (Env) - Delta Control
# ==========================================
class MujocoHeuristicEnv:
    def __init__(self):
        xml = """
        <mujoco model="heuristic_arm_expert">
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
        
        # PPO와 완벽히 동일한 Delta Control 매핑
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
# 2. 역운동학 수학 전문가 (자코비안 붕괴 방지 적용)
# ==========================================
def get_ik_action(env):
    # 1. 실제 목표까지의 전체 오차 벡터
    error = env.target_pos - env.data.site_xpos[env.ee_site_id]
    
    # === [핵심 수학적 트릭: 에러 캡핑 (Error Capping)] ===
    # 자코비안 미분 공식이 붕괴하지 않도록, 한 스텝당 처리할 최대 에러 길이를 5cm(0.05m)로 강제합니다.
    # 이렇게 하면 로봇이 멀리 있는 공을 볼 때 무리하게 관절을 꺾지 않고, 
    # 조금씩 부드럽게 직선으로 허공을 가르며 이동하게 됩니다.
    error_norm = np.linalg.norm(error)
    max_step_dist = 0.05 
    if error_norm > max_step_dist:
        error = (error / error_norm) * max_step_dist
        
    # 2. 자코비안 행렬 추출
    jacp = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, None, env.ee_site_id)
    
    # 3. DLS(감쇠 최소제곱법) 기반 델타 관절 각도 계산
    damping = 0.1
    inv_term = np.linalg.inv(jacp @ jacp.T + (damping ** 2) * np.eye(3))
    delta_q = jacp.T @ inv_term @ error
    
    # 4. 환경(Env)이 매 스텝 모터명령에 (action * 0.1)을 더해주므로 역산하여 action 추출
    action = delta_q / 0.1 
    
    # PPO 인공지능이 뱉는 값과 동일하게 -1.0 ~ 1.0 으로 클리핑
    return np.clip(action, -1.0, 1.0)

# ==========================================
# 3. 메인 실행 블록 (테스트 뷰어)
# ==========================================
if __name__ == "__main__":
    print("🧮 [전문가 모드] 오차 정규화가 적용된 완벽하고 부드러운 수학적 궤적을 확인합니다.")
    
    env = MujocoHeuristicEnv()
    state = env.reset()
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # 수학적으로 완벽히 통제된 행동(Action) 추출
            action = get_ik_action(env)
            
            state, reward, done = env.step(action)
            
            if done:
                if reward > 0:
                    print(f"🎉 [IK 전문가] 로봇팔이 완벽한 직선 궤적으로 타격했습니다! (소요 스텝: {env.current_step})")
                    env.model.geom_rgba[env.target_geom_id] = [0.2, 1.0, 0.2, 1.0]
                    viewer.sync()
                    time.sleep(1.0)
                else:
                    print("⏰ 시간 초과! (물리적으로 도달할 수 없는 특이점 영역)")
                
                state = env.reset()
                step_start = time.time()
            
            viewer.sync()
            
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)