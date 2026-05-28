import os
import time
import numpy as np
import mujoco
import mujoco.viewer

class SanityCheckEnv:
    def __init__(self):
        xml = """
        <mujoco>
          <compiler angle="radian"/>
          <include file="scene.xml"/>
          <worldbody>
            <body name="target" pos="0.5 -0.1 0.025">
              <freejoint name="target_joint"/>
              <geom name="target_geom" type="box" size="0.03 0.03 0.03" rgba="0.2 0.8 0.2 1" mass="0.05" friction="2 0.1 0.001"/>
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
        
        self.q_home = np.zeros(self.nv)
        self.q_home[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.data.qpos[:self.nv] = self.q_home.copy()
        self.data.ctrl[:self.nu] = self.q_home[:self.nu].copy()
        mujoco.mj_forward(self.model, self.data)
        
        self.fixed_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.fixed_quat, self.data.xmat[self.ee_id])

        # ==========================================
        # [핵심 수정] 모터의 한계치를 직접 읽어옵니다.
        # ==========================================
        self.gripper_min = self.model.actuator_ctrlrange[7][0]
        self.gripper_max = self.model.actuator_ctrlrange[7][1]
        
        # Franka 로봇은 값이 클수록 열림, 작을수록 닫힘입니다.
        self.gripper_open = self.gripper_max
        self.gripper_close = self.gripper_min

def get_planar_ik_action(env, target_pos, gripper_state):
    curr_pos = env.data.xpos[env.ee_id]
    err_pos = target_pos - curr_pos

    curr_quat = np.zeros(4)
    mujoco.mju_mat2Quat(curr_quat, env.data.xmat[env.ee_id])
    if np.dot(env.fixed_quat, curr_quat) < 0: curr_quat = -curr_quat 
    err_rot = np.zeros(3)
    mujoco.mju_subQuat(err_rot, env.fixed_quat, curr_quat)

    err = np.concatenate([err_pos, err_rot])
    if np.linalg.norm(err) > 0.05: 
        err = (err / np.linalg.norm(err)) * 0.05

    jacp = np.zeros((3, env.nv))
    jacr = np.zeros((3, env.nv))
    mujoco.mj_jacBody(env.model, env.data, jacp, jacr, env.ee_id)
    J = np.vstack([jacp, jacr]) 

    J[:, 2] = 0.0 
    J[:, 4] = 0.0 
    J[:, 6] = 0.0 

    lambda_sq = 0.01
    J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
    delta_q = J_pinv @ err * 0.5 

    ctrl_target = env.data.ctrl[:env.nu].copy()
    ctrl_target[:7] = env.data.qpos[:7] + delta_q[:7]
    
    # 비틀림 관절 고정
    ctrl_target[2] = 0.0
    ctrl_target[4] = 0.0
    ctrl_target[6] = 0.785
    
    # 6번 수직 보정
    curr_z_axis = env.data.xmat[env.ee_id].reshape(3, 3)[:, 2]
    target_z_axis = np.array([0.0, 0.0, -1.0])
    err_rot_6 = np.cross(curr_z_axis, target_z_axis)
    j6_axis = jacr[:, 5]
    delta_q6 = np.dot(j6_axis, err_rot_6) / (np.dot(j6_axis, j6_axis) + 1e-6)
    ctrl_target[5] = env.data.qpos[5] + delta_q6 * 2.0
    
    # ==========================================
    # [핵심 수정] 절대적인 한계치를 할당합니다.
    # ==========================================
    if gripper_state == "open":
        ctrl_target[7:] = env.gripper_open
    else:
        ctrl_target[7:] = env.gripper_close
        
    return ctrl_target

if __name__ == "__main__":
    env = SanityCheckEnv()
    print("🚀 [물리 환경 검증기] 완벽한 정답 궤적을 강제로 실행합니다.")
    print(f"🔧 읽어온 그리퍼 제어값 -> 열림: {env.gripper_open} / 닫힘: {env.gripper_close}")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -15
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.4, 0, 0.2]
        
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time
            
            # [시나리오 궤적 설계]
            if elapsed < 2.0:
                ik_target = np.array([0.5, 0.0, 0.15])
                gripper = "open"
                phase = "1. Hover (접근)"
            elif elapsed < 4.0:
                ik_target = np.array([0.5, 0.0, 0.02])
                gripper = "open"
                phase = "2. Descend (하강)"
            elif elapsed < 5.0:
                ik_target = np.array([0.5, 0.0, 0.02])
                gripper = "close"
                phase = "3. Grasp (파지)"
            else:
                ik_target = np.array([0.5, 0.0, 0.25])
                gripper = "close"
                phase = "4. Lift (들어올림)"
            
            # 모터에 제어값 적용
            target_ctrl = get_planar_ik_action(env, ik_target, gripper)
            for i in range(env.nu):
                env.data.ctrl[i] = np.clip(target_ctrl[i], env.model.actuator_ctrlrange[i][0], env.model.actuator_ctrlrange[i][1])
            
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            
            current_box_z = env.data.xpos[env.target_body_id][2]
            
            if int(elapsed * 100) % 10 == 0:
                print(f"[{phase}] 박스 높이: {current_box_z:.3f}m")
            
            if elapsed > 8.0:
                print("🎯 시나리오 완료! 다시 시작합니다.")
                start_time = time.time()
                mujoco.mj_resetData(env.model, env.data)
                env.data.qpos[:env.nv] = env.q_home.copy()
                env.data.ctrl[:env.nu] = env.q_home[:env.nu].copy()
                
                # 박스 원위치
                jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
                qpos_adr = env.model.jnt_qposadr[jnt_id]
                env.data.qpos[qpos_adr:qpos_adr+3] = [0.5, 0.0, 0.02]
                env.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
                mujoco.mj_forward(env.model, env.data)
            
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)