import mujoco
import mujoco.viewer
import numpy as np
import time
import random

# 1. XML 정의: 타겟(공/박스)과 충돌 감지를 위한 사이트(Site) 추가
xml = """
<mujoco model="auto_reaching_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.1" rgba="0.8 0.9 0.8 1"/>
    
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.1 0.05" rgba="0.2 0.2 0.2 1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0" damping="2.0"/>
        <geom type="capsule" size="0.05 0.2" pos="0 0 0.2" rgba="0.1 0.5 0.8 1"/>
        <body name="link2" pos="0 0 0.4">
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0" damping="2.0"/>
          <geom type="capsule" size="0.04 0.2" pos="0 0 0.2" rgba="0.8 0.5 0.1 1"/>
          <body name="link3" pos="0 0 0.4">
            <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0" damping="2.0"/>
            <geom name="end_effector" type="capsule" size="0.03 0.15" pos="0 0 0.15" rgba="0.1 0.8 0.5 1"/>
            <site name="ee_site" pos="0 0 0.3" size="0.02" rgba="1 0 0 0"/> 
          </body>
        </body>
      </body>
    </body>

    <body name="target_sphere" mocap="true" pos="0 0 -1">
      <geom name="sphere_geom" type="sphere" size="0.06" rgba="1 0.3 0.3 1"/>
    </body>
    <body name="target_box" mocap="true" pos="0 0 -1">
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" rgba="0.8 0.2 0.8 1"/>
    </body>
  </worldbody>

  <actuator>
    <position joint="joint1" name="ctrl_j1" kp="100"/>
    <position joint="joint2" name="ctrl_j2" kp="100"/>
    <position joint="joint3" name="ctrl_j3" kp="100"/>
  </actuator>
</mujoco>
"""

# 2. 충돌 감지 함수
def check_contact(data, model, geom_name1, geom_name2):
    """두 geom(기하학적 물체)이 맞닿았는지 확인합니다."""
    id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name1)
    id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name2)
    
    for i in range(data.ncon):
        contact = data.contact[i]
        # 두 물체의 ID가 contact 배열에 동시에 존재하면 충돌한 것
        if (contact.geom1 == id1 and contact.geom2 == id2) or \
           (contact.geom1 == id2 and contact.geom2 == id1):
            return True
    return False

# 3. 타겟 랜덤 생성 (리셋) 함수
def reset_target(model, data):
    """공과 박스 중 하나를 선택해 로봇 주변에 랜덤으로 배치합니다."""
    # 먼저 두 타겟을 바닥 아래(보이지 않는 곳)로 숨깁니다.
    sph_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_sphere")]
    box_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")]
    data.mocap_pos[sph_id] = [0, 0, -10]
    data.mocap_pos[box_id] = [0, 0, -10]

    # 로봇이 닿을 수 있는 범위 내에서 랜덤 좌표 생성 (앞쪽 반원 형태)
    r = random.uniform(0.3, 0.7)
    theta = random.uniform(-1.0, 1.0)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = random.uniform(0.1, 0.6)
    new_target_pos = np.array([x, y, z])

    # 랜덤하게 모양 선택 (True=공, False=박스)
    is_sphere = random.choice([True, False])
    active_geom = "sphere_geom" if is_sphere else "box_geom"
    active_mocap = sph_id if is_sphere else box_id

    # 선택된 타겟을 생성된 위치로 이동
    data.mocap_pos[active_mocap] = new_target_pos
    
    return new_target_pos, active_geom

# 모델과 데이터 로드
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 초기 타겟 생성
target_pos, active_geom = reset_target(model, data)
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

print("시뮬레이션 시작! 로봇이 타겟을 향해 움직입니다.")

# 뷰어 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    # IK 연산을 위한 자코비안(Jacobian) 행렬 초기화
    jacp = np.zeros((3, model.nv))
    
    while viewer.is_running():
        step_start = time.time()
        
        # 4. 물체를 향해 움직이는 로직 (역운동학 - Jacobian Transpose)
        # 현재 로봇 끝단의 위치 구하기
        ee_pos = data.site_xpos[ee_site_id]
        
        # 타겟까지의 거리(오차) 계산
        error = target_pos - ee_pos
        
        # 로봇의 현재 자세에서 각 관절이 끝단 위치에 미치는 영향(자코비안) 계산
        mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)
        
        # 오차를 줄이기 위해 각 관절을 얼마나 움직여야 하는지 계산 (기초적인 IK 알고리즘)
        delta_q = jacp.T @ error
        
        # 제어 신호 업데이트 (부드럽게 움직이도록 게인 값 조절)
        data.ctrl[:] += delta_q * 2.0 * model.opt.timestep
        
        # 물리 엔진 1스텝 진행
        mujoco.mj_step(model, data)
        
        # 5. 성공 여부 확인 및 리셋
        if check_contact(data, model, "end_effector", active_geom):
            print("성공! 새로운 타겟을 생성합니다.")
            target_pos, active_geom = reset_target(model, data)
            
            # (선택 사항) 성공 후 로봇을 초기 자세로 되돌리기
            data.qpos[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            
            # 너무 빨리 움직이는 것을 방지하기 위해 잠깐 대기
            time.sleep(0.5) 
            step_start = time.time()
            
        viewer.sync()
        
        # 실제 시간 동기화
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)