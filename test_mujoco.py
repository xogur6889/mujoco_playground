import mujoco
import mujoco.viewer
import time

xml_string = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1" pos="0 0 1"/>
    <geom name="floor" type="plane" size="1 1 .1" rgba=".9 .9 .9 1"/>
  </worldbody>
</mujoco>
"""

print("MuJoCo 모델 로딩 중...")
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

print("MuJoCo 뷰어 실행 (5초 후 자동 종료됩니다)...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while viewer.is_running() and time.time() - start_time < 5.0:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print("테스트 완료!")
