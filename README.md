# 🤖 MuJoCo Playground

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xogur6889/mujoco_playground/blob/main/mujoco_colab_test.ipynb)

이 저장소는 **MuJoCo** 물리 엔진을 로컬 PC의 무거운 환경 구축 없이 **Google Colab** 환경에서 가볍게 테스트하고 실행해 보기 위해 만들어진 플레이그라운드입니다. [DeepMind의 MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) 프로젝트에서 영감을 받아 작성되었습니다.

## 🚀 Getting Started

가장 쉽고 빠른 실행 방법은 Google Colab을 이용하는 것입니다.
상단의 **"Open In Colab"** 뱃지를 클릭하면 `mujoco_colab_test.ipynb` 파일이 여러분의 코랩 환경에서 즉시 열립니다.

### ✨ Features
- **Headless Rendering**: 화면 창(GUI)을 띄울 수 없는 코랩 환경의 특성을 고려하여, `mediapy`를 활용해 시뮬레이션 결과를 MP4 비디오 형태로 렌더링합니다.
- **Basic Physics Check**: 바닥(Plane), 조명, 그리고 중력에 의해 떨어지는 빨간색 박스를 렌더링하여 MuJoCo 엔진이 정상적으로 작동하는지 검증합니다.

## 🛠️ Local Installation (Optional)
만약 로컬 환경에서 실행하고자 한다면 아래 명령어로 의존성을 설치하세요:
```bash
pip install -r requirements.txt
```