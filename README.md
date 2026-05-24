# 🤖 MuJoCo Playground

이 저장소는 **MuJoCo** 물리 엔진을 로컬 PC의 무거운 환경 구축 없이 **Google Colab** 환경에서 가볍게 테스트하고 실행해 보기 위해 만들어진 플레이그라운드입니다. [DeepMind의 MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) 프로젝트에서 영감을 받아 작성되었습니다.

## 🚀 Quick Start (Google Colab)

아래의 "Open In Colab" 뱃지를 클릭하면 각 주피터 노트북 파일이 여러분의 코랩 환경에서 즉시 열립니다. 원하는 테스트 환경을 선택해 보세요.

### 1. 기본 환경 테스트 (기초)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xogur6889/mujoco_playground/blob/main/mujoco_colab_test.ipynb)
- **내용**: 중력에 의해 바닥으로 떨어지는 빨간색 큐브를 통해 MuJoCo 엔진 및 렌더러가 정상 작동하는지 검증합니다.

### 2. ALOHA 양팔 로봇 테스트 (심화)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xogur6889/mujoco_playground/blob/main/aloha_colab_test.ipynb)
- **내용**: 구글 딥마인드의 공식 로봇 저장소(`mujoco_menagerie`)에서 ALOHA 쌍팔 로봇 모델을 불러와 부드럽게 조종하고 렌더링합니다.

## ✨ Features
- **Headless Rendering**: 화면 창(GUI)을 띄울 수 없는 코랩 환경의 특성을 고려하여, `mediapy`를 활용해 시뮬레이션 결과를 MP4 비디오 형태로 렌더링합니다.
- **DeepMind Menagerie Integration**: 딥마인드 공식 깃허브와 연동하여 실제 연구용 로봇 모델을 가볍게 불러와 볼 수 있습니다.

## 🛠️ Local Installation (Optional)
만약 로컬 PC 환경에서 실행하고자 한다면 아래 명령어로 의존성을 설치하세요:
```bash
pip install -r requirements.txt
```