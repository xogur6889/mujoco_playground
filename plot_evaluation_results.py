import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('evaluation_results.csv')

# 모델 이름 깔끔하게 변경 (논문용)
model_mapping = {
    '1_Pure_PPO': 'Pure PPO',
    '2_Heuristic_IK': 'Heuristic IK',
    '3_Hybrid_PPO_BC': 'Hybrid (Ours)'
}
df['Model'] = df['Model'].map(model_mapping)

# 성공 여부를 숫자로 변환 (Yes=100, No=0)
df['Success_Num'] = df['Success'].apply(lambda x: 100 if x == 'Yes' else 0)

# 논문 스타일 폰트 및 색상 설정
sns.set_theme(style="whitegrid", font_scale=1.2)
colors = ["#e74c3c", "#3498db", "#2ecc71"] # 빨강, 파랑, 초록

plt.figure(figsize=(18, 5))

# 1. 성공률 막대 그래프 (Success Rate)
plt.subplot(1, 3, 1)
sns.barplot(x='Model', y='Success_Num', data=df, palette=colors, capsize=.1, errorbar=None)
plt.title('Target Reaching Success Rate (%)', fontweight='bold')
plt.ylabel('Success Rate (%)')
plt.ylim(0, 100)
for i, v in enumerate(df.groupby('Model')['Success_Num'].mean()):
    plt.text(i, v + 2, f"{v:.1f}%", color='black', ha='center', fontweight='bold')

# 2. 평균 보상 박스 플롯 (Average Reward)
plt.subplot(1, 3, 2)
sns.boxplot(x='Model', y='Reward', data=df, palette=colors, width=0.5)
plt.title('Episode Reward Distribution', fontweight='bold')
plt.ylabel('Total Reward')

# 3. 소요 스텝 막대 그래프 (Steps to Reach)
plt.subplot(1, 3, 3)
sns.barplot(x='Model', y='Steps', data=df, palette=colors, capsize=.1, errorbar=None)
plt.title('Average Steps to Completion (Lower is Better)', fontweight='bold')
plt.ylabel('Number of Steps')
plt.ylim(0, 350)
for i, v in enumerate(df.groupby('Model')['Steps'].mean()):
    plt.text(i, v + 5, f"{v:.1f}", color='black', ha='center', fontweight='bold')

# 레이아웃 조정 및 저장
plt.tight_layout()
plt.savefig('paper_results_figure.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 논문용 고화질 그래프(paper_results_figure.png)가 성공적으로 생성되었습니다!")