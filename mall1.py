# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

# 데이터 불러오기
data = pd.read_csv("Mall_Customers.csv")

# 데이터 탐색
print("데이터의 처음 5행 확인:")
print(data.head())
print("\n데이터 정보 확인:")
print(data.info())
print("\n데이터 통계 요약:")
print(data.describe())

# 데이터 전처리
# 결측치 확인 및 처리
print("\n결측치 확인:")
print(data.isnull().sum())

# 분석에 필요한 열 선택 (예: 나이, 연간 소득, 지출 점수)
selected_columns = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 데이터 스케일링 (표준화)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_columns)

# 스케일링된 데이터 확인
print("\n스케일링된 데이터 확인 (앞 5행):")
print(scaled_data[:5])

# ==============================
# 클러스터링 기법 적용
# ==============================

# K-Means 클러스터링
# 엘보우 방법을 사용하여 최적의 클러스터 수 결정
inertia = []
range_k = range(1, 11)

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# 엘보우 방법 결과 시각화
plt.figure(figsize=(8, 5))
plt.plot(range_k, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# 최적 클러스터 수 설정 (예: 3으로 설정)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_clusters = kmeans.fit_predict(scaled_data)

# 계층적 군집화 (Agglomerative Clustering)
hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_clusters = hierarchical.fit_predict(scaled_data)

print("\n계층적 군집화 결과 (클러스터 라벨):")
print(hierarchical_clusters)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(scaled_data)

print("\nDBSCAN 클러스터링 결과 (클러스터 라벨):")
print(dbscan_clusters)

data['DBSCAN Cluster'] = dbscan_clusters
print(data.head())

# ==============================
# 최적의 클러스터 수 결정
# ==============================

# 실루엣 점수 계산 (K-Means 기준)
silhouette_avg = silhouette_score(scaled_data, kmeans_clusters)
print(f"K-Means의 평균 실루엣 점수: {silhouette_avg:.3f}")

# ==============================
# 결과 시각화 (2D 및 3D)
# ==============================

# 2D 시각화 (K-Means 결과)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=scaled_data[:, 1], y=scaled_data[:, 2], hue=kmeans_clusters, palette='viridis', s=50
)
plt.title('K-Means Clustering Results (2D)')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend(title='Cluster')
plt.show()

# 3D 시각화 (K-Means 결과)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    scaled_data[:, 2],
    c=kmeans_clusters,
    cmap='viridis',
    s=50,
)
ax.set_title('K-Means Clustering Results (3D)')
ax.set_xlabel('Age (scaled)')
ax.set_ylabel('Annual Income (scaled)')
ax.set_zlabel('Spending Score (scaled)')
plt.show()

# ==============================
# 결과 비교
# ==============================

# 클러스터링 결과를 데이터프레임에 추가
data['KMeans_Cluster'] = kmeans_clusters
data['Hierarchical_Cluster'] = hierarchical_clusters
data['DBSCAN_Cluster'] = dbscan_clusters

print("\n클러스터링 결과가 추가된 데이터프레임:")
print(data.head())

# ==============================
# 마무리
# ==============================

print("\n분석 완료! 클러스터링 결과를 확인하세요.")

