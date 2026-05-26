# import sqlite3
# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['font.family'] = 'sans-serif'
# sns.set_theme(style="whitegrid")

# class NASPaperLogger:
#     def __init__(self, db_paths=["nas_seed1.db"], output_dir="paper_figures"):
#         self.db_paths = [db for db in db_paths if os.path.exists(db)]
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         if not self.db_paths:
#             print("⚠️ No databases found. Please run the main loop to generate databases.")
#             self.dfs = []
#         else:
#             self.dfs = [self._load_data(db) for db in self.db_paths]

#     def _load_data(self, db_path) -> pd.DataFrame:
#         conn = sqlite3.connect(db_path)
#         df = pd.read_sql_query("SELECT * FROM evaluated_genomes", conn)
#         conn.close()
#         df['fps'] = 1000.0 / (df['latency'] + 1e-5)
#         return df

#     def get_pareto_front(self, df: pd.DataFrame) -> pd.DataFrame:
#         pareto_front = []
#         df_sorted = df.sort_values('latency', ascending=True)
#         max_map = -1.0
#         for _, row in df_sorted.iterrows():
#             if row['mAP'] > max_map:
#                 pareto_front.append(row)
#                 max_map = row['mAP']
#         return pd.DataFrame(pareto_front)

#     def calculate_exact_2d_hypervolume(self, pf_df: pd.DataFrame, ref_latency=30.0, ref_map=0.0) -> float:
#         """논문용 절대 기준점 (Target Deployment Constraint) 적용"""
#         if pf_df.empty: return 0.0
#         pf_sorted = pf_df.sort_values('latency', ascending=False)
#         hv = 0.0
#         prev_lat = ref_latency
        
#         for _, row in pf_sorted.iterrows():
#             lat, mAP = row['latency'], row['mAP']
#             if lat >= ref_latency: continue
#             hv += (prev_lat - lat) * (mAP - ref_map)
#             prev_lat = lat
#         return hv

#     def plot_1_pareto_evolution(self):
#         """Population Cloud Scatter 시각화"""
#         if not self.dfs: return
#         df = self.dfs[0] 
#         plt.figure(figsize=(10, 6))
        
#         target_gens = [0, df['generation'].max() // 2, df['generation'].max()]
#         colors = ['#FF9999', '#99CCFF', '#99FF99'] 
#         line_colors = ['red', 'blue', 'green']     
        
#         for gen, c_cloud, c_line in zip(target_gens, colors, line_colors):
#             gen_df = df[df['generation'] == gen]
#             pf = self.get_pareto_front(gen_df)
            
#             # Numpy array 변환 (Pandas 인덱싱 오류 방지)
#             plt.scatter(gen_df['latency'].to_numpy(), gen_df['mAP'].to_numpy(), 
#                         color=c_cloud, alpha=0.5, s=30, label=f'Gen {gen} Population')
            
#             plt.plot(pf['latency'].to_numpy(), pf['mAP'].to_numpy(), 
#                      color=c_line, marker='o', linewidth=2, label=f'Gen {gen} Pareto')

#         plt.title('Population Distribution & Pareto Front Evolution')
#         plt.xlabel('Latency (ms)')
#         plt.ylabel('mAP')
#         plt.legend()
#         plt.savefig(f"{self.output_dir}/fig1_pareto_population_cloud.pdf", bbox_inches='tight')
#         plt.close()
#         print("Saved: fig1_pareto_population_cloud.pdf")

#     def plot_2_hypervolume_curve_multiseed(self, ref_latency=30.0):
#         """Multi-Seed 신뢰구간 (Mean ± Std) 포함 Hypervolume 곡선"""
#         if not self.dfs: return
        
#         max_gen = max([df['generation'].max() for df in self.dfs])
#         generations = np.arange(0, max_gen + 1)
#         all_hv_curves = []

#         for df in self.dfs:
#             hv_curve = []
#             for gen in generations:
#                 gen_df = df[df['generation'] <= gen]
#                 pf = self.get_pareto_front(gen_df)
#                 hv = self.calculate_exact_2d_hypervolume(pf, ref_latency=ref_latency)
#                 hv_curve.append(hv)
#             all_hv_curves.append(hv_curve)

#         hv_matrix = np.array(all_hv_curves)
#         hv_mean = np.mean(hv_matrix, axis=0)
#         hv_std = np.std(hv_matrix, axis=0)

#         plt.figure(figsize=(8, 5))
#         plt.plot(generations, hv_mean, color='purple', linewidth=2, label='NSGA-II (Mean)')
#         plt.fill_between(generations, hv_mean - hv_std, hv_mean + hv_std, color='purple', alpha=0.2, label='± 1 Std Dev')
        
#         plt.title('Hypervolume Convergence (Multi-Seed)')
#         plt.xlabel('Generation')
#         plt.ylabel(f'Hypervolume (Ref Lat: {ref_latency}ms)')
#         plt.legend()
#         plt.savefig(f"{self.output_dir}/fig2_hypervolume_multiseed.pdf", bbox_inches='tight')
#         plt.close()
#         print("Saved: fig2_hypervolume_multiseed.pdf")

#     def export_ablation_table(self):
#         """논문 필수 기여도 분리표"""
#         data = {
#             "Method": [
#                 "Baseline (YOLO11n)", 
#                 "Random Search (Predictor Enabled)", 
#                 "Ours w/o Multi-Fidelity", 
#                 "Ours (Fixed YOLO Backbone + NAS Neck/Head)",
#                 "Ours w/o Active Learning (No RF Uncertainty)",
#                 "Ours (Full HW-NAS)"
#             ],
#             "Search Cost (GPU Hrs)": ["0", "TBD", "TBD", "TBD", "TBD", "TBD"],
#             "Best mAP (@10ms)": ["TBD", "TBD", "TBD", "TBD", "TBD", "TBD"]
#         }
#         pd.DataFrame(data).to_csv(f"{self.output_dir}/table_ablation_study.csv", index=False)
#         print("Saved: table_ablation_study.csv")


# if __name__ == "__main__":
#     logger = NASPaperLogger(db_paths=["nas_global_cache.db"], output_dir="paper_figures")
#     logger.plot_1_pareto_evolution()
#     logger.plot_2_hypervolume_curve_multiseed()
#     logger.export_ablation_table()
#     print("✅ 논문용 그래프 추출이 완료되었습니다! 'paper_figures' 폴더를 확인하세요.")

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 폰트 및 스타일 세팅
plt.rcParams['font.family'] = 'Malgun Gothic' # 영문 논문일 경우 'sans-serif'로 변경
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

os.makedirs("paper_figures", exist_ok=True)

def load_chronological_data(db_name):
    """DB에 저장된 순서(rowid)대로 데이터를 불러와 시간 흐름을 복원합니다."""
    if not os.path.exists(db_name): return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_name)
        # rowid를 가져와서 평가된 순서(Chronological Order)를 추적
        df = pd.read_sql_query("SELECT rowid, * FROM evaluated_genomes WHERE mAP IS NOT NULL", conn)
        conn.close()
        return df.sort_values('rowid').reset_index(drop=True)
    except:
        return pd.DataFrame()

def get_pareto_front(df):
    if df.empty: return pd.DataFrame()
    sorted_df = df.sort_values(by=['latency', 'mAP'], ascending=[True, False])
    pareto_points = []
    max_map = -1.0
    for _, row in sorted_df.iterrows():
        if row['mAP'] > max_map:
            pareto_points.append(row)
            max_map = row['mAP']
    return pd.DataFrame(pareto_points)

def calculate_hypervolume(pf_df, ref_latency=30.0, ref_map=0.0):
    """지연시간(낮을수록 좋음)과 mAP(높을수록 좋음)의 2D 하이퍼볼륨 면적 계산"""
    if pf_df.empty: return 0.0
    pf_sorted = pf_df.sort_values('latency', ascending=False)
    hv = 0.0
    prev_lat = ref_latency
    for _, row in pf_sorted.iterrows():
        lat, mAP = row['latency'], row['mAP']
        if lat >= ref_latency: continue
        hv += (prev_lat - lat) * (mAP - ref_map)
        prev_lat = lat
    return hv

# 데이터 로드
df_nas = load_chronological_data("nas_global_cache.db")
df_rand = load_chronological_data("random_search_cache.db")

# =====================================================================
# Plot 1: Pareto Evolution (NAS의 탐색 과정이 어떻게 진화했는가?)
# =====================================================================
if not df_nas.empty and len(df_nas) >= 3:
    plt.figure(figsize=(10, 6))
    
    # 데이터를 3단계(초기, 중기, 후기)로 나누어 진화 과정을 시각화
    n_total = len(df_nas)
    stages = [
        ("Early Stage (33%)", df_nas.iloc[:n_total//3], '#99CCFF', 'blue'),
        ("Mid Stage (66%)", df_nas.iloc[:(n_total*2)//3], '#99FF99', 'green'),
        ("Final Stage (100%)", df_nas, '#FF9999', 'red')
    ]
    
    for label, sub_df, color_cloud, color_line in stages:
        pf = get_pareto_front(sub_df)
        # 구름(전체 탐색) 찍기
        plt.scatter(sub_df['latency'], sub_df['mAP'], color=color_cloud, alpha=0.3, s=20)
        # 파레토 프론트 선 긋기
        plt.plot(pf['latency'], pf['mAP'], color=color_line, marker='o', linewidth=2, label=label)

    plt.title('Pareto Front Evolution Over Search Progress (NAS)', fontsize=15, fontweight='bold')
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('mAP_50', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("paper_figures/fig1_pareto_evolution.png")
    print("✅ 생성 완료: paper_figures/fig1_pareto_evolution.png")

# =====================================================================
# Plot 2: Hypervolume Convergence (탐색 비용 대비 성능 향상 속도 비교)
# =====================================================================
if not df_nas.empty:
    plt.figure(figsize=(9, 5))
    
    def get_hv_trajectory(df, ref_lat=25.0): # ref_lat은 측정된 latency 중 가장 큰 값 언저리로 설정
        trajectory = []
        for i in range(1, len(df) + 1):
            sub_df = df.iloc[:i]
            pf = get_pareto_front(sub_df)
            hv = calculate_hypervolume(pf, ref_latency=ref_lat)
            trajectory.append(hv)
        return trajectory

    # 레퍼런스 Latency 설정 (NAS 데이터 기준 최대 Latency + 2ms 여유)
    ref_latency = df_nas['latency'].max() + 2.0 
    
    hv_nas = get_hv_trajectory(df_nas, ref_latency)
    plt.plot(range(1, len(hv_nas) + 1), hv_nas, color='dodgerblue', linewidth=2.5, label='Hardware-Aware NAS')
    
    if not df_rand.empty:
        hv_rand = get_hv_trajectory(df_rand, ref_latency)
        plt.plot(range(1, len(hv_rand) + 1), hv_rand, color='darkorange', linewidth=2.5, linestyle='--', label='Random Search')

    plt.title('Hypervolume Convergence: NAS vs Random Search', fontsize=15, fontweight='bold')
    plt.xlabel('Number of Evaluated Architectures (Search Cost)', fontsize=12)
    plt.ylabel(f'Hypervolume Area (Ref Latency: {ref_latency:.1f}ms)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("paper_figures/fig2_hypervolume_convergence.png")
    print("✅ 생성 완료: paper_figures/fig2_hypervolume_convergence.png")
