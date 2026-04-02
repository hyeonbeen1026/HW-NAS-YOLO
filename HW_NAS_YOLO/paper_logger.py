import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")

class NASPaperLogger:
    def __init__(self, db_paths=["nas_seed1.db"], output_dir="paper_figures"):
        self.db_paths = [db for db in db_paths if os.path.exists(db)]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not self.db_paths:
            print("⚠️ No databases found. Please run the main loop to generate databases.")
            self.dfs = []
        else:
            self.dfs = [self._load_data(db) for db in self.db_paths]

    def _load_data(self, db_path) -> pd.DataFrame:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM evaluated_genomes", conn)
        conn.close()
        df['fps'] = 1000.0 / (df['latency'] + 1e-5)
        return df

    def get_pareto_front(self, df: pd.DataFrame) -> pd.DataFrame:
        pareto_front = []
        df_sorted = df.sort_values('latency', ascending=True)
        max_map = -1.0
        for _, row in df_sorted.iterrows():
            if row['mAP'] > max_map:
                pareto_front.append(row)
                max_map = row['mAP']
        return pd.DataFrame(pareto_front)

    def calculate_exact_2d_hypervolume(self, pf_df: pd.DataFrame, ref_latency=30.0, ref_map=0.0) -> float:
        """논문용 절대 기준점 (Target Deployment Constraint) 적용"""
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

    def plot_1_pareto_evolution(self):
        """Population Cloud Scatter 시각화"""
        if not self.dfs: return
        df = self.dfs[0] 
        plt.figure(figsize=(10, 6))
        
        target_gens = [0, df['generation'].max() // 2, df['generation'].max()]
        colors = ['#FF9999', '#99CCFF', '#99FF99'] 
        line_colors = ['red', 'blue', 'green']     
        
        for gen, c_cloud, c_line in zip(target_gens, colors, line_colors):
            gen_df = df[df['generation'] == gen]
            pf = self.get_pareto_front(gen_df)
            
            # Numpy array 변환 (Pandas 인덱싱 오류 방지)
            plt.scatter(gen_df['latency'].to_numpy(), gen_df['mAP'].to_numpy(), 
                        color=c_cloud, alpha=0.5, s=30, label=f'Gen {gen} Population')
            
            plt.plot(pf['latency'].to_numpy(), pf['mAP'].to_numpy(), 
                     color=c_line, marker='o', linewidth=2, label=f'Gen {gen} Pareto')

        plt.title('Population Distribution & Pareto Front Evolution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('mAP')
        plt.legend()
        plt.savefig(f"{self.output_dir}/fig1_pareto_population_cloud.pdf", bbox_inches='tight')
        plt.close()
        print("Saved: fig1_pareto_population_cloud.pdf")

    def plot_2_hypervolume_curve_multiseed(self, ref_latency=30.0):
        """Multi-Seed 신뢰구간 (Mean ± Std) 포함 Hypervolume 곡선"""
        if not self.dfs: return
        
        max_gen = max([df['generation'].max() for df in self.dfs])
        generations = np.arange(0, max_gen + 1)
        all_hv_curves = []

        for df in self.dfs:
            hv_curve = []
            for gen in generations:
                gen_df = df[df['generation'] <= gen]
                pf = self.get_pareto_front(gen_df)
                hv = self.calculate_exact_2d_hypervolume(pf, ref_latency=ref_latency)
                hv_curve.append(hv)
            all_hv_curves.append(hv_curve)

        hv_matrix = np.array(all_hv_curves)
        hv_mean = np.mean(hv_matrix, axis=0)
        hv_std = np.std(hv_matrix, axis=0)

        plt.figure(figsize=(8, 5))
        plt.plot(generations, hv_mean, color='purple', linewidth=2, label='NSGA-II (Mean)')
        plt.fill_between(generations, hv_mean - hv_std, hv_mean + hv_std, color='purple', alpha=0.2, label='± 1 Std Dev')
        
        plt.title('Hypervolume Convergence (Multi-Seed)')
        plt.xlabel('Generation')
        plt.ylabel(f'Hypervolume (Ref Lat: {ref_latency}ms)')
        plt.legend()
        plt.savefig(f"{self.output_dir}/fig2_hypervolume_multiseed.pdf", bbox_inches='tight')
        plt.close()
        print("Saved: fig2_hypervolume_multiseed.pdf")

    def export_ablation_table(self):
        """논문 필수 기여도 분리표"""
        data = {
            "Method": [
                "Baseline (YOLO11n)", 
                "Random Search (Predictor Enabled)", 
                "Ours w/o Multi-Fidelity", 
                "Ours (Fixed YOLO Backbone + NAS Neck/Head)",
                "Ours w/o Active Learning (No RF Uncertainty)",
                "Ours (Full HW-NAS)"
            ],
            "Search Cost (GPU Hrs)": ["0", "TBD", "TBD", "TBD", "TBD", "TBD"],
            "Best mAP (@10ms)": ["TBD", "TBD", "TBD", "TBD", "TBD", "TBD"]
        }
        pd.DataFrame(data).to_csv(f"{self.output_dir}/table_ablation_study.csv", index=False)
        print("Saved: table_ablation_study.csv")


if __name__ == "__main__":
    logger = NASPaperLogger(db_paths=["nas_global_cache.db"], output_dir="paper_figures")
    logger.plot_1_pareto_evolution()
    logger.plot_2_hypervolume_curve_multiseed()
    logger.export_ablation_table()
    print("✅ 논문용 그래프 추출이 완료되었습니다! 'paper_figures' 폴더를 확인하세요.")