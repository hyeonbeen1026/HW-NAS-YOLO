import os
import uuid
import yaml
import time
import json
import logging
import sqlite3
import numpy as np
import random
import torch
import gc

from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend  # [추가됨] 순수 TRT 측정을 위한 코어 백엔드
from architecture_decoder import GenomeDecoder, WeightSurgeon
from latency_predictor import LatencyPredictor
from multi_fidelity_evaluator import MultiFidelityEvaluator
from evolution_engine import NSGA2Engine, GenomeOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MAIN] - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """실험 재현성을 위한 Seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed} for reproducibility.")

class SQLiteGenomeCache:
    def __init__(self, db_path="nas_global_cache.db"):
        # 멀티스레드 충돌 방지를 위한 SQLite 설정
        self.conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=30000;") 
        
        # 논문용 통계 지표 저장 테이블
        self.conn.execute('''CREATE TABLE IF NOT EXISTS evaluated_genomes
                             (genome_str TEXT PRIMARY KEY,
                              mAP REAL, 
                              mAP_3e REAL, 
                              slope REAL, 
                              predicted_latency REAL, 
                              actual_latency REAL, 
                              latency REAL, 
                              generation INTEGER)''')
        
        self.conn.execute('''CREATE TABLE IF NOT EXISTS kv_store
                             (key TEXT PRIMARY KEY, value TEXT)''')

    def exists(self, genome: list) -> bool:
        cursor = self.conn.execute("SELECT 1 FROM evaluated_genomes WHERE genome_str=?", (str(genome),))
        return cursor.fetchone() is not None

    def add(self, genome: list, result: dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO evaluated_genomes 
               (genome_str, mAP, mAP_3e, slope, predicted_latency, actual_latency, latency, generation) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(genome), result['mAP'], result.get('mAP_3e', 0.0), result['slope'], 
             result.get('predicted_latency', 0.0), result.get('actual_latency', -1.0), 
             result['latency'], result['generation'])
        )

    def get_all_evaluated(self) -> list:
        cursor = self.conn.execute("SELECT genome_str, mAP, mAP_3e, slope, predicted_latency, actual_latency, latency, generation FROM evaluated_genomes")
        results = []
        for row in cursor.fetchall():
            results.append({
                'genome': json.loads(row[0]),
                'mAP': row[1], 'mAP_3e': row[2], 'slope': row[3],
                'predicted_latency': row[4], 'actual_latency': row[5],
                'latency': row[6], 'generation': row[7]
            })
        return results

    def get_all_hashes(self) -> set:
        cursor = self.conn.execute("SELECT genome_str FROM evaluated_genomes")
        return set([tuple(json.loads(row[0])) for row in cursor.fetchall()])

    def save_state(self, gen: int, population: list, status: str):
        self.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_gen', ?)", (str(gen),))
        self.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('population', ?)", (json.dumps(population),))
        self.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('status', ?)", (status,))
        
        # 세대 완료 시 WAL 파일 Truncate (디스크 공간 확보)
        if status == "completed":
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            logger.info("SQLite WAL checkpoint truncated to save disk space.")

    def load_state(self):
        cur_gen = self.conn.execute("SELECT value FROM kv_store WHERE key='current_gen'").fetchone()
        cur_pop = self.conn.execute("SELECT value FROM kv_store WHERE key='population'").fetchone()
        status = self.conn.execute("SELECT value FROM kv_store WHERE key='status'").fetchone()
        
        if cur_gen and cur_pop and status:
            return int(cur_gen[0]), json.loads(cur_pop[0]), status[0]
        return 0, None, None


def measure_real_trt_latency(genome: list, nc: int = 7) -> float:
    """Blackwell/RTX 환경을 위한 실제 TensorRT FP16 지연 시간 측정 (AutoBackend 적용)"""
    temp_yaml = f"temp_trt_arch_{uuid.uuid4().hex}.yaml"
    
    try:
        # 1. 아키텍처 디코딩 및 가중치 수술
        decoder = GenomeDecoder(num_classes=nc)
        cfg, layer_map, _ = decoder.decode(genome)
        surgeon = WeightSurgeon(pretrained_path="yolo11n.pt")
        custom_model = surgeon.transplant(cfg, layer_map)

        with open(temp_yaml, 'w') as f:
            yaml.dump(cfg, f)

        # 2. YOLO 엔진 로드
        yolo_engine = YOLO(temp_yaml, task='detect')
        yolo_engine.model.load_state_dict(custom_model.state_dict())

        # 3. TensorRT FP16 포맷으로 Export (변환)
        engine_path = yolo_engine.export(format='engine', half=True, workspace=8, verbose=False)

        # 4. TRT 모델 로드 및 Warm-up (예열)
        # [수정됨] YOLO wrapper 대신 AutoBackend를 사용하여 PyTorch Fallback 에러 원천 차단
        trt_model = AutoBackend(engine_path, device=torch.device('cuda'), fp16=True)
        trt_model.eval()
        dummy_input = torch.zeros((1, 3, 640, 640), device='cuda', dtype=torch.float16)

        # 예열 20회로 최적화
        for _ in range(20):
            trt_model(dummy_input)

        # 5. 실제 지연 시간 실측 (50회 평균) - 순수 NPU 추론 시간만 측정
        latencies = []
        for _ in range(50):
            torch.cuda.synchronize() # 정확한 실측을 위한 동기화
            start_time = time.perf_counter()
            
            trt_model(dummy_input)
            
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start_time) * 1000.0)

        avg_latency = float(np.mean(latencies))
        
        # 엔진 객체 삭제
        del trt_model
        
        return avg_latency

    except Exception as e:
        logger.error(f"🚨 TRT Measurement Failed: {e}")
        return 100.0 # 파레토 왜곡 방지를 위한 합리적 페널티
        
    finally:
        # 생성된 임시 파일들 청소
        if 'surgeon' in locals():
            surgeon.cleanup()
        for ext in ['.yaml', '.engine', '.onnx']:
            file_to_remove = temp_yaml.replace('.yaml', ext)
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)
                
        # [핵심] 완벽한 메모리 누수 방지 (OOM 예방)
        gc.collect()
        torch.cuda.empty_cache()


def main():
    logger.info("🚀 Starting Autonomous HW-NAS Production Pipeline 🚀")
    set_seed(42) 
    
    # [설정] 최대 세대 및 개체군 크기 
    MAX_GENERATIONS = 20
    POP_SIZE = 20
    ACTIVE_LEARNING_RATIO = 20 
    WARMUP_GENERATIONS = 2 # Surrogate 모델 안정화를 위한 Warm-up 기간
    
    db = SQLiteGenomeCache()
    predictor = LatencyPredictor(n_estimators=100) 
    evaluator = MultiFidelityEvaluator(num_workers=1)
    engine = NSGA2Engine(pop_size=POP_SIZE, max_gen=MAX_GENERATIONS)
    optimizer = GenomeOptimizer()
    
    start_gen, saved_population, status = db.load_state()
    
    # [핵심 로직 검증] Predictor(Replay Buffer) 기반 Active Learning 복구 로직 (논문 기조 완벽 일치)
    if start_gen > 0:
        logger.info("🔄 Restoring Predictor Replay Buffer from Database...")
        past_evals = db.get_all_evaluated()
        recovery_genomes = []
        recovery_latencies = []
        
        for record in past_evals:
            # 실측된 지연 시간(actual_latency)이 있는 경우만 수집
            if record['actual_latency'] > 0:
                recovery_genomes.append(record['genome'])
                recovery_latencies.append(record['actual_latency'])
                
        if recovery_genomes:
            predictor.calibrate(recovery_genomes, recovery_latencies, start_gen)
            logger.info(f"✅ Predictor successfully restored with {len(recovery_genomes)} past TRT samples!")

    if status == "completed":
        start_gen += 1
        saved_population = None
        logger.info(f"Previous generation fully completed. Resuming from Generation {start_gen}...")
    elif status == "in_progress":
        logger.warning(f"🚨 Crash detected mid-generation! Resuming Generation {start_gen} from checkpoint...")

    for gen in range(start_gen, MAX_GENERATIONS):
        logger.info(f"{'='*20} Generation {gen}/{MAX_GENERATIONS} {'='*20}")
        
        if saved_population is not None:
            candidates = saved_population
            saved_population = None 
        elif gen == 0:
            logger.info("Initializing Random Population...")
            candidates = []
            while len(candidates) < POP_SIZE:
                g = optimizer.generate_random_genome()
                if not db.exists(g):
                    candidates.append(g)
        else:
            logger.info("Generating next generation via NSGA-II...")
            all_evaluated = db.get_all_evaluated()
            global_hashes = db.get_all_hashes()
            candidates = engine.generate_next_generation(all_evaluated, gen, global_hashes)
            
        db.save_state(gen, candidates, status="in_progress")
            
        logger.info("Predicting Latencies & Checking Uncertainty...")
        pred_latencies, uncertainties = predictor.predict_batch(candidates)
        
        if gen >= WARMUP_GENERATIONS:
            dynamic_threshold = np.percentile(uncertainties, 100 - ACTIVE_LEARNING_RATIO)
        else:
            dynamic_threshold = 0.0 
            
        min_trt_samples = max(3, int(0.1 * POP_SIZE))
        top_unc_indices = set(np.argsort(uncertainties)[-min_trt_samples:])
        
        genomes_to_eval = []
        trt_measured_genomes = []
        trt_measured_latencies = []
        
        for i, genome in enumerate(candidates):
            pred_lat = pred_latencies[i]
            unc = uncertainties[i]
            
            # Warm-up 기간 내이거나 Uncertainty 임계치 초과 시 실제 지연 시간 측정 (논문 Active Learning 전략)
            if gen < WARMUP_GENERATIONS or unc >= dynamic_threshold or i in top_unc_indices:
                reason = "Warm-up" if gen < WARMUP_GENERATIONS else ("Threshold" if unc >= dynamic_threshold else "Floor Guarantee")
                logger.info(f"TRT Fallback Triggered ({reason}) -> Unc: {unc:.4f}")
                
                # 진짜 TensorRT 실측 함수 호출 (nc=7 동기화)
                actual_latency = measure_real_trt_latency(genome, nc=7)
                
                trt_measured_genomes.append(genome)
                trt_measured_latencies.append(actual_latency)
                
                genomes_to_eval.append({
                    'genome': genome, 
                    'predicted_latency': pred_lat,
                    'actual_latency': actual_latency,
                    'latency': actual_latency
                })
            else:
                genomes_to_eval.append({
                    'genome': genome, 
                    'predicted_latency': pred_lat,
                    'actual_latency': -1.0, 
                    'latency': pred_lat
                })
                
        if trt_measured_genomes:
            logger.info(f"Calibrating Predictor with {len(trt_measured_genomes)} new TRT samples...")
            predictor.calibrate(trt_measured_genomes, trt_measured_latencies, gen)

        untested_genomes = [item['genome'] for item in genomes_to_eval if not db.exists(item['genome'])]
        
        if untested_genomes:
            logger.info(f"Dispatching {len(untested_genomes)} untested genomes to Ray Cluster...")
            eval_results = evaluator.evaluate_population(untested_genomes)
            
            for res in eval_results:
                if res['status'] == 'success':
                    g = res['genome']
                    match_item = next(item for item in genomes_to_eval if item['genome'] == g)
                    
                    final_data = {
                        'mAP': res['mAP'], 
                        'mAP_3e': res.get('mAP_3e', 0.0), 
                        'slope': res['slope'],
                        'predicted_latency': match_item['predicted_latency'],
                        'actual_latency': match_item['actual_latency'],
                        'latency': match_item['latency'], 
                        'generation': gen
                    }
                    db.add(g, final_data)

        db.save_state(gen, candidates, status="completed")
        logger.info(f"Generation {gen} Completed. Checkpoint saved.")

    logger.info("🎉 HW-NAS Pipeline Finished Successfully! Ready for Deployment! 🎉")

if __name__ == "__main__":
    main()
