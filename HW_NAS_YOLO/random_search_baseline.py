import logging
import sqlite3
import numpy as np
from architecture_decoder import GenomeDecoder
from multi_fidelity_evaluator import MultiFidelityEvaluator
from latency_predictor import LatencyPredictor
from evolution_engine import GenomeOptimizer
from main_loop import set_seed, measure_real_trt_latency

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [RANDOM] - %(message)s')
logger = logging.getLogger(__name__)

def main_random_search_fair_baseline():
    logger.info("🚀 Starting FAIR Random Search Baseline (Predictor Enabled) 🚀")
    set_seed(777) 
    
    # 가상 탐색 예산(Cost) 제어
    TARGET_BUDGET_COST = 5000.0 
    current_cost = 0.0
    BATCH_SIZE = 30
    
    conn = sqlite3.connect("random_search_cache.db", isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute('''CREATE TABLE IF NOT EXISTS evaluated_genomes
                    (genome_str TEXT PRIMARY KEY, mAP REAL, slope REAL, latency REAL, generation INTEGER)''')
    
    optimizer = GenomeOptimizer()
    evaluator = MultiFidelityEvaluator(num_workers=1)
    predictor = LatencyPredictor(n_estimators=100)
    
    gen = 0
    while current_cost < TARGET_BUDGET_COST:
        logger.info(f"--- Random Batch Gen {gen} (Cost: {current_cost:.1f}/{TARGET_BUDGET_COST}) ---")
        candidates = [optimizer.generate_random_genome() for _ in range(BATCH_SIZE)]

        pred_latencies, uncertainties = predictor.predict_batch(candidates)
        dynamic_threshold = np.percentile(uncertainties, 80) if gen > 1 else 0.0

        genomes_to_eval = []
        trt_measured_genomes, trt_measured_latencies = [], []
        
        for i, genome in enumerate(candidates):
            if gen < 2 or uncertainties[i] >= dynamic_threshold:
                actual_latency = measure_real_trt_latency(genome, nc=7)
                trt_measured_genomes.append(genome)
                trt_measured_latencies.append(actual_latency)
                genomes_to_eval.append({'genome': genome, 'latency': actual_latency})
                current_cost += 5.0 
            else:
                genomes_to_eval.append({'genome': genome, 'latency': pred_latencies[i]})
                current_cost += 0.1 
                
        if trt_measured_genomes:
            predictor.calibrate(trt_measured_genomes, trt_measured_latencies, gen)

        eval_results = evaluator.evaluate_population([item['genome'] for item in genomes_to_eval])
        current_cost += len(eval_results) * 10.0 
        
        for res in eval_results:
            if res['status'] == 'success':
                g = res['genome']
                match_item = next(item for item in genomes_to_eval if item['genome'] == g)
                conn.execute(
                    "INSERT OR REPLACE INTO evaluated_genomes (genome_str, mAP, slope, latency, generation) VALUES (?, ?, ?, ?, ?)",
                    (str(g), res['mAP'], res['slope'], match_item['latency'], gen)
                )
        
        gen += 1
        
    logger.info("🎉 Fair Random Search Baseline Completed!")

if __name__ == "__main__":
    main_random_search_fair_baseline()
