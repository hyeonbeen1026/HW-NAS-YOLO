import numpy as np
import logging
import random
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenomeFeatureExtractor:
    """
    TensorRT Topology-Aware 피처 추출기.
    해상도(Stage) 위치 정보를 보존하여 비선형적인 지연 시간을 포착합니다.
    """
    def __init__(self, num_stages=4):
        self.num_stages = num_stages

    def transform(self, genome: list) -> np.ndarray:
        features = []
        
        early_depth = 0; heavy_depth = 0
        early_ghost = 0; heavy_ghost = 0
        early_attn = 0; heavy_attn = 0
        
        for i in range(self.num_stages):
            block_code, depth, attn_code = genome[i*3:(i+1)*3]
            
            features.extend([depth, block_code, attn_code])
            
            is_heavy = (i >= 2) 
            
            if is_heavy:
                heavy_depth += depth
                if block_code == 2: heavy_ghost += depth
                if attn_code > 0: heavy_attn += 1
            else:
                early_depth += depth
                if block_code == 2: early_ghost += depth
                if attn_code > 0: early_attn += 1

        total_depth = early_depth + heavy_depth
        
        features.extend([
            early_depth, heavy_depth,
            early_ghost, heavy_ghost,
            early_attn, heavy_attn,
            early_depth * early_attn, 
            heavy_ghost / (heavy_depth + 1e-5),
            total_depth,
            (early_attn + heavy_attn) 
        ])
        
        return np.array(features, dtype=np.float32)


class ReplayBuffer:
    def __init__(self, max_size=3000):
        self.max_size = max_size
        self.data = []

    def add(self, X_batch, y_batch, gen):
        for x, y in zip(X_batch, y_batch):
            self.data.append((x, y, gen))
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]

    def sample_balanced(self):
        sample_size = min(len(self.data), 1000)
        sampled = random.sample(self.data, sample_size)
        
        X, y = zip(*[(d[0], d[1]) for d in sampled])
        return np.array(X), np.array(y)

class LatencyPredictor:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, 
                                           random_state=random_state, 
                                           n_jobs=-1) 
        self.extractor = GenomeFeatureExtractor()
        self.buffer = ReplayBuffer(max_size=3000)
        self.is_trained = False

    def calibrate(self, genomes: list, trt_latencies: list, current_gen: int):
        X_new = [self.extractor.transform(g) for g in genomes]
        self.buffer.add(X_new, trt_latencies, current_gen)
        
        X_train, y_train = self.buffer.sample_balanced()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"Predictor Calibrated! Trained on {len(y_train)} balanced samples.")

    def predict_batch(self, genomes: list) -> tuple:
        """Batch Uncertainty 계산 가속화"""
        if not self.is_trained:
            return np.ones(len(genomes)) * 10.0, np.ones(len(genomes)) * 999.0
            
        X_batch = np.vstack([self.extractor.transform(g) for g in genomes])
        
        pred_latencies = self.model.predict(X_batch)
        
        tree_preds = np.array([tree.predict(X_batch) for tree in self.model.estimators_])
        uncertainties = np.std(tree_preds, axis=0)
        
        return pred_latencies, uncertainties
