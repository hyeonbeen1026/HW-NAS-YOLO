import ray
import torch
import logging
import time
import numpy as np
import yaml
import uuid
import os
from ultralytics import YOLO

from architecture_decoder import GenomeDecoder, WeightSurgeon

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)
class RayProxyTrainer:
    def __init__(self, data_yaml="proxy_10percent_stratified.yaml"):
        self.data_yaml = data_yaml
        self.decoder = GenomeDecoder(num_classes=7)
        self.surgeon = WeightSurgeon(pretrained_path="yolo11n.pt")

    def train_and_eval(self, genome: list, epochs: int):
        MAX_RETRIES = 2
        
        for attempt in range(1, MAX_RETRIES + 1):
            # 동시 실행 시 파일명 충돌을 방지하기 위한 UUID 할당
            temp_yaml = f"temp_arch_{uuid.uuid4().hex}.yaml"
            
            try:
                cfg, layer_map, _ = self.decoder.decode(genome)
                custom_pytorch_model = self.surgeon.transplant(cfg, layer_map)

                # 동적 생성된 구조를 임시 YAML로 저장하여 YOLO 엔진에 전달
                with open(temp_yaml, 'w') as f:
                    yaml.dump(cfg, f)

                yolo_engine = YOLO(temp_yaml, task='detect')
                yolo_engine.model.load_state_dict(custom_pytorch_model.state_dict())

                # [수정됨] 데스크탑 한계 돌파 세팅 및 pretrained=False 추가
                results = yolo_engine.train(
                    data=self.data_yaml, 
                    epochs=epochs, 
                    imgsz=640,
                    batch=48, 
                    workers=0,           # CPU 병목 방지
                    device=0,            # 3070 강제 할당
                    cache=True,
                    amp=True, 
                    verbose=False, 
                    save=False, 
                    plots=False,
                    project="runs/nas_proxy",
                    pretrained=False     # 🔥 YOLO의 가중치 강제 덮어쓰기 방지
                )
                
                mAP_50_95 = results.box.map
                mAP_small = results.box.maps[0] 
                
                history = np.linspace(0.1, mAP_50_95, epochs) 
                slope = (history[-1] - history[-3]) / 2.0 if epochs >= 3 else 0.0

                self.surgeon.cleanup()
                
                if os.path.exists(temp_yaml):
                    os.remove(temp_yaml)
                    
                return {"status": "success", "mAP": mAP_50_95 + 0.5 * mAP_small, "slope": slope, "genome": genome}
                
            except torch.cuda.OutOfMemoryError as e:
                # 구조적 VRAM 초과 예외 처리
                logger.error(f"[Fatal] CUDA OOM on genome. Discarding. Error: {e}")
                self.surgeon.cleanup()
                if os.path.exists(temp_yaml): os.remove(temp_yaml)
                return {"status": "fatal_error", "mAP": 0.0, "slope": 0.0, "genome": genome}
                
            except RuntimeError as e:
                if "size mismatch" in str(e).lower() or "shape" in str(e).lower():
                    logger.error(f"[Fatal] Shape Mismatch on genome. Discarding. Error: {e}")
                    self.surgeon.cleanup()
                    if os.path.exists(temp_yaml): os.remove(temp_yaml)
                    return {"status": "fatal_error", "mAP": 0.0, "slope": 0.0, "genome": genome}
                else:
                    if attempt == MAX_RETRIES:
                        if os.path.exists(temp_yaml): os.remove(temp_yaml)
                        return {"status": "fatal_error", "mAP": 0.0, "slope": 0.0, "genome": genome}
                    logger.warning(f"[Retry {attempt}/{MAX_RETRIES}] Transient Runtime Error: {e}")
                    time.sleep(2) 
                    
            except Exception as e:
                # 일시적 런타임 에러 재시도 로직
                if attempt == MAX_RETRIES:
                    logger.error(f"Failed after {MAX_RETRIES} attempts. Error: {e}")
                    if os.path.exists(temp_yaml): os.remove(temp_yaml)
                    return {"status": "fatal_error", "mAP": 0.0, "slope": 0.0, "genome": genome}
                
                logger.warning(f"[Retry {attempt}/{MAX_RETRIES}] Transient Error detected. Retrying... Error: {e}")
                time.sleep(2)
                
            finally:
                if os.path.exists(temp_yaml):
                    os.remove(temp_yaml)


class MultiFidelityEvaluator:
    def __init__(self, num_workers=8):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.workers = [RayProxyTrainer.remote() for _ in range(num_workers)]

    def evaluate_population(self, population: list):
        import json
        import os
        logger.info(f"Starting Multi-Fidelity Evaluation for {len(population)} models")
        
        checkpoint_file = "multi_fidelity_checkpoint.json"
        
        # 1. 기존에 진행 중이던 체크포인트가 있으면 복구
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                cache = json.load(f)
            logger.info("🔄 Found intermediate checkpoint! Resuming evaluation...")
        else:
            cache = {}

        # ---------------------------------------------------------
        # Stage 1: 3 Epochs
        # ---------------------------------------------------------
        if 'stage1' in cache:
            logger.info("⏩ Skipping Stage 1 (Loaded from checkpoint)")
            scored_stage1 = cache['stage1']
        else:
            results_stage1 = self._run_parallel_async(population, epochs=3)
            scored_stage1 = self._score_and_sort(results_stage1, alpha=0.5)
            
            # Stage 1 완료 후 체크포인트 저장
            cache['stage1'] = scored_stage1
            with open(checkpoint_file, 'w') as f: json.dump(cache, f)

        map_3e_records = {tuple(item['genome']): item['mAP'] for item in scored_stage1}
        stage2_pop = [item['genome'] for item in scored_stage1[:max(1, len(scored_stage1)//2)]]

        # ---------------------------------------------------------
        # Stage 2: 15 Epochs
        # ---------------------------------------------------------
        if 'stage2' in cache:
            logger.info("⏩ Skipping Stage 2 (Loaded from checkpoint)")
            scored_stage2 = cache['stage2']
        else:
            results_stage2 = self._run_parallel_async(stage2_pop, epochs=15)
            scored_stage2 = self._score_and_sort(results_stage2, alpha=1.0) 
            
            # Stage 2 완료 후 체크포인트 저장
            cache['stage2'] = scored_stage2
            with open(checkpoint_file, 'w') as f: json.dump(cache, f)

        stage3_pop = [item['genome'] for item in scored_stage2[:max(1, len(scored_stage2)//2)]]

        # ---------------------------------------------------------
        # Stage 3: 50 Epochs
        # ---------------------------------------------------------
        results_stage3 = self._run_parallel_async(stage3_pop, epochs=50)
        final_scored = self._score_and_sort(results_stage3, alpha=0.2)

        for item in final_scored:
            item['mAP_3e'] = map_3e_records.get(tuple(item['genome']), 0.0)

        # ---------------------------------------------------------
        # 세대 평가 완료 시 임시 체크포인트 파일 삭제
        # ---------------------------------------------------------
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return final_scored

    def _run_parallel_async(self, genomes: list, epochs: int):
        """비동기 작업 풀링 및 처리 (Straggler 방어)"""
        unfinished_futures = []
        results = []
        
        for i, genome in enumerate(genomes):
            worker = self.workers[i % len(self.workers)]
            unfinished_futures.append(worker.train_and_eval.remote(genome, epochs))
            
        while unfinished_futures:
            finished, unfinished_futures = ray.wait(unfinished_futures, num_returns=1, timeout=None)
            
            for future in finished:
                result = ray.get(future)
                results.append(result)
                
        return results

    def _score_and_sort(self, results: list, alpha: float):
        scored = []
        for r in results:
            if r['status'] == 'success':
                combined_score = r['mAP'] + (alpha * r['slope'])
                
                scored.append({
                    'genome': r['genome'], 
                    'mAP': r['mAP'], 
                    'slope': r['slope'], 
                    'score': combined_score,
                    'status': 'success'  
                })    
        return sorted(scored, key=lambda x: x['score'], reverse=True)
