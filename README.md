# HW-Aware-NAS-YOLO 🚀

**Hardware-Aware Asynchronous Multi-Objective NAS Pipeline for Autonomous Driving**

본 연구는 자율주행과 같이 소형 객체 비중이 높고 실시간 제약이 강한 엣지(Edge) 환경에서, 객체 탐지 모델의 **정확도(mAP)와 지연 시간(Latency) 간의 최적 파레토 균형(Pareto Optimal)**을 자동으로 탐색하는 아키텍처 탐색(NAS) 파이프라인입니다.

기존 NAS의 막대한 하드웨어 실측 비용과 장시간 학습 병목을 해결하기 위해, **다중 충실도 평가(Multi-Fidelity Evaluation)**와 **불확실성 기반 능동 학습(Uncertainty-Aware Active Learning)**을 결합한 비동기 진화 알고리즘을 제안합니다.

---

## ✨ Key Features

- **Topology-Aware Search Space & Weight Surgeon**
  - YOLO11n 시드 아키텍처의 채널 너비를 고정하여 사전 학습 가중치의 상속 안정성(Zero-shot Performance)을 보장합니다.
  - 연산 블록(`C2f`, `C3k2`, `GhostConv`)의 종류와 깊이, 그리고 Exact Identity 초기화가 적용된 Attention 모듈(`SE`, `CBAM`)의 삽입 여부를 진화 변수로 탐색합니다.
- **Asynchronous Multi-Fidelity Evaluation (Ray)**
  - Ray 프레임워크를 활용한 비동기 병렬 평가를 수행합니다.
  - `3 -> 15 -> 50 Epochs`의 Successive Halving 전략과 학습 곡선 기울기(Slope) 지표를 결합하여 탐색 효율을 극대화합니다.
- **Uncertainty-Aware Active Learning & Latency Predictor**
  - Random Forest 기반의 앙상블 예측기를 도입하여 구조의 비선형적 지연 시간을 모델링합니다.
  - 매 세대 예측 불확실성(Variance)이 높은 상위 모델만 실제 하드웨어(TensorRT FP16) 지연 시간을 실측하여 Replay Buffer 기반 온라인 캘리브레이션을 수행합니다.
- **Curriculum-based NSGA-II Optimization**
  - 탐색 초기에는 잠재력을 반영한 3-objective 최적화를 수행하고, 점진적으로 [mAP, Latency] 2-objective로 수렴하는 커리큘럼 탐색 전략을 적용합니다.
  - SQLite WAL(Write-Ahead Logging) 기반의 상태 보존 캐시를 통해 중단 없는 결함 내성(Fault-tolerant) MLOps 환경을 제공합니다.

---

## 📂 Repository Structure

본 파이프라인은 직관적인 실행을 위해 Flat Structure로 구성되어 있습니다.

```text
HW-Aware-NAS-YOLO/
├── main_loop.py                 # (Core) NSGA-II 기반 메인 NAS 파이프라인 실행 스크립트
├── random_search_baseline.py    # (Baseline) 성능 비교를 위한 FAIR Random Search 실행 스크립트
├── paper_logger.py              # (Log/Plot) DB 파싱 및 논문용 고화질 그래프/Ablation 표 추출기
├── evolution_engine.py          # NSGA-II 진화 엔진 및 하이퍼볼륨(HV) 계산 모듈
├── multi_fidelity_evaluator.py  # Ray 기반 비동기 병렬 분산 훈련/평가 매니저
├── latency_predictor.py         # RF 기반 지연 시간 예측기 및 Replay Buffer
├── architecture_decoder.py      # Genome 1D Array -> YOLO 아키텍처 변환 및 가중치 이식(Surgeon)
│
├── proxy_template.yaml          # 커스텀 데이터셋 세팅을 위한 YAML 템플릿 가이드
├── requirements.txt             # 환경 구성 패키지 목록 (CUDA 12.6 기준)
└── README.md                    # 프로젝트 설명서
```

---

## ⚙️ Installation

본 프로젝트는 **Ubuntu 22.04, Python 3.10, CUDA 12.6** 환경에서 최적화되었습니다.

1. **Repository Clone**
   ```bash
   git clone [https://github.com/your-username/HW-Aware-NAS-YOLO.git](https://github.com/your-username/HW-Aware-NAS-YOLO.git)
   cd HW-Aware-NAS-YOLO
   ```

2. **Dependencies Install**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pre-trained Seed Weights Download**
   - 가중치 이식(Weight Surgery)을 위해 Ultralytics 공식 `yolo11n.pt` 파일이 루트 폴더에 필요합니다.
   ```bash
   wget [https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
   ```

4. **Dataset Configuration**
   - `proxy_template.yaml`을 복사하여 본인의 데이터 경로에 맞게 수정한 후 `proxy_10percent_stratified.yaml` 이름으로 저장합니다.

---

## 🚀 Quick Start

### 1. Run Main NAS Pipeline (Ours)
가장 최적화된 파레토 프론트(Pareto Front)를 탐색하기 위해 본 게임을 시작합니다.
```bash
python3 main_loop.py
```
> **Note:** 평가가 완료된 모델은 실시간으로 `nas_global_cache.db`에 저장되며, 훈련이 중단되더라도 재실행 시 자동으로 이어서(Resume) 탐색을 진행합니다.

### 2. Run Random Search Baseline
동일한 탐색 예산(Budget) 내에서 제안하는 파이프라인의 우수성을 증명하기 위한 무작위 탐색 대조군입니다.
```bash
python3 random_search_baseline.py
```
> `random_search_cache.db`에 결과가 분리되어 저장됩니다.

### 3. Extract Paper Figures & Tables
탐색이 완료된 데이터베이스에서 논문 첨부용 고화질 그래프(.pdf)와 Ablation Study 결과(.csv)를 추출합니다.
```bash
python3 paper_logger.py
```
> `paper_figures/` 폴더가 생성되며 시각화 자료가 자동 저장됩니다.

---

## 📊 Evaluation & Visualization
`paper_logger.py` 실행 시 다음과 같은 논문용 시각화 결과를 획득할 수 있습니다.
1. **Population Distribution & Pareto Evolution:** 세대별 군집의 이동과 파레토 프론트의 향상 과정
2. **Hypervolume Convergence:** Multi-seed 기반 2D Exact Hypervolume(HV) 수렴 신뢰구간
3. **Ablation Study Table:** 제안 기법들(Multi-fidelity, Active Learning 등)의 기여도 분리 표

---

## 📝 Citation
(추후 논문 게재 시 Citation 정보 업데이트 예정)
```bibtex
@article{hw_aware_nas_yolo_2026,
  title={Hardware-Aware Asynchronous Multi-Objective NAS Pipeline for Autonomous Driving},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```
```
