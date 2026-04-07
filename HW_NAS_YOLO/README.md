flowchart TD
    %% 스타일 정의
    classDef engine fill:#f9f0ff,stroke:#d0bdf4,stroke-width:2px;
    classDef process fill:#eef2ff,stroke:#a5b4fc,stroke-width:2px;
    classDef predictor fill:#fffbeb,stroke:#fde047,stroke-width:2px;
    classDef trt fill:#fee2e2,stroke:#fca5a5,stroke-width:2px;
    classDef eval fill:#ecfdf5,stroke:#6ee7b7,stroke-width:2px;
    classDef db fill:#f3f4f6,stroke:#9ca3af,stroke-width:2px;

    %% 메인 노드
    A[<b>Evolution Engine</b><br>NSGA-II / Uniform Crossover<br>Immigrant Injection] :::engine
    B[<b>Architecture Decoder</b><br>YOLO11n + Attention<br>Weight Surgeon] :::process
    
    C[<b>Latency Predictor</b><br>Random Forest] :::predictor
    D{Uncertainty<br>Top 20%?} :::predictor
    
    E[<b>Active Learning</b><br>TensorRT FP16 실측] :::trt
    F[(Replay Buffer<br>Calibration)] :::trt
    
    G[<b>Multi-Fidelity Evaluator</b><br>Ray Asynchronous Workers<br>10% Proxy Dataset] :::eval
    H[Stage 1: 3 Epochs] :::eval
    I[Stage 2: 15 Epochs] :::eval
    J[Stage 3: 50 Epochs] :::eval
    
    K[(<b>SQLite WAL</b><br>State Cache)] :::db
    
    L[<b>Pareto Archive</b><br>Exact 2D Hypervolume<br>Slope 결합] :::engine

    %% 흐름 (Flow)
    A -->|1. Generate Genomes| B
    B -->|2. Network Topology| C
    C -->|3. Predict Latency & Variance| D
    
    D -->|Yes (High Uncertainty)| E
    E --> F
    F -.->|Update Model| C
    
    D -->|No (Confident)| G
    E -->|Measured Latency| G
    
    G --> H
    H -->|Top 50%| I
    I -->|Top 50%| J
    J -->|mAP & Slope| K
    
    K --> L
    L -->|Feedback: Next Gen Parents| A
