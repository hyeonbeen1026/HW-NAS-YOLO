#!/bin/bash

echo "🚀 [Step 1] 시스템 필수 라이브러리 업데이트 및 설치 (YOLO/OpenCV용)..."
apt-get update -y
apt-get install -y libgl1-mesa-glx libglib2.0-0 screen

echo "📦 [Step 2] 파이썬 패키지 설치 중..."
# requirements.txt 기반으로 설치하되, 버전 충돌 시 최신 버전으로 덮어쓰도록 -U 옵션 사용
pip install -r requirements.txt
# 혹시 빠져있을 수 있는 핵심 패키지 강제 확인
pip install -U ray ultralytics scikit-learn

echo "🧹 [Step 3] 충돌 방지를 위한 임시 파일 청소..."
rm -f temp_arch_*.yaml temp_trt_arch_*.yaml temp_trt_arch_*.engine

echo "💾 [Step 4] 기존 DB 백업 (안전제일)..."
if [ -f "nas_global_cache.db" ]; then
    cp nas_global_cache.db nas_global_cache_backup_$(date +%F_%T).db
    echo "✅ 기존 18세대 DB 백업 완료!"
fi

echo "🔥 [Step 5] 백그라운드 탐색 루프 실행 (SSH가 끊겨도 유지됨)..."
# screen 이나 nohup을 사용하여 백그라운드에서 실행합니다.
nohup python main_loop.py > nas_pipeline_output.log 2>&1 &

echo "=========================================================="
echo "🎉 성공적으로 실행되었습니다! 파이프라인이 100% 가동 중입니다."
echo "👀 실시간 로그를 보려면 아래 명령어를 입력하세요:"
echo "tail -f nas_pipeline_output.log"
echo "=========================================================="