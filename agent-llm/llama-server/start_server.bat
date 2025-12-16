@echo off
echo ========================================
echo Starting llama.cpp CUDA Server
echo ========================================
echo Model: qwen2-7b-instruct-q6_k
echo Context: 4096
echo GPU Layers: 28
echo Batch: 2048
echo Port: 8090
echo ========================================
echo.

llama-server.exe ^
  -m ..\models\qwen2-7b-instruct-q6_k.gguf ^
  -c 4096 ^
  --ctx-size 4096 ^
  -ngl 28 ^
  -b 2048 ^
  -ub 512 ^
  --port 8090 ^
  --host 0.0.0.0 ^
  -t 4 ^
  -fa on

pause