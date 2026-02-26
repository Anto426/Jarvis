Write-Host "========================================"
Write-Host "   JARVIS 700M PRETRAINING START"
Write-Host "========================================"

$env:CUDA_VISIBLE_DEVICES="0"

accelerate launch --mixed_precision=fp16 training/train.py

Write-Host "========================================"
Write-Host "   PRETRAINING FINISHED"
Write-Host "========================================"