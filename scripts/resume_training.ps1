Write-Host "========================================"
Write-Host "   JARVIS 700M RESUME TRAINING"
Write-Host "========================================"

$env:CUDA_VISIBLE_DEVICES="0"

accelerate launch --mixed_precision=fp16 training/train.py --resume_from_checkpoint checkpoints/

Write-Host "========================================"
Write-Host "   RESUME FINISHED"
Write-Host "========================================"