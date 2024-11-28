$batch = $args[0]
$epochs = 20..180 | Where-Object { $_ % 2 -eq 0 }

# Load Python venv
.\venv\Scripts\activate.ps1

# Check cuda
Write-Output "CUDA Support"
python tf_test.py

# Batch Train
foreach ($epoch in $epochs) {
    # Train
    Write-Output "-------------------------------"
    $tick = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "[$tick] Train: B$batch, E$epoch"
    python train.py --batch $batch --epoch $epoch --cli
}
