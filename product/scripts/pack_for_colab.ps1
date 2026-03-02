# Pack everything Colab needs for ALL remaining k-fold runs
# Pitt (13 remaining), ESC-50 (15), EmoDB (15)
# Run from PROJECT root: powershell -ExecutionPolicy Bypass -File product\scripts\pack_for_colab.ps1

$dest = "colab_upload"
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
New-Item -ItemType Directory -Path $dest | Out-Null

# Training + model code
Copy-Item -Recurse product\training $dest\product\training
Copy-Item -Recurse product\models $dest\product\models

# All split CSVs
New-Item -ItemType Directory -Path $dest\product\artifacts\splits -Force | Out-Null
Copy-Item product\artifacts\splits\*.csv $dest\product\artifacts\splits\

# Existing completed Pitt results (so Colab skips them)
if (Test-Path product\artifacts\runs\pitt) {
    New-Item -ItemType Directory -Path $dest\product\artifacts\runs -Force | Out-Null
    Copy-Item -Recurse product\artifacts\runs\pitt $dest\product\artifacts\runs\pitt
}

# Spectrograms
$specBase = "product\audio_preprocessing\outputs"
New-Item -ItemType Directory -Path "$dest\$specBase" -Force | Out-Null

Write-Host "Copying Pitt spectrograms (3836 files)..."
Copy-Item -Recurse "$specBase\spectrograms_pitt" "$dest\$specBase\spectrograms_pitt"

Write-Host "Copying ESC-50 spectrograms (8000 files)..."
Copy-Item -Recurse "$specBase\spectrograms" "$dest\$specBase\spectrograms"

Write-Host "Copying EmoDB spectrograms (3210 files)..."
Copy-Item -Recurse "$specBase\spectrograms_emodb" "$dest\$specBase\spectrograms_emodb"

# Zip
$zipPath = "colab_all_kfold.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath }
Write-Host "Zipping..."
Compress-Archive -Path "$dest\*" -DestinationPath $zipPath
Remove-Item $dest -Recurse -Force

$size = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Done! Created $zipPath ($size MB)"
Write-Host "Upload this to the root of your Google Drive."
