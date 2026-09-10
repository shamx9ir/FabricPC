# Start JAX CUDA Jupyter server in WSL2
# Run this script, then connect Windsurf to the printed URL.
#
# Usage:
#   .\start_jax_cuda_kernel.ps1
#
# In Windsurf: click kernel picker -> "Existing Jupyter Server" -> paste the URL

Write-Host ""
Write-Host "Starting JAX CUDA Jupyter server in WSL2..." -ForegroundColor Cyan
Write-Host "Copy the URL below and paste it into Windsurf's kernel picker." -ForegroundColor Yellow
Write-Host "(Kernel picker -> 'Select Another Kernel' -> 'Existing Jupyter Server')" -ForegroundColor Yellow
Write-Host ""

wsl -d Ubuntu -e /home/shamir/jax-cuda-venv/bin/jupyter notebook --no-browser --port=8888
