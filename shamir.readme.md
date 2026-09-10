# Shamir's Dev Notes

## Running notebooks in VS Code (WSL GPU kernel)

VS Code on Windows **cannot launch WSL kernels directly**. You must start a
Jupyter server inside WSL first, then connect VS Code to it.

### Step 1 — Install Jupyter (one-time)

```bash
wsl -d Ubuntu -u shamir bash --norc --noprofile -c '/mnt/e/projects/FabricPC/.venv-wsl/bin/pip install jupyter notebook 2>&1 | tail -3'
```

### Step 2 — Start the Jupyter server

```bash
wsl -d Ubuntu -u shamir bash --norc --noprofile -c '/mnt/e/projects/FabricPC/.venv-wsl/bin/jupyter notebook --no-browser --port=8888 --notebook-dir=/mnt/e/projects/FabricPC/examples 2>&1'
```

Leave this terminal open. Copy the URL printed in the output, e.g.:
`http://localhost:8888/?token=abc123...`

### Step 3 — Connect VS Code to the running server

1. Open a notebook in VS Code.
2. Click the kernel picker (top-right corner of the notebook).
3. Choose **"Select Another Kernel..."** → **"Existing Jupyter Server..."**
4. Paste the `http://localhost:8888/?token=...` URL.
5. Select the **"Python (fabricpc GPU)"** kernel.

> If the kernel still does not respond after connecting, restart it with
> **Kernel → Restart Kernel** in VS Code (not the server).
