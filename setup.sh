#!/bin/bash
set -euo pipefail

echo "==> Cloning optimal-stopping repo..."
git clone https://github.com/catchshashank/optimal-stopping.git
cd optimal-stopping

echo "==> Installing dependencies..."
pip install -q -r requirements.txt

echo "==> Setting HF_TOKEN..."
if [ -z "${HF_TOKEN:-}" ]; then
    read -rsp "Paste your Hugging Face token: " HF_TOKEN
    echo
    export HF_TOKEN
else
    echo "HF_TOKEN already set in environment."
fi

echo "==> Done! Launch Jupyter with:"
echo "    jupyter notebook --ip=0.0.0.0 --no-browser --allow-root"
