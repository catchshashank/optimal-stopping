- The notebooks directory contains the original, replication, and extension python notebooks of the optimal-stopping paper by Manzoor, Ascarza, and Netzer (2025).
- It also contains markdown documents detailing the code implementation, model learning, and backward induction process.

## Running locally

### 1. Clone the repo

```bash
git clone https://github.com/catchshashank/optimal-stopping.git
cd optimal-stopping
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scriptsctivate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training (recommended), install the CUDA-enabled PyTorch wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Set your tokens

```bash
cp .env.example .env
# Edit .env and fill in HF_TOKEN (and OPENAI_API_KEY if using GPT-4o)
```

Then load the variables before launching Jupyter:

```bash
export    # Linux / macOS
# Windows (PowerShell): Get-Content .env | ForEach-Object { =extglob -split '=',2; [System.Environment]::SetEnvironmentVariable([0],[1]) }
```

Or paste your token directly into Cell 4 of the notebook ().

### 5. Launch Jupyter

```bash
jupyter notebook notebooks/optimal-stopping.ipynb
```

### Hardware notes

| Backbone | VRAM needed | Training support |
|---|---|---|
| LLaMA 3.2 3B | ~8 GB | Yes (bf16) |
| Mistral 7B | ~16 GB | Yes (bf16) |
| GPT-4o | None (API) | No (zero-shot) |

Training with  requires a GPU that supports bfloat16 (e.g. A100, H100, RTX 3090+).  
For CPU-only inference set  in Cell 21.

### Dataset

The notebook downloads  automatically from the public  
[stopping-agents](https://github.com/emaadmanzoor/stopping-agents) repository — no manual download needed.
