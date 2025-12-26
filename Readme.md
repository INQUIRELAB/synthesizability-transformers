# AI-assisted Materials Synthesizability Prediction from Crystal Structure (Structure-Aware Transformers)

🧑‍🔬 **Authors**
- **Danial Ebrahimzadeh** (University of Oklahoma)
- **Sarah Sharif** (University of Oklahoma)
- **Nisha Geng** (University of Oklahoma)
- **Yaser Mike Banad** (University of Oklahoma) — *Corresponding Author*

📄 **Abstract (Paraphrased)**
Computational workflows can generate millions of hypothetical inorganic crystal structures, but only a small subset are experimentally realizable. This project implements transformer-based models trained directly on crystallographic structure representations to predict **materials synthesizability** beyond thermodynamic stability heuristics. We evaluate three transformer paradigms over the **Fourier-Transformed Crystal Properties (FTCP)** representation: (i) a domain-agnostic Feature Tokenizer Transformer (FT-T), (ii) a hierarchical windowed-attention Swin Transformer (SwinT), and (iii) a domain-informed Structure-Aware Transformer (SAT) that explicitly encodes crystallographic components. A weighted ensemble of SAT + SwinT provides the strongest performance and improves precision while significantly reducing false positives compared to common DFT stability screening rules.

🧾 **License**
This repository is released under the **MIT License**.

---

📦 **Dataset (Required)**
> **Important:** The primary dataset file is **>2 GB** and is **not included** in the GitHub repository.

✅ **Download the FTCP dataset and place it in `data/`: `ftcp_data.h5` (Hugging Face)**  
- Target path (inside this repository): `data/ftcp_data.h5`  
- Source:
  - `https://huggingface.co/datasets/danial199472/synthesizability-transformers/resolve/main/ftcp_data.h5`

Example (Linux/macOS):
```bash
mkdir -p data
wget -O data/ftcp_data.h5 "https://huggingface.co/datasets/danial199472/synthesizability-transformers/resolve/main/ftcp_data.h5"


🗂️ Repository Structure

.
├─ data/
│  ├─ ftcp_data.h5                      # (download separately; >2GB)
│  └─ mp_structures_with_synthesizability1.xlsx
├─ ft-t/
│  ├─ dataset_balanced_fixed.py
│  ├─ data_split_info.json
│  ├─ ft_transformer_model.py
│  ├─ save_data_split.py
│  └─ train_ft.py
├─ SwinT/
│  ├─ dataset_balanced_fixed.py
│  ├─ requirements.txt
│  ├─ save_data_split.py
│  ├─ swin_transformer_model.py
│  ├─ train_swin.py
│  └─ results/
│     └─ best_model.pth
├─ SAT/
│  ├─ train_model.py
│  ├─ configs/
│  │  └─ structure_transformer.yaml
│  ├─ src/
│  │  ├─ data/
│  │  │  └─ dataset.py
│  │  ├─ models/
│  │  │  └─ structure_transformer.py
│  │  └─ training/
│  │     └─ train.py
│  └─ results/
│     └─ best_model.pt
├─ Ensemble/
│  ├─ ensemble_model.py
│  └─ optimize_weights.py
├─ requirements.txt
└─ REPO-TREE.txt


⚙️ Installation

python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt


🧠 Models Included

FT-T (Feature Tokenizer Transformer): ft-t/

SwinT (Shifted-window hierarchical attention): SwinT/

SAT (Structure-Aware Transformer with component-wise encoding): SAT/

Weighted Ensemble (SAT + SwinT): Ensemble/


🚀 Quickstart (Typical Workflow)

Download dataset → place at data/ftcp_data.h5

Install requirements

Train or evaluate a model:

FT-T: ft-t/train_ft.py

SwinT: SwinT/train_swin.py

SAT: SAT/train_model.py

Ensemble: Ensemble/ensemble_model.py

Because training scripts may assume specific paths/configs, review the top of each script and adjust dataset paths if needed.


🔁 Ensemble Weights
The ensemble combines SAT and SwinT probabilities via a weighted average. Weight-search utilities are provided in:

Ensemble/optimize_weights.py


📬 Contact

Corresponding author: Yaser Mike Banad (bana@ou.edu)
First author: Danial Ebrahimzadeh (danial.ebrahimzadeh@ou.edu)