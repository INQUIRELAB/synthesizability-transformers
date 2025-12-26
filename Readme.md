# AI-assisted materials synthesizability prediction from crystallographic structure using structure-aware transformers

🧩 **Overview**  
This repository contains the official codebase for **structure-based materials synthesizability prediction** using transformer architectures operating on the **Fourier-Transformed Crystal Properties (FTCP)** representation. The goal is to learn synthesizability patterns directly from crystallographic structure (rather than relying only on thermodynamic stability heuristics), enabling improved screening of hypothetical inorganic crystals and more reliable prioritization for experimental validation.

🧑‍🔬 **Authors**
- **Danial Ebrahimzadeh** (University of Oklahoma)
- **Sarah Sharif** (University of Oklahoma)
- **Nisha Geng** (University of Oklahoma)
- **Yaser Mike Banad** (University of Oklahoma) — *Corresponding Author* (bana@ou.edu)

📄 **Abstract**
Computational methods can now predict millions of hypothetical crystalline materials with desirable properties, yet only a tiny fraction can be experimentally synthesized. Traditional screening relies on thermodynamic stability calculations, which achieve modest accuracy ( 60%) in distinguishing synthesizable from non-synthesizable phases. Here, we demonstrate that transformer neural networks trained on crystallographic structure data substantially outperform energy-based approaches for synthesizability prediction. We compare three transformer architectures processing Fourier-transformed crystal properties (FTCP) representation: a domain-agnostic design, a hierarchical spatial model, and a structure-aware architecture that explicitly decomposes crystallographic components. An ensemble combining the two best-performing models achieves 90.88% accuracy and 96.47% ROC-AUC, providing twofold higher precision and fivefold fewer false positives than DFT stability criteria. Case study of twelve lithium niobate polymorphs shows that our weight-optimized ensemble prediction successfully discriminates among structurally distinct variants with near-identical thermodynamics, demonstrating the benefit of combining complementary architectural features. These results establish that learning from experimental synthesis outcomes captures the complex interplay of thermodynamics, kinetics, and synthesis pathways governing materials realizability.

🧾 **License**
This repository is released under the **MIT License**.

---

📦 **Dataset (Required)**
> **Important:** The primary dataset file is **>2 GB** and is **not included** in the GitHub repository.

✅ **Download the FTCP dataset and place it here:** `data/ftcp_data.h5`  
- **Hugging Face (click to download):** [ftcp_data.h5](https://huggingface.co/datasets/danial199472/synthesizability-transformers/resolve/main/ftcp_data.h5)

If you already have the file locally (example Windows location):
- `C:\Users\dania\Research\SyntheFormer\Nature_MC\files\data\ftcp_data.h5`  
Copy it into this repository as:
- `./data/ftcp_data.h5`

---

🗂️ **Repository structure**
```text
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

````

---
⚙️ Installation
Create a virtual environment and install dependencies:
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt


🧠 Models included

FT-T (Feature Tokenizer Transformer): ft-t/

SwinT (Shifted-window hierarchical attention): SwinT/

SAT (Structure-Aware Transformer with component-wise encoding): SAT/

Weighted Ensemble (SAT + SwinT): Ensemble/


🚀 Quickstart

Download the dataset and place it at: data/ftcp_data.h5

Install requirements

Train or run a model:

FT-T: ft-t/train_ft.py

SwinT: SwinT/train_swin.py

SAT: SAT/train_model.py

Ensemble: Ensemble/ensemble_model.py

Note: Some scripts may assume specific local paths or configurations. If needed, adjust dataset paths at the top of each script/config.

🔁 Ensemble weights
The ensemble combines SAT and SwinT probabilities via a weighted average. Weight-search utilities are provided in:

Ensemble/optimize_weights.py

📬 Contact

Corresponding author: Yaser Mike Banad — bana@ou.edu

First author: Danial Ebrahimzadeh — danial.ebrahimzadeh@ou.edu