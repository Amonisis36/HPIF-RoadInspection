# HPIF: Physics-Guided Inverse Rendering for Line-Scan Pavement Inspection

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.style=for-the-badge)
![Paper](https://img.shields.io/badge/Paper-Coming_Soon-blue?style=for-the-badge)

> **"Breaking the Bas-Relief Ambiguity via Spatial-Illumination Priors"** > Official PyTorch Implementation.

## 💡 Overview

This repository provides the official implementation of the **Hierarchical Prior Injection Framework (HPIF)**. We tackle the severe bas-relief ambiguity in active line-scan pavement inspection systems. By leveraging the rigid-body kinematics of vehicle-mounted multi-LED arrays, we pre-compute deterministic spatial-illumination physical priors and inject them into a Dual-Axial U-Net via Feature-wise Linear Modulation (FiLM). 

**Key Features:**
- 🚀 **Physics-Decoupled**: Replaces heavy, non-differentiable rendering engines with pre-computed, ultra-lightweight global illumination tensors.
- 🧬 **Dual-Axial Attention**: Eliminates macroscopic travel-direction bias while preserving microscopic topological high-frequency details.
- 📦 **Plug-and-Play**: Training and inference run instantly without complex environment setups or physical engine dependencies.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/Amonisis36/HPIF-Inverse-Rendering.git](https://github.com/Amonisis36/HPIF-Inverse-Rendering.git)
cd HPIF-Inverse-Rendering

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

We release the highly-controlled, synthetic physically-based rendered (PBR) dataset: **ActiveLight-PBR**.

1. Download the `ActiveLight-PBR_Release.zip` dataset from [https://doi.org/10.5281/zenodo.19603543].
2. Extract the dataset into your preferred data directory.
3. The dataset automatically includes the crucial `Global_Left_Prior.npy` and `Global_Right_Prior.npy` files.

---

## ⚡ Quick Inference (Demo)

Want to see the results immediately? Use our provided inference script on your own images.

1. Download our pre-trained model `Official_Best.pth` from [https://doi.org/10.5281/zenodo.19603543] and place it in the `checkpoints/` folder.
2. Run the prediction script:

```bash
python predict.py \
  --input demo_image.jpg \
  --checkpoint checkpoints/Official_Best.pth \
  --side L \
  --prior_dir /path/to/ActiveLight-PBR_Release \
  --out_dir ./results
```
*Note: `--side L` selects the left camera prior. Use `R` for the right camera. The output will be saved as visually intuitive color maps in the `./results` folder.*

---

## 🏃‍♂️ Training

To reproduce our results from scratch, simply point the training script to the extracted dataset:

```bash
python train.py \
  --data_dir /path/to/ActiveLight-PBR_Release \
  --batch_size 4 \
  --epochs 50 \
  --lr 2e-4
```
*The script handles 8-sequence slicing and active physics-prior mapping automatically. Checkpoints will be saved in `checkpoints/`.*

---

## 📊 Evaluation Metrics

We provide a rigorous pure-PyTorch physical evaluation toolkit in `common_utils.py`. Our metrics decompose 3D Normal errors into strictly physical components:
- **Total Angular Error (MAE)**
- **Zenith Error** (Depth magnitude)
- **Azimuth Error** (Directional structure)
- **High-Frequency Masked Metrics** (Preservation of cracks and potholes)

## 📈 Performance Note (Refactoring Bonus)

> **Note to Researchers & Reviewers:** > This official open-source release features a highly optimized and strictly decoupled data-loading pipeline. By utilizing pre-computed static physical priors (`Global_Prior.npy`) instead of on-the-fly rendering, we successfully eliminated runtime computational noise and hidden alignment inconsistencies. 
> 
> As a result of these engineering optimizations, users training the network from scratch using this repository will observe **even better performance** on core high-frequency details and structural distortion metrics than the baseline originally reported in our manuscript.

**Benchmark Comparison (Epoch 50 Evaluation):**

| Method | HighFreq (MAE) ↓ | Flat (MAE) ↓ | Dark (MAE) ↓ | Bright (MAE) ↓ | Macro (Tilt) ↓ | Nx SSIM ↑ | GMSD ↓ | Acc <11.25° ↑ | Acc <22.5° ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Original** *(Seq8 LightMod)* | 9.924 | 9.809 | 9.126 | **9.231** | **1.327** | **0.5338** | 0.1300 | 67.04% | 95.31% |
| **Official Release** *(This Repo)* | **9.854** | **9.769** | **9.017** | 9.290 | 1.403 | 0.5328 | **0.1292** | **67.32%** | **95.43%** |

*(Metrics are evaluated using the rigorous `eval_opensource.py` script provided in this repository. Bold indicates the superior metric. The Official Release demonstrates superior high-frequency damage preservation and lower structural distortion.)*

---

## 📜 Citation

If you find our work, code, or the ActiveLight-PBR dataset useful, please consider citing our paper:

```bibtex
@article{YourName2025HPIF,
  title={Physics-Guided Inverse Rendering for Line-Scan Pavement Inspection: Breaking the Bas-Relief Ambiguity via Spatial-Illumination Priors},
  author={Your Name and Co-authors},
  journal={TBD},
  year={2025}
}
```

## ⚖️ License
The code and dataset are released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license for academic purposes. Commercial usage requires authorization.
