# CalPerUn

Official repository for the paper  
**Calibrating Perception Uncertainty for Autonomous Driving**

This repository contains code and experiments for training perception models and evaluating calibrated uncertainty in autonomous driving scenarios.

---

## Repository Structure

### 1. Training
**Folder:** `1. FIERY train and validate (pedestrians)`

Contains code and instructions for training the FIERY model on pedestrian data using the NuScenes dataset.

- Based on the official FIERY implementation
- Modified to support pedestrian-focused training
- Pretrained model checkpoints used in the paper are provided in the `checkpoints/` folder

---

### 2. Experiments
**Folder:** `2. Experiments and Figures (notebooks)`

Contains all experiments, uncertainty extraction methods, and figure generation notebooks, including:

- Calibration experiments
- Autocorrelation analysis
- Downstream planning task evaluation

---

## Citation

If you use this work, please cite:

```bibtex
@article{kangsepp2026calibrating,
  author = {K{\"a}ngsepp, Markus and Kull, Meelis},
  title = {Calibrating Perception Uncertainty for Autonomous Driving},
  journal = {International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems},
  volume = {34},
  number = {05},
  pages = {621--645},
  year = {2026},
  doi = {10.1142/S0218488526500212},
}
```




---

## Contact

For questions or issues, please contact:  
**markus.kangsepp@ut.ee**
