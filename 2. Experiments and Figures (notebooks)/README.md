# CalPerUn – Experiments and Figures

Code and notebooks accompanying the paper  
**Calibrating Perception Uncertainty for Autonomous Driving**

This folder contains all experiments, uncertainty extraction methods, and figure generation used in the paper.

---

## Notebooks Overview

### Main Methodology and Figures

1. FIERY - Figure - segmentation to inputs (and Fig 2)  
   Generates the input visualisation.

2. FIERY - Figures (Fig. 4–8)  
   Generates all calibration figures used in the paper and poster.  
   All required data for these figures is included.

---

### Uncertainty Extraction (Preprocessing)

The following notebooks generate uncertainty estimates used in later experiments.  
Data for these notebooks is not included.

- FIERY - _pre - Location uncertainty
- FIERY - _pre - Presence uncertainty
- FIERY - _pre - Undetected object uncertainty
- FIERY - _pre - Trajectory

---

### Downstream Task

3. FIERY - planning - get object calibrated results  
   - Applies calibrated uncertainty to a planning task.
4. FIERY - planning - trajectories (Figure 9 and Table 1)
   - Generates the trajectories, calculates uncalibrated and calibrated probabilities, and finds the results for Table 1.
5. FIERY - planning - get number of collisions.ipynb
   - Find how many potential collisions would be there if the best path is taken.

---

### Additional Experiments

6. FIERY - autocorrelation check (Fig 1).ipynb  
   Autocorrelation experiment.

---

## Code Utilities

- `helper.py`  
  Helper functions used across notebooks

- `binnings.py`  
  Calibration utilities, including Expected Calibration Error (ECE) and reliability diagrams

---

## Results Files

- NPZ result files required by the FIERY - Figures notebook
- Contain precomputed outputs used for figure generation

---

## Contact

**markus.kangsepp@ut.ee**
