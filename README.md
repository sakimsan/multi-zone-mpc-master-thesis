<div align="center">

# 🏠 Predictive Control for Multi-Zone Buildings

<p align="center">
  <em>Development and Analysis of Predictive Control Strategies for Temperature Setbacks in Multi-Zone Buildings</em>
</p>

[![Thesis](https://img.shields.io/badge/Master's_Thesis-RWTH_Aachen-00549F?style=for-the-badge&logo=academia&logoColor=white)](./Thesis.pdf)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Modelica](https://img.shields.io/badge/Modelica-BESMod-FF6B35?style=for-the-badge&logo=sim&logoColor=white)](https://openmodelica.org/)
[![EnergyPlus](https://img.shields.io/badge/Simulation-EnergyPlus-76B900?style=for-the-badge&logo=energyplus&logoColor=white)](https://energyplus.net/)
[![CasADi](https://img.shields.io/badge/Optimizer-CasADi-FF4B4B?style=for-the-badge&logo=wolfram&logoColor=white)](https://web.casadi.org/)

<br/>

**Sakimsan Sriskandaruban** · Matr. 463953 · November 2025

*E.ON Energy Research Center | Institute for Energy Efficient Buildings and Indoor Climate (EBC)*  
*RWTH Aachen University*

📄 **[Read the Full Thesis](./Thesis.pdf)**

</div>

---

## 📖 About

This thesis develops and analyzes **Model Predictive Control (MPC) strategies** for temperature setbacks in multi-zone residential buildings. The core research question: do thermally **coupled** multi-zone models offer real advantages over **uncoupled** approaches — and at what cost?

A key contribution is an **automated framework** that generates physical MPC process models directly from an EnergyPlus IDF file, based on the VDI 6007 building model standard.

## 🔧 Simulation Stack

<div align="center">

|                                                                      | | |
|:--------------------------------------------------------------------:|:---:|:---:|
| <img src="images/energyplus_logo.png" height="50" alt="EnergyPlus"/> | <img src="images/spawn_logo.png" height="50" alt="Spawn of EnergyPlus"/> | <img src="images/modelica_logo.png" height="50" alt="Modelica"/> |
|                            **EnergyPlus**                            | **Spawn of EnergyPlus** | **Modelica / BESMod** |
|                     Building envelope simulation                     | Co-simulation coupling | Heat pump & hydraulics |

</div>

### 🔬 Evaluation Pipeline

The MPC is evaluated in two stages:

```
EnergyPlus IDF File
        │
        ▼
  Automated Framework  ──────────────────────────────────┐
  (VDI 6007 based)                                       │
        │                                                │
   ┌────┴──────────────┐                                 │
   │                   │                                 │
   ▼                   ▼                                 │
Uncoupled           Coupled                              │
Multi-Zone          Multi-Zone                           │
Model               Model                                │
   │                   │                                 │
   └────────┬──────────┘                                 │
            │                                            │
            ▼                                            │
     Centralized MPC                                     │
     (CasADi / AgentLib-MPC)                             │
            │                                            │
     ┌──────┴───────┐                                    │
     ▼              ▼                                    │
 Stage 1:        Stage 2:                                │
 Ideal Model     EnergyPlus + Spawn ◄────────────────────┘
 Evaluation      (+ BESMod / Modelica)
```


### 📊 Key Results

| Metric | Uncoupled Model | Coupled Model |
|--------|:--------------:|:-------------:|
| Optimization convergence | ✅ All scenarios | ⚠️ Fails with many temp. jumps |
| Comfort violations | Baseline | **+17.9%** |
| Energy consumption | Baseline | **+14.7%** |

> ⚠️ **Important finding:** Both models systematically overestimate the heating load due to incorrect handling of solar radiation and wall temperatures — identified as a key area for future work.

---

## 🗂️ Repository Structure

```
multi-zone-mpc-master-thesis/
│
├── 🐍 AgentLib-MPC/     # MPC agent framework (Python)
├── 📐 BESMod/           # Building Energy System Modeling (Modelica)
├── ⚡ BESGriConOp/      # Grid Control Optimization (Modelica)
├── 🔧 bes-rules/        # Simulation-based optimization framework (Python)
└── 📄 Thesis.pdf        # Full Master's thesis
```

<details>
<summary><b>AgentLib-MPC</b> — MPC Agent Library</summary>
<br>

Python library providing the MPC infrastructure used in this thesis:
- CasADi-based nonlinear optimization backends
- Centralized and ADMM-based distributed MPC
- Moving Horizon Estimation (MHE)
- Machine learning surrogate model integration
</details>

<details>
<summary><b>BESMod</b> — Building Energy System Modeling</summary>
<br>

Comprehensive Modelica library for building energy systems including heat pumps, hydraulic distribution, solar thermal, and electrical components. Used for the Modelica-side of the Spawn of EnergyPlus co-simulation.
</details>

<details>
<summary><b>BESGriConOp</b> — Grid Control Optimization</summary>
<br>

Modelica models for building energy system grid control. Contains building models, weather files (`.epw`), and SGReady control scenarios used in the thesis simulations.
</details>

<details>
<summary><b>bes-rules</b> — Simulation-Based Optimization Framework</summary>
<br>

Python framework for multi-objective design and control optimization of heat pump systems:
- MPC case studies: `studies/sfh_mpc_hom_monovalent_spawn/`
- Automated model generation from EnergyPlus IDF files
- Boundary condition handling (weather, electricity prices, CO₂)
- Post-processing and plotting utilities
</details>

---

## 🚀 Getting Started

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.12  | MPC framework & optimization |
| CasADi | latest  | Nonlinear optimization solver |
| EnergyPlus + Spawn | latest  | Building envelope simulation |
| Dymola  | —       | Modelica simulation |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sakimsan/multi-zone-mpc-master-thesis.git
cd multi-zone-mpc-master-thesis

# 2. Install requirements
pip pip install -r requirements.txt
```

### Running the MPC Case Study

```bash
# Single simulation run
python bes-rules/studies/sfh_mpc_hom_monovalent_spawn/run_mpc.py

# Full study (parallel)
python bes-rules/studies/sfh_mpc_hom_monovalent_spawn/run_mpc_fullStudy_parallel.py

# Plot results
python bes-rules/studies/sfh_mpc_hom_monovalent_spawn/plotting/plot_mpc.py
```

---

## 📚 Thesis Structure

| # | Chapter | Description |
|---|---------|-------------|
| 1 | Introduction | Motivation for predictive heating control in buildings |
| 2 | State of the Art | Heat pumps, thermal storage, building models, MPC literature |
| 3 | Modeling | Single-zone, uncoupled & coupled multi-zone models + parameterization |
| 4 | Case Studies | Evaluation: ideal model & EnergyPlus co-simulation |
| 5 | Discussion | Analysis of results and limitations |
| 6 | Summary & Outlook | Conclusions and future work |

---

## 👤 Author

**Sakimsan Sriskandaruban**

E.ON Energy Research Center  
Institute for Energy Efficient Buildings and Indoor Climate (EBC)  
RWTH Aachen University · Mathieustraße 10, 52074 Aachen

*Supervised by: Fabian Römer M.Sc., Tobias Spratte M.Sc., Univ.-Prof. Dr.-Ing. Dirk Müller*