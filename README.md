# veloAgent
## 📝 Overview
**veloAgent** is a deep generative framework designed to model cell state transitions using single-cell transcriptomics data and has the additional capability of integrating spatial information into the model. **veloAgent** estimates gene- and cell-specific transcriptional kinetics—capturing transcription, splicing, and degradation rates—with a deep neural network with connections informed by gene-gene association data. Using an agent-based model (ABM) **veloAgent** can simulate local cellular microenvironments in order to add spatial context to the initial velocity estimate. By combining transcriptional kinetics with spatial context, veloAgent provides a scalable and flexible solution for dissecting dynamic cellular processes across tissues, developmental stages, and conditions. Additionally, **veloAgent** includes a unique in silico perturbation module that lets users manipulate RNA velocity vectors, simulate regulatory interventions, and predict their impact on cell fate dynamics.

![Figure 1](https://github.com/user-attachments/assets/1447cfee-bab0-488d-b6fb-a2c3c318d305)

## 🔬 Why veloAgent
Traditional RNA velocity methods infer cell state transitions by modeling transcriptional dynamics from spliced and unspliced mRNA. However, existing tools:

- Neglect spatial context, missing key tissue-level organization.

- Struggle to scale to large or multi-batch datasets.

- Lack tools for simulating regulatory interventions in silico.

## 🔑 Key Features
- **Joint Kinetics Modeling**
Captures cell- and gene-specific transcription, splicing, and degradation rates.

- **Spatial Integration**
Models local cellular microenvironments using agent-based model simulations, improving velocity accuracy.

- **Scalable & Efficient**
Sublinear memory scaling allows veloAgent to handle large spatial and multi-batch datasets.

- **In Silico Perturbation**
Unique module for targeted manipulation of RNA velocity vectors, enabling simulation of regulatory interventions and prediction of downstream cell fate dynamics.

## 🧬 Applications
- Dissecting developmental trajectories of heterogeneous cell state datasets

- Studying dynamic tissue on temporal and spatiotemporal mappings

- Guiding experimental design through in silico cell fate manipulation

## 📦 Installation
### 1️⃣ Create a conda environment with Python 3.10
conda create -n myvirtenv python=3.10

### 2️⃣ Activate the environment
conda activate myvirtenv

### 3️⃣ Install PyTorch separately
Install PyTorch separately so you can choose the correct CPU or CUDA build for your platform.

For platform-specific CPU/GPU install instructions for PyTorch, see the official PyTorch guide:
https://pytorch.org/get-started/locally/

### 4️⃣ Install veloAgent directly from GitHub
```
git clone https://github.com/mcgilldinglab/veloAgent.git
cd veloAgent
pip install .
```

## 🧬 STRING DB setup for GG Net
For the gene-gene network utilities, users must download STRING DB files manually from:

https://string-db.org/cgi/download?sessionId=bXn82hPMr4fT

You need these three files for the species you plan to use:

- `protein.aliases`
- `protein.links`
- `protein.info`

At this point, `veloAgent` supports:

- `mouse`
- `human`
- `chicken`

These files should be stored under the `base` directory passed to `veloagent.load_protein_paths(species, base)`, inside a subdirectory named after the species.

Example:

```python
paths = veloagent.load_protein_paths(species="mouse", base="data/conn_mat")
```

Expected directory layout:

```text
data/conn_mat/
  mouse/
    10090.protein.info.v12.0.txt
    10090.protein.aliases.v12.0.txt
    10090.protein.links.v12.0.txt
  human/
    9606.protein.info.v12.0.txt
    9606.protein.aliases.v12.0.txt
    9606.protein.links.v12.0.txt
  chicken/
    9031.protein.info.v12.0.txt
    9031.protein.aliases.v12.0.txt
    9031.protein.links.v12.0.txt
```

## 📄 Release Notes
See [RELEASE_NOTES.md](RELEASE_NOTES.md).

## Contact
[Brent Yoon](mailto:ji.s.yoon@mail.mcgill.ca), [Vishvak Raghavan](mailto:vishvak.raghavan@mail.mcgill.ca), [Jun Ding](mailto:jun.ding@mcgill.ca)
