# veloAgent
## 📝 Overview
**veloAgent** is a deep generative framework designed to model cell state transitions using single-cell transcriptomics data and has the additional capability of integrating spatial information into the model. **veloAgent** estimates gene- and cell-specific transcriptional kinetics—capturing transcription, splicing, and degradation rates—with a deep neural network with connections informed by gene-gene association data. Using an agent-based model (ABM) **veloAgent** can simulate local cellular microenvironments in order to add spatial context to the initial velocity estimate. By combining transcriptional kinetics with spatial context, veloAgent provides a scalable and flexible solution for dissecting dynamic cellular processes across tissues, developmental stages, and conditions. Additionally, **veloAgent** includes a unique in silico perturbation module that lets users manipulate RNA velocity vectors, simulate regulatory interventions, and predict their impact on cell fate dynamics.

![Figure 1]()

## 🔬 Why veloAgent
Traditional RNA velocity methods infer cell state transitions by modeling transcriptional dynamics from spliced and unspliced mRNA. However, existing tools:

- Neglect spatial context, missing key tissue-level organization.

- Struggle to scale to large or multi-batch datasets.

- Lack tools for simulating regulatory interventions in silico.

**veloAgent** overcomes these limitations by:

- Jointly estimating gene- and cell-specific transcriptional kinetics (transcription, splicing, and degradation rates).

- Integrating spatial information through agent-based simulations of local cellular microenvironments.

- Offering sublinear memory scaling for efficient analysis of large-scale spatial datasets.

- Including an in silico perturbation module to simulate regulatory interventions and predict their impact on cell fate dynamics.

## 🔑 Key Features
- **Joint Kinetics Modeling**
Captures cell- and gene-specific transcription, splicing, and degradation rates.

- **Spatial Integration**
Models local cellular microenvironments using agent-based simulations, improving velocity accuracy.

- **Scalable & Efficient**
Sublinear memory scaling allows veloAgent to handle large spatial and multi-batch datasets.

- **In Silico Perturbation**
Unique module for targeted manipulation of RNA velocity vectors, enabling simulation of regulatory interventions and prediction of downstream cell fate dynamics.

## 🧬 Applications
- Dissecting spatially organized developmental trajectories

- Studying dynamic tissue responses across conditions

- Guiding experimental design through in silico cell fate manipulation

## 📦 Installation
Coming soon...

## Contact
[Brent Yoon](mailto:ji.s.yoon@mail.mcgill.ca), [Vishvak Raghavan](mailto:vishvak.raghavan@mail.mcgill.ca), [Jun Ding](mailto:jun.ding@mcgill.ca)
