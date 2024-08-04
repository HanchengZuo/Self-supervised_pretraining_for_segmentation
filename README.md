<p align="center">
    <h1 align="center">Self-Supervised_Pretraining_For_Segmentation</h1>
</p>
<p align="center">
    <em>Pre-text Pre-training -> Segmentation Fine-tuning pipeline</em>
</p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
   <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=default&logo=PyTorch&logoColor=white" alt="PyTorch">	
   <img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=default&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">

</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Data and Data Download](#data-and-data-download)
   - [Experiments Reproduction](#experiments-reproduction)
      - [1, Data Similarity](#1-data-similarity)
      - [2, Fine-tuning Data Size](#2-fine-tuning-data-size)
      - [3, Baseline Model](#3-baseline-model)
- [Acknowledgments](#acknowledgments)
</details>
<hr>

##  Overview

The Self-Supervised_Pretraining_For_Segmentation project is designed to optimize image segmentation models through self-supervised learning techniques, including extensive data augmentation and contrastive learning techniques. The project includes sophisticated neural network architectures with ViT as encoder and MLP as decoder and leverages contrastive learning durning pre-training. Core experiments within the project assess performance variations using diverse training datasets and different piplines.

---

##  Repository Structure

```sh
└── Self-supervised_pretraining_for_segmentation/
    ├── data_augmentation.py
    ├── data_utils.py
    ├── instruction.pdf
    ├── losses.py
    ├── metrics.py
    ├── models.py
    ├── requirements.txt
    ├── utils.py
    ├── experiments
    │   ├── compare_data_similarity_for_segmentation.ipynb
    │   ├── compare_data_similarity_for_segmentation.py
    │   ├── compare_pretrained_model_finetuning_sizes.ipynb
    │   ├── compare_pretrained_model_finetuning_sizes.py
    │   ├── compare_pretrained_with_baseline_transfer.ipynb
    │   ├── compare_pretrained_with_baseline.ipynb
    │   └── compare_pretrained_with_baseline.py
    ├── Images
    │   ├── compare_data_similarity_for_segmentation
    │   ├── compare_pretrained_model_finetuning_sizes
    │   └── compare_pretrained_with_baseline
    ├── datasets
    │   ├── data
    │   └── 102flowers
```

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [data_augmentation.py](data_augmentation.py) | `DataAugmentation.py` enables the enhancement of pre-training image datasets by applying various transformations and augmentations, supporting improved model generalization within the project, crucial for experiments involving pre-trained model evaluations and segmentation tasks. |
| [data_utils.py](data_utils.py)               | `Data_utils.py` establishes tools for handling image datasets crucial for contrastive learning, including classes for loading and transforming both augmented and original images. It provisions functionality for ensuring image correct format in both training and testing phases. |
| [losses.py](losses.py)                       | `Losses.py` introduces specialized loss functions, including ContrastiveLoss for pre-training, and DiceLoss for fine-tuning. |
| [metrics.py](metrics.py)                     | `Metrics.py` defines functions for model evaluation, specifically computing pixel-wise accuracy and Intersection over Union (IoU) score. |
| [models.py](models.py)                       | `Models.py` defines models essential for pre-text and segmentation tasks, including a Masked Autoencoder and a Decoder.   |
| [requirements.txt](requirements.txt)         | `Requirements.txt` specifies the necessary libraries for this project.  |
| [utils.py](utils.py)                         | Facilitates visualization of segmentation results within the SelfSup-SegFinetune project, enhancing comparison between ground-truth and predicted segmentation masks.                        |


<details open><summary>experiments</summary>

This folder contains the experiments in this project, in both python scripts and notebooks. All the results are shown in these notebooks.

| File                                 | Summary |
| ---                                  | --- |
| [compare_data_similarity_for_segmentation.ipynb](experiments\compare_data_similarity_for_segmentation.ipynb)                                                                             | This notebook contains the code to explore how similar the pre-training and fine-tuning data need to be for better segmentation performance, as well as the results. |
| [compare_data_similarity_for_segmentation.py](experiments\compare_data_similarity_for_segmentation.py)                                                                                   | This python script contains the code to explore how similar the pre-training and fine-tuning data need to be for better segmentation performance. |
| [compare_pretrained_model_finetuning_sizes.ipynb](experiments\compare_pretrained_model_finetuning_sizes.ipynb)                                                                           | This notebook contains the code to explore the effects of the fine-tuning data sized on the segmentation performance, as well as the results. |
| [compare_pretrained_model_finetuning_sizes.py](experiments\compare_pretrained_model_finetuning_sizes.py)                                                                                 | This python script contains the code to explore the effects of the fine-tuning data sizes on the segmentation performance. |
| [compare_pretrained_with_baseline_transfer.ipynb](experiments\compare_pretrained_with_baseline_transfer.ipynb) | This notebook contains code to compare the pre-text pre-training -> segmentation fine-tuning model with the model using fully supervised methods. Additionally, the pre-trained weights of `vit_b_16` downloaded from pytorch is directly used. In other words, pre-training is not down in this experiment. |
| [compare_pretrained_with_baseline.ipynb](experiments\compare_pretrained_with_baseline.ipynb)                                                                                             | This notebook contains code to compare the pre-text pre-training -> segmentation fine-tuning model with the model using fully supervised methods, as well as the results.  |
| [compare_pretrained_with_baseline.py](experiments\compare_pretrained_with_baseline.py)                                                                                                   | This python script contains code to compare the pre-text pre-training -> segmentation fine-tuning model with the model using fully supervised methods. |

</details>

</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.10.8`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the repository:
>
> ```console
> $ git clone https://github.com/HanchengZuo/Self-supervised_pretraining_for_segmentation.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd Self-supervised_pretraining_for_segmentation
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

###  Data and Data Download

**[Fine-tuning-data](https://www.robots.ox.ac.uk/%7Evgg/data/pets/)**: This `Oxford-IIIT Pet Dataset` data for fine-tuning across different experiments. This doesn't need to be manually downloaded, since this is handled in the code.

**[Pre-training-data-relevant](https://drive.google.com/file/d/1QHC18el6TemZBouOXOumJ-WEivq4blxN/view?usp=sharing)**: This dataset contains pets-related data. This needs to be mannally downloaded to `datasets/data`. We use this data during pre-training. The [augmented-data](https://drive.google.com/file/d/1–AAWSQTVH5opJy9vIfGxLaUdpAPB4If/view?usp=sharing), i.e., data after augmentation, is also available.

**[Pre-training-data-irrelevant](https://www.robots.ox.ac.uk/vgg/data/flowers/102/102flowers.tgz)**: This dataset contains flower images. This needs to be mannally downloaded to `datasets/102flowers`.

###  Experiments Reproduction

Three experiments were conducted, which can be reproduced by either python script or notebook.

#### 1, Data Similarity

The purpose of this experiment is to explore how similar the pre-training and
fine-tuning data need to be for better segmentation performance. All of the three datasets - Fine-tuning-data, Pre-training-data-relevant and Pre-training-data-irrelevant are used.

<h4>From <code>source</code></h4>

> Use python script
>
> Run the command below:
> ```console
> $ python experiments\compare_data_similarity_for_segmentation.py
> ```
> The visualization results will be stored in `Images\compare_data_similarity_for_segmentation`

> Use Jupyter Notebook
> 
> Run the cells in [experiments\compare_data_similarity_for_segmentation.ipynb](experiments\compare_data_similarity_for_segmentation.ipynb)
>
> All the results will be shown in the notebook.

#### 2, Fine-tuning Data Size

The purpose of this experiment is to explore the effects of the fine-tuning data sizes on the segmentation performance. Datasets Fine-tuning-data and Pre-training-data-relevant are used.

<h4>From <code>source</code></h4>

> Use python script
>
> Run the command below:
> ```console
> $ python experiments\compare_pretrained_model_finetuning_sizes.py
> ```
> The visualization results will be stored in `Images\compare_pretrained_model_finetuning_sizes`

> Use Jupyter Notebook
> 
> Run the cells in [experiments\compare_pretrained_model_finetuning_sizes.ipynb](experiments\compare_pretrained_model_finetuning_sizes.ipynb)
>
> All the results will be shown in the notebook.

#### 3, Baseline Model

The purpose of this experiment is to compare the pre-text pre-training -> segmentation fine-tuning model with the model using fully supervised methods. Datasets Fine-tuning-data and Pre-training-data-relevant are used.

<h4>From <code>source</code></h4>

> Use python script
>
> Run the command below:
> ```console
> $ python experiments\compare_pretrained_with_baseline.py
> ```
> The visualization results will be stored in `Images\compare_pretrained_with_baseline`

> Use Jupyter Notebook
> 
> Run the cells in [experiments\compare_pretrained_with_baseline.ipynb](experiments\compare_pretrained_with_baseline.ipynb)
>
> All the results will be shown in the notebook.

---

##  Acknowledgments

Thanks all the 8 members of our team.

---
