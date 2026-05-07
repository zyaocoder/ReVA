<div align="center">

  <h1 style="margin: 0; font-size: 1.8em;">
    ReVA
  </h1>

  [![arXiv](https://img.shields.io/badge/arxiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)]()
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Chrisathy/reva)

</div>

<div align="center">
  <b>
    <a href="https://zyaocoder.github.io/" target="_blank">Zhen Yao</a><sup>1*</sup>,
    <a target="_blank">Likai Wang</a><sup>1*</sup>,
    <a target="_blank">Yuming Yang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=7qMGAvQAAAAJ&hl=en" target="_blank">Zhihao Zheng</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=46_OVzoAAAAJ&hl=en&oi=ao" target="_blank">Bo Lang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=Pon9h8IAAAAJ&hl=en" target="_blank">Qiuyu Tang</a><sup>1</sup>,
    <a target="_blank">Jialu Sheng</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=GgAWQksAAAAJ&hl=en" target="_blank">Jingqi Xu</a><sup>2</sup>,
    <a target="_blank">Yuehai Yang</a><sup>1</sup>,
    <a target="_blank">Jumal Barker</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=bMapOgMAAAAJ&hl=en" target="_blank">Xiaowen Ying</a><sup>3</sup>,
    <a href="https://scholar.google.com/citations?user=SZBKvksAAAAJ&hl=en" target="_blank">Mooi Choo Chuah</a><sup>1</sup>,
  </b><br>
  
  <span style="font-size: 1em; color: #555;">
    Lehigh University<sup>1</sup><br>
    University of Southern California<sup>2</sup><br>
    Qualcomm AI Research<sup>3</sup>
  </span>

  <p style="color: #555; font-size: 0.9em; margin-top: 8px; margin-bottom: 0;">
    *Equal Contribution
  </p>
</div>

## 🔥 News

[2026-05-05] 🤗 Our dataset and code are available [here](https://huggingface.co/datasets/ReVA-Benchmark/ReVA).

## 📝 Table of Contents

- [🌟 Abstract](#-abstract)
- [🚀 Quick Start](#-quick-start)
- [📌 TODO](#-todo)
- [🪪 License](#-license)
- [✉️ Contact](#-contact)
<!-- - [🤗 Model Zoo](#-model-zoo) -->

## 🌟 Abstract

Multimodal Large Language Models (MLLMs) have demonstrated remarkable advances in remote sensing. However, existing remote sensing multimodal reasoning benchmarks exhibit two critical limitations: they rely on (i) template-driven questions, which is ill-posed for real-world scenarios; and (ii) static image inputs that fail to capture the inherent temporal nature of drone/UAV videos. This leaves systematic evaluation of remote sensing video reasoning largely unexplored. To address this gap, we introduce ReVA, a new dataset for remote sensing video question answering, designed to assess spatiotemporal, scene-centric, and reasoning-oriented capabilities of MLLMs. ReVA comprises 2,798 drone videos spanning 18 cities worldwide (580K frames) and 22K high-quality question–answer pairs across 11 challenging QA tasks. We develop a semi-automatic annotation pipeline that leverages Text LLMs and MLLMs for question-answer generation with human verification. We evaluate 21 proprietary and open-source Video LLMs on ReVA, exposing fundamental limitations of current models. These findings position ReVA as a critical benchmark for advancing MLLMs toward better remote sensing video understanding and temporal reasoning capabilities for real-world deployments.

## 🚀 Quick Start!

### Data preparation
Please follow [DATASET.md](assets/readmes/DATASET.md) to prepare the datasets. <br>

### Installation
Please follow [INSTALL.md](assets/readmes/INSTALL.md) to install the environment. <br>

### Training pipeline

We provide our training code [TRAIN.md](assets/readmes/TRAIN.md). Start training ReMoSense by following this instruction 🥳 <br>

<!-- ### Model Zoo

We release our ReMoSense model pretrained weights on Hugging Face with benchmark performance:

|Base LLM| Training data | Parameters | ReVA | Link |
|-----|--------|--------|------|------|------------|----------|------|
|Qwen2.5-VL-7B-Instruct| ReVA | 7B | 80.1 | 🤗 [HuggingFace]() |
 -->

## 📌 TODO

- [x] Release the dataset.
- [x] Release the training code.
- [ ] Release our model's pretrained weights.
- [ ] Support more VLMs as the base models.

## 🪪 License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.

## ✉️ Contact


If you're have any questions about the code,please feel free to open an github issue in the repo. 😊

For collaboration opportunities or other questions, feel free to send us an email at zhenyaocv@gmail.com. 


## Citation

If you use this work in your research, please cite: