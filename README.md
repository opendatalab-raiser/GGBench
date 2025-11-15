# GGBench: A Geometric Generative Reasoning Benchmark 🎯

Official repository for the project "A Geometric Generative Reasoning Benchmark for Unified Multimodal Models"

[🌍 [Homepage](https://opendatalab-raiser.github.io/GGBench/)] [📜 [OpenReview Paper](#)] [🤗 [HF Datasets](https://huggingface.co/datasets/opendatalab-raiser/GGBench)] [💻 [GitHub Code](https://github.com/opendatalab-raiser/GGBench)]


## 📖 Study Overview

<p align="center">
  <img src="static/images/teaser.png" alt="Study overview" width="70%">
</p>

<p align="center"><em>Overview of GGBench: A Geometric Generative Reasoning Benchmark for Unified Multimodal Models.</em></p>

We introduce **GGBench**, a geometric generative reasoning benchmark purpose-built for unified multimodal models (UMMs). Unlike prior evaluations that treat discriminative understanding and unconstrained image generation separately, GGBench diagnoses whether a model can fuse language comprehension with precise visual construction. Geometric construction serves as an ideal testbed, revealing how well a system can actively reason and synthesize structured solutions across modalities.

We investigate a key question: ***Can unified multimodal models integrate reasoning with controlled visual synthesis?*** While modern UMMs can perceive and understand complex visual scenes, their **actual reliability in generative reasoning**—where language understanding must guide precise geometric construction—remains unverified.

We conduct a comprehensive evaluation across multiple dimensions including planning, middle process, and final result quality, introducing GGBench as a standardized benchmark for systematic generative reasoning assessment. Our findings reveal the current capabilities and limitations of UMMs in geometric generative reasoning tasks.

## 🔍 Deep-Dive Analysis

We provide comprehensive investigation of unified multimodal models to analyze their geometric generative reasoning potential, detailing representative successes, characteristic errors, and the conditions under which generative reasoning emerges, holds, or breaks.

Visit our [homepage](https://opendatalab-raiser.github.io/GGBench/) to see video demonstrations showing how different models solve geometric problems step by step.


## 🧐 Evaluation

### Download Dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/opendatalab-raiser/GGBench
```

### Run Evaluation

The evaluation script supports multiple evaluation dimensions including VLM-based text/image evaluation, mid-process evaluation, and image quality metrics (LPIPS, PSNR, SSIM).

1. Navigate to the `dataset/` directory  
2. Edit **line 52-53** in `evaluate.py` to add your Judge Model URL and API Key  
3. Configure `MODEL_OUTPUT_PATH` in `evaluate.py` to point to your model's output JSON file
4. Run: `python evaluate.py`  

Results will be saved to `eval_output/result.json` and aggregated scores to `eval_output/score.json`

### Evaluation Metrics

- **VLM-T**: Text-based step reasoning evaluation (1-5 scale)
- **VLM-I-Mid**: Middle process image quality evaluation (Step Accuracy, Process Consistency, Problem-Solution Accuracy)
- **VLM-I-Res**: Final result image quality evaluation (1-5 scale)
- **LPIPS ×10⁻²**: Learned Perceptual Image Patch Similarity
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM ×10⁻²**: Structural Similarity Index


## ⚖️ GGBench Benchmark

We curate GGBench, a comprehensive benchmark providing a standardized taxonomy and an evaluation protocol, enabling consistent and category-wise assessment beyond surface-level metrics.

<p align="center">
  <img src="static/images/timu_type3_01.png" alt="GGBench radar evaluation" width="35%">
  <img src="static/images/data_analysis_01.png" alt="GGBench category distribution" width="45%">
</p>


<p align="center"><em>Evaluation Radar Map and Category Distribution of GGBench.</em></p>

### Dataset Statistics

- **Total Samples**: 1,411 geometric construction problems
- **Categories**: Multiple geometric problem types including basic constructions, circle properties, geometric transformations, triangle properties, theorem applications, polygon properties, measurement & ratios, and locus construction
- **Evaluation Dimensions**: Planning, Middle Process, Final Result, and Overall Scores


## 📊 Leaderboard

Main results on GGBench. VLM-T and VLM-I denote step reasoning and final diagram quality, respectively. VLM-Avg averages middle and final stages. All values are percentages.

See the [full leaderboard](https://opendatalab-raiser.github.io/GGBench/#leaderboard) for detailed results across all evaluated models.

