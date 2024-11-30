# LLM Self Correction

This is the official implementation for [On the Intrinsic Self-Correction Capability of LLMs: Uncertainty and Latent Concept](https://arxiv.org/pdf/2406.02378). A summarization can be found at [here](https://haitaomao.github.io/categories/ResearchSummary/#section32)

![Example Image](https://pic1.zhimg.com/80/v2-de98cc1b608c8114a83774108e92dfd2_1440w.png)


## Abstract

Large Language Models (LLMs) are able to improve their responses when instructed to do so, a capability known as elf-correction. When instructions provide only the task’s goal without specific details about potential issues in the response, LLMs must rely on their internal knowledge to improve response quality, a process referred to as intrinsic self-correction. The empirical success of intrinsic self-correction is evident in various applications, but how and why it is effective remains unknown. In this paper, we unveil that intrinsic self-correction can be progressively improved, allowing it to pproach a converged state. Our findings are verified in: (1) the scenario of multi-round question answering, by comprehensively demonstrating that intrinsic self-correction can progressively introduce performance gains through iterative interactions, ultimately converging to stable performance; and (2) the context of intrinsic self-correction for enhanced morality, in which we provide empirical evidence that iteratively applying instructions reduces model uncertainty towards convergence, which then leads to convergence of both the calibration error and self-correction performance, ultimately resulting in a stable state of intrinsic self-correction. Furthermore, we introduce a mathematical formulation and a simulation task indicating that the latent concepts activated by self-correction instructions drive the reduction of model uncertainty. Based on our experimental results and analysis of the convergence of intrinsic self-correction, we reveal its underlying mechanism: consistent injected instructions reduce model uncertainty which yields converged, improved performance.

## Environment Installation

```
git clone https://github.com/HaitaoMao/LLM-self-correction.git  
cd LLM-self-correction
conda env create -n SelfCorrect -f environment.yml
conda activate SelfCorrrect
```

### Data
All the required data can be found in dataset. Notably, debiasing data is splited into ten equal split for faster sanity check.


### How to run

python3 main_exp/prompt_`task`.py

Key Arguments Interpretation:
- `--benchmark`: Specifies the benchmark dataset (`BBQ`, `WMT`, `REALTOXITY`).
- `--instruct_type`: Defines the instruction level (`INSTRUCT_LEVEL0`, `INSTRUCT_LEVEL1`, `INSTRUCT_GROUNDTRUTH`).
- `--questions`: Path to the question dataset (e.g., `datasets/BBQ/sexual_orientation.txt`).
- `--llm`: Name of the large language model to use (e.g., `mistral`).
- `--name_or_path`: Path or name of the model checkpoint (e.g., `alignment-handbook/zephyr-7b-sft-full`).
- `--prompt_len`: Path to the prompt length configuration (e.g., `alignment-handbook/zephyr-7b-sft-full`).


## Cite us

If you found this work useful, please cite our paper

```
@article{liu2024intrinsic,
  title={On the Intrinsic Self-Correction Capability of LLMs: Uncertainty and Latent Concept},
  author={Liu, Guangliang and Mao, Haitao and Cao, Bochuan and Xue, Zhiyu and Johnson, Kristen and Tang, Jiliang and Wang, Rongrong},
  journal={arXiv preprint arXiv:2406.02378},
  year={2024}
}
```