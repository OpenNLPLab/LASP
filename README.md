# Linear Attention Sequence Parallelism (LASP)

<p align="center">
💻 <a href="https://github.com/OpenNLPLab/LASP" target="_blank">GitHub </a> •
💬 <a href="https://discord.gg/JEU3nTcWKC" target="_blank">Discord</a> •
💬 <a href="./images/contact_me_qr.png" target="_blank">WeChat</a>
</p>

This repo provides the implementation of Linear Attention Sequence Parallelism (https://arxiv.org/abs/).

![LASP Fig1](./images/LASP_fig1.png)

Sequence Parallel (SP) serves as a prevalent strategy to handle long sequences that exceed the memory limit of a single GPU. However, existing SP methods do not take advantage of linear attention features, resulting in sub-optimal parallelism efficiency and usability for linear attention-based language models. In this paper, we introduce Linear Attention Sequence Parallel (LASP), an efficient SP method tailored to linear attention-based language models. Specifically, we design an efficient point-to-point communication mechanism to leverage the right-product kernel trick of linear attention, which sharply decreases the communication overhead of SP. We also enhance the practical efficiency of LASP by performing kernel fusion and intermediate state caching, making the implementation of LASP hardware-friendly on GPU clusters. Furthermore, we meticulously ensure the compatibility of sequence-level LASP with all types of batch-level data parallel methods, which is vital for distributed training on large clusters with long sequences and large batches.
We conduct extensive experiments on two linear attention-based models with varying sequence lengths and GPU cluster sizes. LASP scales sequence length up to 4096K using 128 A100 80G GPUs on 1B models, which is 8$\times$ longer than existing SP methods while being significantly faster.


## Installation
```
pip install -e .
```
The code has been test under the following environment:
```
triton                   2.0.0
triton-nightly           2.1.0.dev20230728172942
```
You can use the following command to install:
```
pip install triton==2.0.0
pip install triton-nightly==2.1.0.dev20230728172942 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/
```

## Example of runing LASP
```
cd tests
bash script.sh
```

## Benchmark


## Todo


## Citation
If you find our work useful, please cite the following papers:
```
@misc{lasp,
      title={Linear Attention Sequence Parallelism},
      author={Weigao Sun and Zhen Qin and Dong Li and Xuyang Shen and Yu Qiao and Yiran Zhong},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
