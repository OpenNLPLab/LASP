# LASP

<p align="center">
💻 <a href="https://github.com/OpenNLPLab/LASP" target="_blank">GitHub </a> •
💬 <a href="https://discord.gg/JEU3nTcWKC" target="_blank">Discord</a> •
💬 <a href="./images/contact_me_qr.png" target="_blank">WeChat</a>
</p>

## Introduction
Official implementation of Linear Attention Sequence Parallelism (LASP) (https://arxiv.org/abs/).


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
