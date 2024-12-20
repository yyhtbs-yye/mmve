# 准备 Vimeo90K-triplet 数据集

<!-- [DATASET] -->

```bibtex
@article{xue2019video,
  title={Video Enhancement with Task-Oriented Flow},
  author={Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  journal={International Journal of Computer Vision (IJCV)},
  volume={127},
  number={8},
  pages={1106--1125},
  year={2019},
  publisher={Springer}
}
```

训练集和测试集可以从 [此处](http://toflow.csail.mit.edu/) 下载。

Vimeo90K-triplet 数据集包含了如下所示的 `clip/sequence/img` 目录结构：

```text
mmve
├── mmve
├── tools
├── configs
├── data
│   ├── vimeo_triplet
│   │   ├── tri_testlist.txt
│   │   ├── tri_trainlist.txt
│   │   ├── sequences
│   │   │   ├── 00001
│   │   │   │   ├── 0001
│   │   │   │   │   ├── im1.png
│   │   │   │   │   ├── im2.png
│   │   │   │   │   └── im3.png
│   │   │   │   ├── 0002
│   │   │   │   ├── 0003
│   │   │   │   ├── ...
│   │   │   ├── 00002
│   │   │   ├── ...
```
