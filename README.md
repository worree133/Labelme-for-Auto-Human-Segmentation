# Labelme-for-Auto-Human-Segmentation

## Features

- [x] Auto-contouring for human segmentation

## Requirements

- Ubuntu / macOS / Windows
- Python2 / Python3
- PyQt5

## Installation

```bash
git clone https://github.com/worree133/Labelme-for-Auto-Human-Segmentation.git
cd labelme
python setup.py build
pip install .
```

Create a info.txt in your current working directory to decide the label name.

In the info.txt, just type the label name of background in the first row and label name of foreground in the second row.

The sample info.txt can be checked in the source code.

## Acknowledgement

This repo is the fork of [wkentaro/labelme](https://github.com/wkentaro/labelme).
The models used:
- P3M-Net [JizhiziLi/P3M](https://github.com/JizhiziLi/P3M/tree/master/core)
- MatteFormer [webtoon/matteformer](https://github.com/webtoon/matteformer)
