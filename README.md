# Labelme-for-Auto-Human-Segmentation

## Features

- [x] Auto-contouring for human segmentation

## Requirements

- Ubuntu / macOS / Windows
- Python3.6
- PyQt5

## Installation

### Anaconda

First, you need to download the official labelme.

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```bash
# python3 version 3.6
conda create --name=labelme python=3
source activate labelme
pip install labelme
```

Then, download the source code.

Replacing the original content of labelme package with the source code.

The path of the labelme package is "C:\Users\user\anaconda3\envs\pythonProject\Lib\site-packages\labelme".

Install the required packages listed in the requirements.txt.

Create a info.txt in your current working directory to decide the label name.

In the info.txt, just type the label name of background in the first row and label name of foreground in the second row.

The sample info.txt can be checked in the source code.

## Acknowledgement

This repo is the fork of [wkentaro/labelme](https://github.com/wkentaro/labelme).
The models used:
- P3M-Net [JizhiziLi/P3M] (https://github.com/JizhiziLi/P3M/tree/master/core)
- MatteFormer [webtoon/matteformer](https://github.com/webtoon/matteformer)
