## Region-Aware Exposure Consistency Network for Mixed Exposure Correction (AAAI 2024)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2402.18217)

<hr />

#### 1. Introduction
This repository is the official implementation of the RECNet, where more implementation details are presented.

#### 2. Requirements
```
python=3.8.16
pytorch=2.0.1
torchvision=0.8
cuda=11.7
opencv-python
```

#### 3. Dataset Preparation
Refer to [ENC](https://github.com/KevinJ-Huang/ExposureNorm-Compensation) for details.

#### 4. Testing
```
bash scripts/test.sh
```
#### 5. Training
```a
bash scripts/train.sh
```
#### A1. convert multi-exp images to a video sequence
```bash

ls *.JPG | sed "s/^/file '/;s/$/'/" > filelist.txt # generate filelist.txt contains images with diverse exposures
ffmpeg -f concat -safe 0 -r 20 -i filelist.txt -c:v libx264 -crf 0 -preset veryslow -pix_fmt yuv444p output.mp4

python v2e.py -i output.mp4 --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder output/video --overwrite --pos_thres=.05 --neg_thres=.05 --sigma_thres=0.03 --dvs_aedat2 video.aedat --output_width=640 --output_height=480 --stop_time=1 --cutoff_hz=15

python v2e.py -i output.mp4 --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder output/video --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 video.aedat --output_width=640 --output_height=480 --stop_time=1 --cutoff_hz=15


python v2e.py -i /extension_space/exposure_related/event_rgb_sample_ds/output.mp4 --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder output/video --overwrite --pos_thres=.05 --neg_thres=.05 --sigma_thres=0.03 --dvs_aedat2 video.aedat --output_width=640 --output_height=480 --stop_time=1 --cutoff_hz=15 

249 event frames for 5 RGB frames

```


<hr />

#### Citation
If you find this work useful for your research, please consider citing:
``` 
@inproceedings{DBLP:conf/aaai/LiuFWM24,
  author       = {Jin Liu and
                  Huiyuan Fu and
                  Chuanming Wang and
                  Huadong Ma},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {Region-Aware Exposure Consistency Network for Mixed Exposure Correction},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {3648--3656},
  publisher    = {{AAAI} Press},
  year         = {2024},
  url          = {https://doi.org/10.1609/aaai.v38i4.28154},
  doi          = {10.1609/AAAI.V38I4.28154}
}
```

#### Acknowledgements
This repository is based on [ENC](https://github.com/KevinJ-Huang/ExposureNorm-Compensation) - special thanks to their code!






