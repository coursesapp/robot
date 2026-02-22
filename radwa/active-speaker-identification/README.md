# TalkNet-ASD — Active Speaker Detection

This repository is based on [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) (ACM MM 2021), with an added **real-time webcam mode** for live active speaker detection.

> **Original Paper**: *Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection* [[Paper](https://arxiv.org/pdf/2107.06592.pdf)] [[Video](https://youtu.be/C6bpAgI9zxE)]

![overall.png](utils/overall.png)

---

## What's New in This Fork

- **Real-time webcam ASD** (`finaltalknet.py`) — detects active speakers live from your camera and microphone
- **Adjustable sensitivity threshold** — tune detection without retraining
- **Live audio level monitoring** — visual bar and stats on screen
- Green box = Speaking | Red box = Silent

---

## Dependencies

**Requirements**: Python 3.10, CUDA GPU, microphone + webcam

```bash
# Create environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirement.txt
```

---

## Real-Time Webcam (New Feature)

#### Download pretrained model

Download `pretrain_TalkSet.model` and place it in the project root:

👉 [Download pretrain_TalkSet.model](https://drive.google.com/file/d/1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea)

#### Run

```bash
python finaltalknet.py
```

Press `q` to quit.

#### Configuration

Edit these values at the top of `finaltalknet.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_ID` | `0` | Camera index (0 = default webcam) |
| `SPEAKING_THRESHOLD` | `-0.5` | Detection sensitivity |
| `AUDIO_RATE` | `16000` | Audio sample rate |

**Threshold guide:**
- `-0.7` Very sensitive
- `-0.5` Balanced (recommended)
- `-0.3` Less sensitive
- `0.0` Strict

#### Troubleshooting

**Audio level too low:**
1. Right-click speaker icon in Windows taskbar
2. Open Sound settings → Input → Device properties
3. Increase microphone volume to 80-100%

---

## Demo on Video File (Original)

Put your video in the `demo` folder (e.g. `demo/001.mp4`) then run:

```bash
python demoTalkNet.py --videoName 001
```

Output: `demo/001/pyavi/video_out.avi` with green/red boxes marking speakers.

---

## TalkNet in AVA-ActiveSpeaker Dataset

#### Data preparation

```bash
python trainTalkNet.py --dataPathAVA AVADataPath --download
```

#### Training

```bash
python trainTalkNet.py --dataPathAVA AVADataPath
```

#### Evaluation with pretrained model

```bash
python trainTalkNet.py --dataPathAVA AVADataPath --evaluation
```

Pretrained model (`pretrain_AVA.model`) downloads automatically. Achieves `mAP: 92.3` on validation set.

---

## TalkNet in TalkSet & Columbia ASD Dataset

```bash
python demoTalkNet.py --evalCol --colSavePath colDataPath
```

| Name | Bell | Boll | Lieb | Long | Sick | Avg. |
|------|------|------|------|------|------|------|
| F1   | 98.1 | 88.8 | 98.7 | 98.0 | 97.7 | 96.3 |

---

## Project Structure

```
TalkNet-ASD/
├── finaltalknet.py      # Real-time webcam ASD (new)
├── demoTalkNet.py       # Original video file demo
├── talkNet.py           # TalkNet model class
├── trainTalkNet.py      # Training script
├── dataLoader.py        # Data loader
├── loss.py              # Loss functions
├── requirement.txt      # Dependencies
├── model/               # Model architecture + face detector
└── utils/               # Utility scripts
```

---

## Citation

```bibtex
@inproceedings{tao2021someone,
  title={Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection},
  author={Tao, Ruijie and Pan, Zexu and Das, Rohan Kumar and Qian, Xinyuan and Shou, Mike Zheng and Li, Haizhou},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  pages = {3927-3935},
  year={2021}
}
```

---

## Acknowledgements

- Original TalkNet-ASD: [TaoRuijie/TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD)
- Audio encoder: [clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer)
- Demo visualization: [joonson/syncnet_python](https://github.com/joonson/syncnet_python)
- Visual frontend: [lordmartian/deep_avsr](https://github.com/lordmartian/deep_avsr)
