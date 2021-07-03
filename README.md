# Stage1 - Image Classification

<div align="center">
    <h1>Formula Image Latex Recognition</h1>
    <img src="assets/logo.png" alt="logo"/>
    <br/>
    <img src="https://img.shields.io/github/stars/pstage-ocr-team6/ocr-teamcode?color=yellow" alt="Star"/>
    <img src="https://img.shields.io/github/forks/pstage-ocr-team6/ocr-teamcode?color=green" alt="Forks">
    <img src="https://img.shields.io/github/issues/pstage-ocr-team6/ocr-teamcode?color=red" alt="Issues"/>
    <img src="https://img.shields.io/github/license/pstage-ocr-team6/ocr-teamcode" alt="License"/>
</div>

---

## 📝 Table of Contents

- [Latex Recognition Task](#-latex-recognition-task)
- [File Structure](#-file-structure)
  - [Code Folder](#code-folder)
  - [Dataset Folder](#dataset-folder)
- [Getting Started](#-getting-started)
  - [Installation](#installation)
  - [Download Dataset](#download-dataset)
  - [Dataset Setting](#dataset-setting)
  - [Create .env for wandb](#create-env-for-wandb)
  - [Config Setting](#config-setting)
- [Usage](#-usage)
  - [Train](#train)
  - [Inference](#inference)
- [Demo](#-demo)
- [References](#-references)
- [Contributors](#-contributors)
- [License](#-license)

---

## ➗ Latex Recognition Task

<div align="center">
  <img src="assets/competition-overview.png" alt="Competition Overview"/>
</div>

수식 인식(Latex Recognition)은 **수식 이미지에서 LaTeX 포맷의 텍스트를 인식하는 태스크**로, 문자 인식(Character Recognition)과 달리 수식 인식의 경우 `좌 → 우` 뿐만 아니라 Multi-line에 대해서 `위 → 아래`에 대한 순서 패턴 학습도 필요하다는 특징을 가집니다.

<br/>

## 📁 File Structure

### Code Folder

```text
src/
│
├── config/                   
│   ├── __init__.py
│   ├── parser.py             # parse yaml file to config object
│   ├── tree.py               # config object
│   └── default.py            # default yaml config file
│
├── data/                     
│   ├── __init__.py           
│   ├── argment.py            # augmentations (most are unused)
│   ├── dataset.py            # datasets
│   └── functional.py         # helper functions for dataset
│
├── exec/
│   ├── __init__.py
│   ├── lr_scheduler.py       # custom learning rate scheduler
│   └── trainer.py            # trainer for training, validation, 
│                             # and creating submission file
├── modules/
│   ├── __init__.py
│   ├── models/               # define or load models
│   │   ├── __init__.py
│   │   ├── basic.py
│   │   ├── classifier.py
│   │   ├── conv.py
│   │   └── presets.py
│   │
│   ├── functional.py         # helper functions for modules
│   └── loss.py               # loss functions
│                             
└── utils/
    ├── __init__.py
    ├── log.py                # functionalities to log on csv and plot
    ├── seed.py               # fix random seeds
    └── utils.py              # utilities

data_tools/
└── copy_data_2.py            # rename and rearange train images
   
```


<br/>

## How to Use

### Install Requiements

```shell
pip install -r requirements.txt
```

### Configurations

학습 시 사용하는 config 파일은 `yaml`파일로 학습 목표에 따라 다음과 같이 설정해주세요.

```yaml
system: &default_system_config
  device: "cuda:0"
  num_workers: 8


path: &default_path_config
  input: '/opt/ml/input'
  train: '/opt/ml/input/data/train'
  test: '/opt/ml/input/data/eval'
  valid: 
  result: '/opt/ml/output/result'
  model: '/opt/ml/output/model'

  logs: '/opt/ml/output/logs'
  configs: '/opt/ml/output/configs'
  


data: &default_data_config
  valid_ratio: 0.2
  upscaling: False
  train_cutmix: False
  cutmix_kernel: 256
  soften_age: False
  custom_augment:
  preprocess: False
  crop_size:
  resize:
  sampler: 
  batch_size: 32


train: &default_train_config
  lr:
    base: 1.0e-3
    classifier: 0
    scheduler: 
    scheduler_manual:
      active: False
      averaging_few: 3
      divider: 10
      low_limit: 1.0e-8
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]
  momentum: 0.9
  nesterov: False
  criterion: CrossEntropyLoss
  optimizer:
    name: AdamP
  
  num_epochs: 10
  valid_iter_limit: 0
  
  valid_min: 1
  test_min: 1
  save_min: 1

  valid_period: 1
  test_period: 0
  save_period: 1
  plot_period: 5

  valid_min_acc: 0
  test_min_acc: 1
  save_min_acc: 0
  plot_min_acc: 0

  #wandb options
  logger: manual


model: &default_model_config
  arc: EfficientNetWithMultihead-pretrained
  name: dummymodel
  state_path: 

  classifying: all
teacher:
  active: False
  state_path: 

```

<br/>

### Train

```shell
python main.py --train [--config] [--state_path] [--empty_logs]
```

- `--config`: config 파일 경로를 나타냅니다. 생략할 경우 `config/default.yaml`가 지정됩니다.
- `--state_path`: 모델 checkpoint 경로입니다. 명시되는 경우에 이어서 학습이 진행됩니다.
- `--empty_log`: 

### Inference

```shell
python main.py --eval --state_path [--config]
```

### Validation
지정된 validation dataset에 대해 validation을 별도로 실행할 수 있습니다. 
config에 따라 별도의 validation file을 불러오거나, train dataset을 split합니다.

## 👩‍💻 Contributor

|**[민지원](https://github.com/peacecheejecake)**                            |
| :------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/29668380?v=4)](https://github.com/peacecheejecake) |
