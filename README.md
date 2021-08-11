# Stage1 - Image Classification

## Overview
마스크 분류를 위한 코드입니다. 인물 이미지에서 마스크 착용 여부, 나이, 성별을 판별합니다.

## File Structure

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
- `--empty_log`: 파일을 이용한 로깅을 할 때, 로그를 지웁니다.

### Inference

```shell
python main.py --eval --state_path [--config]
```

### Validation
지정된 validation dataset에 대해 validation을 별도로 실행할 수 있습니다. 
config에 따라 별도의 validation file을 불러오거나, train dataset을 split합니다.

```shell
python main.py --valid --state_path [--config]
```
