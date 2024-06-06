# NeurIPS2024_Benchmark


This repository contains code for training various machine learning models on fMRI datasets. The datasets include HCP, ADNI, PPMI, ABIDE, OASIS.


## Installation

Install the required packages using conda:
```bash
conda create --name <env_name> --file conda_requirements.txt
conda activate <env_name>
```

## Usage

### Training a Model

To train a model, use the following command:
```bash
python main.py --gpu <gpu_id> --model <model_name> --dataset <dataset_name> --epoch <num_epochs>
```

### Example

```bash
python main.py --gpu 0 --model GCN --dataset ADNI --epoch 300
```


## License

MIT License



## Dataset: (we will release all the data upon acceptance.)

### Data Folder Structure


```bash
data
├── HCP
│   ├── TASK_LR
│   ├── TASK_RL
│   └── WM
│       ├── WM_LR
│       ├── WM_RL
│       └── label
│           ├── WM_LR
│           └── WM_RL
├── ADNI
│   ├── AAL116
│   └── label
├── OASIS
├── PPMI
└── ABIDE
