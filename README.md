# Train

## 1. Prepare Datasets
Follow the instructions [here](legacy/training/README.md)

## 2. Model Configuration
The training process will load the initialized NHA model with model path set in [cmd_qwen_moe.sh](legacy/training/scripts/cmd_qwen_moe.sh). If you want to change the model architechture, edit the `config.json` in the model path such as `/cpfs01/shared/MoE/Ali_MOE/models/Qwen3-30B-A3B-HMA/config.json` and then train the model.

## 3. Training Scripts
```bash
cd legacy/training/scripts
bash cmd_qwen_moe.sh
```

# Eval

You can change model path, fsdp configuration and task list in [eval.sh](eval.sh), then run the scripts
```bash
sudo bash eval.sh
```