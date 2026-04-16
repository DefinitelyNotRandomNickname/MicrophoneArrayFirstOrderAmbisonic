DATA_CONFIG="32khz"
MODEL_CONFIG="UNet"
TRAINING_CONFIG="base"

REPO_PATH="$(pwd)"

DATA_CONFIG_PATH="$REPO_PATH/configs/data/${DATA_CONFIG}.yaml"
MODEL_CONFIG_PATH="$REPO_PATH/configs/models/${MODEL_CONFIG}.yaml"
TRAINING_CONFIG_PATH="$REPO_PATH/configs/training/${TRAINING_CONFIG}.yaml"

EXP_NAME="UNet_base_8M1"
LOG_DIR="$REPO_PATH/.logs"


python projects/train.py \
  --data_cfg "$DATA_CONFIG_PATH" \
  --model_cfg "$MODEL_CONFIG_PATH" \
  --train_cfg "$TRAINING_CONFIG_PATH" \
  --exp_name "$EXP_NAME" \
  --log_dir "$LOG_DIR"
