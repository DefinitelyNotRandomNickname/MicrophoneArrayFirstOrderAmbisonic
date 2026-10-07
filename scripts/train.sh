DATA_CONFIG="tetra/32khz"
MODEL_CONFIG="tfgridnet/TFGridNet"
TRAINING_CONFIG="base_mapping"

REPO_PATH="$(pwd)"

DATA_CONFIG_PATH="$REPO_PATH/configs/data/${DATA_CONFIG}.yaml"
MODEL_CONFIG_PATH="$REPO_PATH/configs/models/${MODEL_CONFIG}.yaml"
TRAINING_CONFIG_PATH="$REPO_PATH/configs/training/${TRAINING_CONFIG}.yaml"

EXP_NAME="${MODEL_CONFIG#*/}_${DATA_CONFIG%%/*}"
LOG_DIR="$REPO_PATH/.logs/${MODEL_CONFIG%%/*}"


python projects/train.py \
  --data_cfg "$DATA_CONFIG_PATH" \
  --model_cfg "$MODEL_CONFIG_PATH" \
  --train_cfg "$TRAINING_CONFIG_PATH" \
  --exp_name "$EXP_NAME" \
  --log_dir "$LOG_DIR"
