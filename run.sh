
_CHECKPOINT_DIRECTORY=/Users/erichansen/Code/learning/calculator/checkpoints
mkdir "${_CHECKPOINT_DIRECTORY}"

ARGS=(
    --checkpoint_directory="${_CHECKPOINT_DIRECTORY}"
)
python3 main.py "${ARGS[@]}"