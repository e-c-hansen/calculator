
_CHECKPOINT_DIRECTORY="${PWD:?}/calculator/checkpoints"
mkdir "${_CHECKPOINT_DIRECTORY}"

ARGS=(
    --checkpoint_directory="${_CHECKPOINT_DIRECTORY}"
)
python3 "${PWD:?}/calculator/main.py" "${ARGS[@]}"s