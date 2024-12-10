# Rube goldberg award-winning calculator.
from absl import app
from absl import flags

import lib

_DEBUG = flags.DEFINE_boolean(
    "debug",
    False,
    "Run in `debug` mode."
)
_TRAIN_STEPS = flags.DEFINE_integer(
    "train_steps",
    1000,
    "Number of training steps."
)
_STEPS_TO_EVALUATE = flags.DEFINE_multi_integer(
    "steps_to_evaluate",
    [1, 10,],
    "Training steps during which to evaluate the model being "
    "trained. Note: the steps to evaluate is a union between "
    "the 'evaluate_every_n_steps' flag and the "
    "'steps_to_evaluate' flag."
)
_EVALUATE_EVERY_N_STEPS = flags.DEFINE_integer(
    "evaluate_every_n_steps",
    50,
    "How frequently to output model evaluations. Note: the steps "
    "to evaluate is a union between the 'evaluate_every_n_steps' "
    "flag and the 'steps_to_evaluate' flag."
)
_CHECKPOINT_DIRECTORY = flags.DEFINE_string(
    "checkpoint_directory",
    None,
    "Directory in which to save the final checkpoint."
)
_OPERATION = flags.DEFINE_enum_class(
    "operation",
    lib.Operation.ADDITION,
    lib.Operation,
    "The mathematical operation to teach the model to perform.",
)

def main(_):
    train_steps = _TRAIN_STEPS.value
    steps_to_evaluate = (
        set(_STEPS_TO_EVALUATE.value) | 
        set(range(1, train_steps, _EVALUATE_EVERY_N_STEPS.value))
    )
    lib.train(
        train_steps=train_steps,
        steps_to_eval=steps_to_evaluate,
        checkpoint_directory = _CHECKPOINT_DIRECTORY.value,
        debug=_DEBUG.value,
        operation=_OPERATION.value
    )

if __name__ == "__main__":
    app.run(main)
