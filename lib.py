# Calculator logic.

from collections.abc import Sequence
from typing import Any, Mapping, Tuple

import enum
import termcolor
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from flax import nnx


class Model(nnx.Module):
    """Multihead attention wrapper."""

    def __init__(self, kwargs):
        """Initializes the model.
        Args:
            kwargs: Key-word arguments for initializing the MultiHeadAttention module.
        """
        self._nn = nnx.MultiHeadAttention(**kwargs)

    def __call__(self, inp):
        return self._nn(inp)

class Trainer(nnx.Optimizer):
    """Training state handler

    Attributes:
        metrics: Metric computed for the model.
    """
    def __init__(
            self,
            model: Model,
            inp: jnp.ndarray,
            metrics: nnx.metrics.Metric,
            scalar: float = 1.0
    ) -> None:
        """Initializes the training state class.

        Args:
            model: The model being trained.
            inp: The input used to initialize the model.
            metrics: The `Flax` metrics module to be computed
                during training.
            scalar: Scalar multiplier to use during computation
                of loss.
        """
        self.metrics = metrics
        self.scalar = scalar
        super().__init__(model, inp)

    def update(
            self, *, grads: jnp.ndarray, **updates: Mapping[str, Any]
    ) -> None:
        """Updates the computed metrics.
        
        Args:
            grads: Gradients computed during model training.
            updates: Updates to the model params based on `grads`.
        """
        self.metrics.update(**updates)
        super().update(grads)
    
    def train(
            self, inp: jnp.ndarray, target: jnp.ndarray
    ) -> None:
        """Train the model.
        
        Args:
            inp: The training example input.
            target: The desired model output.
        """
        values, grads = nnx.value_and_grad(loss)(
            self.model, inp, target, self.scalar
        )
        self.update(grads=grads, values=values)

def loss(
        model: Model, inp: jnp.ndarray, target: jnp.ndarray, scalar: float
) -> jnp.ndarray:
    """Computes the mean-squared error loss
    
    Args:
        model: Model to assess loss.
        inp: Input array to `model`.
        target: Target output array of the `model`.
        scalar: Scalar value by which to multiply the computed loss
            value.
    """
    return scalar * ((model(inp) - target) ** 2).mean()

class Operation(enum.Enum):
    """Mathematical operation used in generating a synthetic example."""
    ADDITION = enum.auto()
    MULTIPLICATION = enum.auto()



def make_synthetic_examples(
        num_examples: int,
        minval: int,
        maxval: int,
        operation: Operation | str,
        num_operations: int = 2,
        batch_size: int = 1,
        seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic examples.

    Args:
        num_examples: Number of synthetic examples to generate.
        minval: Lower bound of each randomly generated element value.
        maxval: Upper bound of each randomly generated element value. 
        num_operations: Number of elements involved in each individual calculation.
        batch_size: Training batch size.
        seed: Random number generator seed.

    Returns:
        A tuple of inputs and targets.
    """
    if isinstance(operation, str):
        operation = Operation(operation)
    inps = jax.random.randint(
        jax.random.key(seed),
        (batch_size, num_operations, num_examples),
        minval=minval,
        maxval=maxval
    )
    if operation == Operation.ADDITION:
        targets = jnp.sum(inps, axis=1)
    elif operation == Operation.MULTIPLICATION:
        targets = jnp.prod(inps, axis=1)
    return inps, targets

def flatten(value: jnp.ndarray | Sequence[float] | float) -> float:
    """Extract a single element value from a nested numpy array.
    
    Args:
        value: An array or list of floats or a single float value
            intended to be flattened.
    
    Returns:
        The flattened input value.

    Raises:
        ValueError: When the array is not able to be flattened
            due to containing more than one element.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, jnp.ndarray):
        value = value.tolist()
    if (size := len(value)) != 1:
        raise ValueError(
            f"Cannot flatten array of size {size}"
        )
    return flatten(value[0])

def gprint(value: Any) -> None:
    """`print`, but green.
    
    Args:
        value: Arbitrary input that can be printed.
    """
    print(termcolor.colored(value, "green"))


def train(
        train_steps: int,
        steps_to_eval: Sequence[int],
        checkpoint_directory: str,
        operation: Operation,
        batch_size: int = 1,
        learning_rate: float = 1e-3,
        num_operations = 2,
        minval: int = 0,
        maxval: int = 100000,
        plot_eval: bool = True,
        debug: bool = False,
) -> None:
    """Train and eval a model, save the checkpoint, and plot eval performance.

    Args:
        train_steps: Number of steps to train.
        steps_to_eval: List of step counts at which the model should
            be evaluated. Note that this also determines which points
            will be included within the final plot generated after
            training completes.
        checkpoint_directory: Directory to save the final model
            checkpoint. Note: the final checkpoint and not
            necessarily the best checkpoint is currently saved.
        operation: Which mathematical operation to train the model
            to perform.
        batch_size: Batch size to use for training. Note: this has
            only been tested with `batch_size` == 1.
        learning_rate: Adam learning rate scalar.
        num_operations: Number of operations to perform in a single
            training example. For instance, 4 + 4 + 4 + 4
            constitutes four `ADDIITON` operations.
        minval: Smallest integer value to use for an element
            within the synthetic training example.
        maxval: Largest integer value to use for an element
            within the synthetic training example.
        debug: Run in debug mode where logging is more verbose.
    """
    model = Model(
        dict(
            num_heads=8,
            in_features=num_operations,
            out_features=1,
            qkv_features=16,
            decode=False,
            rngs=nnx.Rngs(42)
        )
    )
    scalar = 1/((maxval - minval) ** 2)
    state = Trainer(
        model,
        optax.adam(learning_rate),
        nnx.metrics.Average(),
        scalar
    )
    num_examples = train_steps * batch_size
    # TODO: Check that batch_size is set up correctly.
    inps, targets = make_synthetic_examples(
        num_examples=num_examples,
        minval=minval,
        maxval=maxval,
        num_operations=num_operations,
        operation=operation
    )
    diffs = []
    steps = list(range(train_steps))
    for step in steps:
        inp = inps[:, :, step]
        target = targets[:, step, ...]
        state.train(inp, target)
        if step in steps_to_eval:
            out = state.model(inp)
            diffs.append(
                100 * abs(
                    flatten(target - out) / 
                    flatten(target.astype(float))
                )
            )
            if debug:
                gprint(
                    f'{step = } Input: {inp.tolist()[0]} '
                    f'Expected: {target.tolist()[0]} vs '
                    f'Actual: {out.tolist()[0][0]:0.3f}'
                )

    gprint(f"Writing checkpoint to {checkpoint_directory}")
    _, ckpt_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        f'{checkpoint_directory}/state',
        ckpt_state,
        force=True
    )
    if plot_eval:
        plt.semilogy(sorted(list(steps_to_eval)), diffs)
        plt.ylabel("Expected vs Actual Output Percent Difference")
        plt.xlabel("Training Step")
        plt.show()
