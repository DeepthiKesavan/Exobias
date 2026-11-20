from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .evaluation import overfitting_observations


def tensorflow_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("tensorflow") is not None


def _import_tf():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for the ANN experiments. Install it with "
            "`pip install tensorflow` (use Python 3.10-3.12 for the widest support)."
        ) from exc
    return tf, keras


@dataclass
class ANNConfig:
    hidden_units: tuple = (64, 32)
    activation: str = "relu"
    dropout: float = 0.20
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 12
    seed: int = 42
    optimizer: str = "adam"

    def __post_init__(self):
        self.hidden_units = tuple(int(u) for u in self.hidden_units)
        if not self.hidden_units or any(u < 1 for u in self.hidden_units):
            raise ValueError("hidden_units must contain at least one positive layer width.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("learning_rate, batch_size, max_epochs and patience must be positive.")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_units"] = list(self.hidden_units)
        return d


def set_global_seed(seed: int) -> None:
    """Fix Python, NumPy and TensorFlow seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        _, keras = _import_tf()
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def architecture_table(n_features: int, n_classes: int, config: ANNConfig) -> pd.DataFrame:
    """Layer-by-layer description computed analytically (works without TensorFlow)."""
    rows, prev = [{"layer": "Input", "type": "Input", "units": n_features, "activation": "-",
                   "dropout": "-", "parameters": 0}], n_features
    for i, units in enumerate(config.hidden_units, start=1):
        rows.append({"layer": f"Hidden {i}", "type": "Dense", "units": units,
                     "activation": config.activation, "dropout": "-",
                     "parameters": prev * units + units})
        prev = units
        if _dropout_after(i - 1, len(config.hidden_units)) and config.dropout > 0:
            rows.append({"layer": f"Dropout {i}", "type": "Dropout", "units": prev, "activation": "-",
                         "dropout": f"{config.dropout:.2f}", "parameters": 0})
    rows.append({"layer": "Output", "type": "Dense", "units": n_classes, "activation": "softmax",
                 "dropout": "-", "parameters": prev * n_classes + n_classes})
    return pd.DataFrame(rows)


def _dropout_after(layer_index: int, n_hidden: int) -> bool:
    
    return n_hidden == 1 or layer_index < n_hidden - 1


def build_ann(n_features: int, n_classes: int, config: ANNConfig):
    _, keras = _import_tf()
    layers = [keras.Input(shape=(n_features,))]
    for i, units in enumerate(config.hidden_units):
        layers.append(keras.layers.Dense(units, activation=config.activation, name=f"hidden_{i + 1}"))
        if config.dropout > 0 and _dropout_after(i, len(config.hidden_units)):
            layers.append(keras.layers.Dropout(config.dropout, name=f"dropout_{i + 1}"))
    layers.append(keras.layers.Dense(n_classes, activation="softmax", name="output"))
    model = keras.Sequential(layers, name="ExoBias_ANN")
    optimizer = {"adam": keras.optimizers.Adam, "sgd": keras.optimizers.SGD,
                 "rmsprop": keras.optimizers.RMSprop}[config.optimizer.lower()](learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@dataclass
class ANNTrainingResult:
    model: object
    config: ANNConfig
    history: dict
    best_epoch: int
    epochs_run: int
    n_features: int
    n_classes: int
    n_train: int
    n_val: int
    weighted: bool
    train_seconds: float
    summary: dict = field(default_factory=dict)
    observations: list = field(default_factory=list)


def train_ann(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    n_classes: int, config: ANNConfig,
    sample_weight_train: Optional[np.ndarray] = None,
    sample_weight_val: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
) -> ANNTrainingResult:
    """Train one ANN. Sample weights (if given) are used only for training/validation loss."""
    _, keras = _import_tf()
    set_global_seed(config.seed)
    model = build_ann(X_train.shape[1], n_classes, config)

    class _BestWeights(keras.callbacks.Callback):
        """Explicit best-weight tracking (independent of Keras-version quirks)."""
        def __init__(self):
            super().__init__()
            self.best, self.weights = np.inf, None

        def on_epoch_end(self, epoch, logs=None):
            current = (logs or {}).get("val_loss", np.inf)
            if current < self.best:
                self.best, self.weights = current, self.model.get_weights()

        def on_train_end(self, logs=None):
            if self.weights is not None:
                self.model.set_weights(self.weights)

    class _Progress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback is not None:
                progress_callback(epoch + 1, config.max_epochs, dict(logs or {}))

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience,
                                      restore_best_weights=True),
        _BestWeights(), _Progress(),
    ]
    validation = (X_val, y_val) if sample_weight_val is None else (X_val, y_val, sample_weight_val)

    start = time.time()
    hist = model.fit(
        X_train, y_train, validation_data=validation, epochs=config.max_epochs,
        batch_size=config.batch_size, sample_weight=sample_weight_train,
        callbacks=callbacks, shuffle=True, verbose=0,
    )
    elapsed = time.time() - start

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    b = best_epoch - 1
    summary = {
        "input_features": int(X_train.shape[1]),
        "hidden_layers": len(config.hidden_units),
        "neurons_per_hidden_layer": list(config.hidden_units),
        "activation": config.activation, "output_activation": "softmax",
        "dropout": config.dropout, "optimizer": config.optimizer, "learning_rate": config.learning_rate,
        "batch_size": config.batch_size, "max_epochs": config.max_epochs,
        "epochs_run": len(history["loss"]), "best_epoch": best_epoch,
        "train_loss_at_best": history["loss"][b], "val_loss_at_best": history["val_loss"][b],
        "train_accuracy_at_best": history.get("accuracy", [np.nan] * (b + 1))[b],
        "val_accuracy_at_best": history.get("val_accuracy", [np.nan] * (b + 1))[b],
        "total_parameters": int(model.count_params()),
        "sample_weighted": sample_weight_train is not None,
    }
    return ANNTrainingResult(
        model=model, config=config, history=history, best_epoch=best_epoch,
        epochs_run=len(history["loss"]), n_features=X_train.shape[1], n_classes=n_classes,
        n_train=len(y_train), n_val=len(y_val), weighted=sample_weight_train is not None,
        train_seconds=elapsed, summary=summary,
        observations=overfitting_observations(history, best_epoch, config.patience),
    )


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X, verbose=0), dtype=float)


def save_ann(result: ANNTrainingResult, directory: Path, name: str) -> Path:
    """Save the Keras model plus its config/summary JSON. Returns the model path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{name}.keras"
    result.model.save(model_path)
    meta = {"config": result.config.to_dict(), "summary": result.summary,
            "history": result.history, "observations": result.observations}
    (directory / f"{name}_ann_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return model_path


def load_ann(model_path: Path):
    _, keras = _import_tf()
    return keras.models.load_model(model_path)
