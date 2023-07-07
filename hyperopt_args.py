from dataclasses import field, dataclass
from typing import Optional


@dataclass
class HyperOptArguments:
    do_hparams_search: bool = field(
        default=False,
        metadata={
            "help": "Whether to do a hyperparameter search on learning rate, batch size, and number of epochs."},
    )
    hparam_lr_min: float = field(
        default=5e-4,
        metadata={"help": "Minimum learning rate in hyperparameter search. Only used if 'do_hparams_search'."},
    )
    hparam_lr_max: float = field(
        default=5e-3,
        metadata={"help": "Maximum learning rate in hyperparameter search. Only used if 'do_hparams_search'."},
    )
    hparam_gr_accum_min: int = field(
        default=1,
        metadata={"help": "Minimum batch size in hyperparameter search, must be a multiple of 4. Only used if"
                          " 'do_hparams_search'."},
    )
    hparam_gr_accum_max: int = field(
        default=16,
        metadata={"help": "Maximum batch size in hyperparameter search, must be a multiple of 4. Only used if"
                          " 'do_hparams_search'."},
    )
    hparam_epoch_min: int = field(
        default=1,
        metadata={"help": "Minimum number of epochs in hyperparameter search. Only used if 'do_hparams_search'."},
    )
    hparam_epoch_max: int = field(
        default=10,
        metadata={"help": "Maximum number of epochs in hyperparameter search. Only used if 'do_hparams_search'."},
    )
    hparam_max_trials: int = field(
        default=16,
        metadata={"help": "Maximum number of hparam search trials to run. Only used if 'do_hparams_search'."},
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "Stop training when the evaluation metric worsens (instead of improves) for"
            " early_stopping_patience evaluation calls."
        },
    )
    early_stopping_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Denote how much the evaluation metric must improve to satisfy early stopping conditions."},
    )

    def __post_init__(self):
        if self.do_hparams_search:
            raise ValueError("Hyperparameter search is currently not supported because trl does not support it.")