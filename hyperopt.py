from argparse import Namespace


def wandb_hp_space(trial, hyperopt_args: Namespace):
    return {
        "method": "bayes",
        "metric": {
            "name": "objective",
            "goal": "minimize" if hyperopt_args.hparam_optimize_for_loss else "maximize"
        },
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values", "min": hyperopt_args.hparam_lr_min,
                              "max": hyperopt_args.hparam_lr_max},
            "per_device_train_batch_size": {
                "values": list(range(hyperopt_args.hparam_bs_min, hyperopt_args.hparam_bs_max + 1, 4))},
            "num_train_epochs": {"min": hyperopt_args.hparam_epoch_min, "max": hyperopt_args.hparam_epoch_max}
        },
        "run_cap": hyperopt_args.hparam_max_trials,
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 2
        }
    }
