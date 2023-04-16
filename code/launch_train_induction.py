import subprocess
import numpy as np
import shlex
import random


def main(testing=False):
    # Base NLL:  0.9099822044372559
    # Reset NLL:  11.174062728881836
    # Reset KL:  10.114195823669434
    # All the losses are of order 10, so we can use the same scale

    regularization_params = np.concatenate(
        [
            10 ** np.linspace(-2, 0, 11),
            np.linspace(1, 10, 10)[1:],
            np.linspace(10, 250, 13)[1:],
        ]
    )
    seed = random.randint(0, 2**31 - 1)

    i = 0
    for reset_subject in [0, 1]:
        for zero_ablation in [0, 1]:
            for loss_type in ["nll", "kl_div", "match_nll"]:
                for lambda_reg in [0.01] if testing else regularization_params:
                    command = [
                        "python",
                        "code/train_induction.py"
                        if testing
                        else "/Automatic-Circuit-Discovery/subnetwork-probing/code/train_induction.py",
                        f"--lambda-reg={lambda_reg:.3f}",
                        "--wandb-project=induction-sp-replicate",
                        "--wandb-entity=remix_school-of-rock",
                        "--wandb-group=reset-with-nll-2",
                        f"--device={'cpu' if testing else 'cuda'}",
                        f"--epochs={1 if testing else 10000}",
                        f"--zero-ablation={zero_ablation}",
                        f"--reset-target=0",
                        f"--reset-subject={reset_subject}",
                        f"--seed={seed}",
                        f"--loss-type={loss_type}",
                        "--num-examples=50",
                        "--seq-len=300",
                    ]
                    if testing:
                        subprocess.call(command)
                        continue

                    command_str = shlex.join(command)
                    print("Launching", command_str)
                    subprocess.call(
                        [
                            "ctl",
                            "job",
                            "run",
                            f"--name=agarriga-sp-{i:03d}",
                            "--shared-host-dir-slow-tolerant",
                            "--container=ghcr.io/rhaps0dy/automatic-circuit-discovery:0.6",
                            "--cpu=4",
                            "--gpu=1",
                            "--login",
                            "--wandb",
                            "--never-restart",
                            f"--command={command_str}",
                            "--working-dir=/Automatic-Circuit-Discovery",
                        ]
                    )
                    i += 1


if __name__ == "__main__":
    main()
