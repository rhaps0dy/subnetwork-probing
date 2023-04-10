import subprocess
import numpy as np
import shlex
import random

regularization_params = 10 ** np.linspace(-2, 3, 36)
seed = random.randint(0, 2**31 - 1)

i = 0
for lambda_reg in regularization_params:
    for zero_ablation in [0, 1]:
        for reset_target in [0, 1]:
            command = shlex.join(
                [
                    "python",
                    "/Automatic-Circuit-Discovery/subnetwork-probing/code/train_induction.py",
                    f"--lambda-reg={lambda_reg:.3f}",
                    "--wandb-entity=adria-garriga",
                    "--wandb-group=regularization",
                    "--device=cuda",
                    "--epochs=3000",
                    f"--zero-ablation={zero_ablation}",
                    f"--reset-target={reset_target}",
                    f"--seed={seed}",
                ]
            )
            print("Launching", command)

            subprocess.call(
                [
                    "ctl",
                    "job",
                    "run",
                    f"--name=agarriga-sp-{i:03d}",
                    "--shared-host-dir-slow-tolerant",
                    "--container=ghcr.io/rhaps0dy/automatic-circuit-discovery:0.3",
                    "--cpu=4",
                    "--gpu=1",
                    "--login",
                    "--wandb",
                    "--never-restart",
                    f"--command={command}",
                    "--working-dir=/Automatic-Circuit-Discovery",
                ]
            )
            i += 1
