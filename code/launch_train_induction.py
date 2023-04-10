import subprocess
import numpy as np
import shlex
import random

regularization_params = 10 ** np.linspace(-2, 2, 29)
seed = random.randint(0, 2**31 - 1)

i = 0
for reset_subject in [0, 1]:
    for zero_ablation in [0, 1]:
        for lambda_reg in regularization_params:
            command = shlex.join(
                [
                    "python",
                    "/Automatic-Circuit-Discovery/subnetwork-probing/code/train_induction.py",
                    f"--lambda-reg={lambda_reg:.3f}",
                    "--wandb-entity=adria-garriga",
                    "--wandb-group=reset-subject",
                    "--device=cuda",
                    "--epochs=12000",
                    f"--zero-ablation={zero_ablation}",
                    f"--reset-target=0",
                    f"--reset-subject={reset_subject}",
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
                    "--container=ghcr.io/rhaps0dy/automatic-circuit-discovery:0.4",
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
