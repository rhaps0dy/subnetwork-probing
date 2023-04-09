import subprocess

regularization_params = [
    1e-2,
    1e-1,
    1e1,
    20,
    40,
    50,
    55,
    60,
    65,
    70,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    250,
    300,
    310,
    320,
    330,
    350,
    360,
    370,
    380,
    400,
    500,
    600,
    700,
    800,
    900,
    1e3,
]

for param in regularization_params:
    print("Launching training for regularization parameter {}".format(param))
    subprocess.call(
        [
            "python3",
            "train_induction.py",
            f"--regularization-param={param}",
        ]
    )
