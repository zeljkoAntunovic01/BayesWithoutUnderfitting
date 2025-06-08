import argparse
from experiments import (
    run_alternated_projections_2d_qloss_classification_experiment,
    run_alternated_projections_2d_qproj_classification_experiment,
    run_alternated_projections_CIFAR10_experiment,
    run_naive_2d_classification_experiment,
    run_naive_sine_experiment,
    run_alternated_projections_MNIST_experiment
)

def main():
    parser = argparse.ArgumentParser(description="Run Bayesian experiment.")
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["sine", "2d_naive", "2d_altproj_qloss", "2d_altproj_qproj", "mnist", "cifar10"],
        help="Name of the experiment to run"
    )

    args = parser.parse_args()

    if args.experiment == "sine":
        run_naive_sine_experiment()
    elif args.experiment == "2d_naive":
        run_naive_2d_classification_experiment()
    elif args.experiment == "2d_altproj_qloss":
        run_alternated_projections_2d_qloss_classification_experiment()
    elif args.experiment == "2d_altproj_qproj":
        run_alternated_projections_2d_qproj_classification_experiment()
    elif args.experiment == "mnist":
        run_alternated_projections_MNIST_experiment()
    elif args.experiment == "cifar10":
        run_alternated_projections_CIFAR10_experiment()
    else:
        raise ValueError("Invalid experiment specified.")

if __name__ == "__main__":
    main()
