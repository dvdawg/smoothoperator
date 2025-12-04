
from test_suite import ExperimentRunner, run_all_datasets

print("ex1: single dataset")
print("-" * 60)

runner = ExperimentRunner(
    dataset_name='darcy',
    grid_size=64,
    modes=12,
    batch_size=20,
    n_train=800,
    n_test=200,
    n_epochs=50,
    learning_rate=0.001,
    seed=42,
    output_dir='results'
)

# results = runner.run_experiment()

print("\nex2: multiple datasets")
print("-" * 60)

# Uncomment to run:
# all_results = run_all_datasets(
#     datasets=['poisson', 'darcy'],
#     grid_size=64,
#     modes=12,
#     n_epochs=50,
#     output_dir='results'
# )

print("\nex3: quick test with fewer epochs")
print("-" * 60)

quick_runner = ExperimentRunner(
    dataset_name='poisson',
    n_epochs=10,  # Fewer epochs for quick testing
    n_train=100,  # Smaller dataset
    n_test=50,
    output_dir='results'
)

# quick_results = quick_runner.run_experiment()
