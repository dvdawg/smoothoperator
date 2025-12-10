"""
Test Suite for Condition-Aware FNO

Runs experiments on multiple datasets and compares:
- Standard FNO
- Condition-Aware FNO with adaptive spectral truncation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Import from main module
from condition_aware_fno import (
    FNO2d, ConditionAwareFNO2d,
    compute_adaptive_mask, ridge_regression_init,
    train_epoch, evaluate
)
from datasets import get_dataset, DATASET_REGISTRY


class ExperimentRunner:
    """Runs experiments on different datasets"""
    
    def __init__(
        self,
        dataset_name: str,
        grid_size: int = 64,
        modes: int = 12,
        batch_size: int = 20,
        n_train: int = 800,
        n_test: int = 200,
        n_epochs: int = 50,
        learning_rate: float = 0.001,
        seed: int = 42,
        device: str = None,
        output_dir: str = "results"
    ):
        self.dataset_name = dataset_name
        self.grid_size = grid_size
        self.modes = modes
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_test = n_test
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic CUDA operations (may be slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Results storage
        self.results = {
            'dataset': dataset_name,
            'config': {
                'grid_size': grid_size,
                'modes': modes,
                'batch_size': batch_size,
                'n_train': n_train,
                'n_test': n_test,
                'n_epochs': n_epochs,
                'learning_rate': learning_rate,
                'seed': seed
            },
            'standard_fno': {},
            'condition_aware_fno': {}
        }
    
    def run_experiment(self):
        """Run the full experiment"""
        print("="*80)
        print(f"EXPERIMENT: {self.dataset_name.upper()} Dataset")
        print("="*80)
        print(f"Using device: {self.device}")
        print()
        
        # Create datasets
        print("Generating synthetic data...")
        train_dataset = get_dataset(
            self.dataset_name,
            n_samples=self.n_train,
            grid_size=self.grid_size,
            seed=self.seed
        )
        test_dataset = get_dataset(
            self.dataset_name,
            n_samples=self.n_test,
            grid_size=self.grid_size,
            seed=self.seed + 1000  # Different seed for test set
        )
        
        # Create DataLoaders with deterministic shuffling
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=generator
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Compute normalization statistics
        # Collect data in deterministic order (no shuffling for stats)
        print("Computing normalization statistics and adaptive spectral mask...")
        a_batch_list = []
        u_batch_list = []
        
        # Collect all data deterministically (without shuffling for statistics)
        stats_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        for a, u in stats_loader:
            a_batch_list.append(a)
            u_batch_list.append(u)
        
        a_all = torch.cat(a_batch_list, dim=0).to(self.device)
        u_all = torch.cat(u_batch_list, dim=0).to(self.device)
        
        u_mean = u_all.mean().item()
        u_std = u_all.std().item()
        print(f"Output statistics: mean={u_mean:.4f}, std={u_std:.4f}")
        print(f"Normalizing outputs to unit scale for training stability...")
        
        u_all_normalized = (u_all - u_mean) / (u_std + 1e-8)
        
        # Compute adaptive mask on fixed subset (first n_samples_for_mask samples)
        # This ensures the same samples are used for mask computation each time
        n_samples_for_mask = min(200, len(a_all))
        a_mask = a_all[:n_samples_for_mask]
        u_mask = u_all_normalized[:n_samples_for_mask]
        
        low_mask, high_mask = compute_adaptive_mask(
            a_mask, u_mask, modes1=self.modes, modes2=self.modes
        )
        n_active_low = low_mask.sum().item()
        n_active_high = high_mask.sum().item()
        print(f"Adaptive masks: Low={n_active_low}/{self.modes*self.modes}, "
              f"High={n_active_high}/{self.modes*self.modes} modes selected")
        
        # Note: We compute the mask using physical variables, but we do NOT use 
        # the ridge regression weights for initialization here because the FNO 
        # operates on lifted channels (64), not physical channels (1).
        
        # Train Standard FNO
        print("\n" + "="*60)
        print("Training Standard FNO (Model A)")
        print("="*60)
        standard_results = self._train_model(
            model_type='standard',
            train_loader=train_loader,
            test_loader=test_loader,
            u_mean=u_mean,
            u_std=u_std
        )
        self.results['standard_fno'] = standard_results
        
        # Train Condition-Aware FNO
        print("\n" + "="*60)
        print("Training Condition-Aware FNO (Model B)")
        print("="*60)
        adaptive_results = self._train_model(
            model_type='adaptive',
            train_loader=train_loader,
            test_loader=test_loader,
            u_mean=u_mean,
            u_std=u_std,
            low_mask=low_mask,
            high_mask=high_mask
        )
        self.results['condition_aware_fno'] = adaptive_results
        
        # Print summary
        self._print_summary()
        
        # Save results
        self._save_results()
        
        # Plot results
        self._plot_results()
        
        return self.results
    
    def _train_model(
        self,
        model_type: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        u_mean: float,
        u_std: float,
        low_mask: torch.Tensor = None,
        high_mask: torch.Tensor = None,
        init_weights1: torch.Tensor = None,
        init_weights2: torch.Tensor = None
    ) -> Dict:
        """Train a model and return results"""
        # Create model
        if model_type == 'standard':
            model = FNO2d(modes1=self.modes, modes2=self.modes, width=64).to(self.device)
        else:
            # FIX: Removed explicit weight passing (init_weights1/2).
            # The model will use random initialization for the 64-channel lifted space,
            # but constrained by the passed masks.
            model = ConditionAwareFNO2d(
                low_mask=low_mask, 
                high_mask=high_mask,
                modes1=self.modes, 
                modes2=self.modes, 
                width=64
            ).to(self.device)
            print("Model initialized with Condition-Aware Spectral Mask.")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        test_losses = []
        
        start_time = time.time()
        for epoch in range(self.n_epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion,
                self.device, u_mean, u_std
            )
            test_loss = evaluate(
                model, test_loader, criterion,
                self.device, u_mean, u_std
            )
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        training_time = time.time() - start_time
        final_test_loss = test_losses[-1]
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final Test Loss: {final_test_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_test_loss': final_test_loss,
            'training_time': training_time
        }
    
    def _print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        std_loss = self.results['standard_fno']['final_test_loss']
        adapt_loss = self.results['condition_aware_fno']['final_test_loss']
        improvement = ((std_loss - adapt_loss) / std_loss * 100) if std_loss > 0 else 0
        
        print(f"Dataset: {self.dataset_name}")
        print(f"\nStandard FNO:")
        print(f"  Final Test Loss: {std_loss:.6f}")
        print(f"  Training Time: {self.results['standard_fno']['training_time']:.2f}s")
        
        print(f"\nCondition-Aware FNO:")
        print(f"  Final Test Loss: {adapt_loss:.6f}")
        print(f"  Training Time: {self.results['condition_aware_fno']['training_time']:.2f}s")
        print(f"  Improvement: {improvement:+.2f}%")
        print()
    
    def _save_results(self):
        """Save results to JSON file"""
        # Convert tensors to lists for JSON serialization
        results_copy = {}
        for key, value in self.results.items():
            if key in ['standard_fno', 'condition_aware_fno']:
                results_copy[key] = {
                    'train_losses': [float(x) for x in value['train_losses']],
                    'test_losses': [float(x) for x in value['test_losses']],
                    'final_test_loss': float(value['final_test_loss']),
                    'training_time': float(value['training_time'])
                }
            else:
                results_copy[key] = value
        
        filename = self.output_dir / f"{self.dataset_name}_results.json"
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        print(f"Results saved to {filename}")
    
    def _plot_results(self):
        """Plot and save convergence curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Full convergence
        ax = axes[0]
        std_train = self.results['standard_fno']['train_losses']
        std_test = self.results['standard_fno']['test_losses']
        adapt_train = self.results['condition_aware_fno']['train_losses']
        adapt_test = self.results['condition_aware_fno']['test_losses']
        
        ax.plot(std_train, label='Standard FNO (Train)', linestyle='--', alpha=0.7)
        ax.plot(std_test, label='Standard FNO (Test)', linewidth=2)
        ax.plot(adapt_train, label='Condition-Aware FNO (Train)', linestyle='--', alpha=0.7)
        ax.plot(adapt_test, label='Condition-Aware FNO (Test)', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'{self.dataset_name.upper()} Dataset - Training Convergence')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Test loss comparison
        ax = axes[1]
        ax.plot(std_test, label='Standard FNO', linewidth=2)
        ax.plot(adapt_test, label='Condition-Aware FNO', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test MSE Loss')
        ax.set_title(f'{self.dataset_name.upper()} Dataset - Test Loss Comparison')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.output_dir / f"{self.dataset_name}_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        plt.close()


def run_all_datasets(
    datasets: List[str] = None,
    output_dir: str = "results",
    **experiment_kwargs
):
    """Run experiments on multiple datasets"""
    if datasets is None:
        datasets = list(DATASET_REGISTRY.keys())
    
    all_results = {}
    
    for dataset_name in datasets:
        print("\n" + "="*80)
        print(f"Running experiment on {dataset_name.upper()} dataset")
        print("="*80)
        
        runner = ExperimentRunner(
            dataset_name=dataset_name,
            output_dir=output_dir,
            **experiment_kwargs
        )
        results = runner.run_experiment()
        all_results[dataset_name] = results
        
        print(f"\nCompleted {dataset_name} experiment\n")
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    for dataset_name, results in all_results.items():
        std_loss = results['standard_fno']['final_test_loss']
        adapt_loss = results['condition_aware_fno']['final_test_loss']
        improvement = ((std_loss - adapt_loss) / std_loss * 100) if std_loss > 0 else 0
        print(f"{dataset_name:12s} | Standard: {std_loss:.6f} | "
              f"Adaptive: {adapt_loss:.6f} | Improvement: {improvement:+.2f}%")
    
    return all_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test Condition-Aware FNO on different datasets')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=list(DATASET_REGISTRY.keys()) + ['all'],
                       help='Dataset to test (default: all)')
    parser.add_argument('--grid_size', type=int, default=64, help='Grid size')
    parser.add_argument('--modes', type=int, default=12, help='Number of modes')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--n_train', type=int, default=800, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=200, help='Number of test samples')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    experiment_kwargs = {
        'grid_size': args.grid_size,
        'modes': args.modes,
        'batch_size': args.batch_size,
        'n_train': args.n_train,
        'n_test': args.n_test,
        'n_epochs': args.n_epochs,
        'learning_rate': args.lr,
        'seed': args.seed
    }
    
    if args.dataset is None or args.dataset == 'all':
        # Run all datasets
        run_all_datasets(output_dir=args.output_dir, **experiment_kwargs)
    else:
        # Run single dataset
        runner = ExperimentRunner(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            **experiment_kwargs
        )
        runner.run_experiment()


if __name__ == "__main__":
    main()