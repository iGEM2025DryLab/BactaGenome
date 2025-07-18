"""
Inference script for BactaGenome with RegulonDB data
Generates random input and visualizes model predictions vs ground truth
"""

import os
import platform
import argparse
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# MPS fallback for Apple Silicon
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bactagenome import BactaGenome, BactaGenomeConfig
from bactagenome.data import RegulonDBDataset, collate_regulondb_batch
from bactagenome.model.heads import RegulonDBLossFunction


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run BactaGenome inference with RegulonDB data")
    parser.add_argument("--config", type=str, default="configs/training/phase1_regulondb_reduced.yaml",
                        help="Path to training configuration file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase1_regulondb/checkpoint_regulondb_epoch_20.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--processed-data-dir", type=str,
                        default="./data/processed/regulondb",
                        help="Directory for processed RegulonDB data")
    parser.add_argument("--output-dir", type=str,
                        default="./inference_outputs",
                        help="Directory to save inference outputs")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of random samples to analyze")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: dict) -> BactaGenome:
    """Create BactaGenome model from configuration"""
    model_config = BactaGenomeConfig(
        dims=tuple(config['model']['dims']),
        context_length=config['model']['context_length'],
        num_organisms=config['model']['num_organisms'],
        transformer_kwargs=config['model'].get('transformer_kwargs', {})
    )
    
    model = BactaGenome(model_config)
    
    # Add RegulonDB-based bacterial heads
    phase = config.get('phase', 0)
    for organism_name in config['organisms']:
        model.add_bacterial_heads(organism_name, phase=phase)
    
    return model


def load_model(model: BactaGenome, checkpoint_path: str, device: torch.device, logger):
    """Load model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Model loaded successfully")
    return model


def create_dataset(config: dict, processed_data_dir: str, split: str = 'val'):
    """Create RegulonDB dataset for inference"""
    genome_fasta_path = "./data/raw/EcoliGene/U00096_details(1).fasta"
    
    dataset = RegulonDBDataset(
        data_dir=processed_data_dir,
        seq_len=config['data']['seq_len'],
        num_organisms=config['model']['num_organisms'],
        organism_name="E_coli_K12",
        split=split,
        enable_augmentation=False,
        process_if_missing=False,
        regulondb_raw_path=None,
        genome_fasta_path=genome_fasta_path
    )
    
    return dataset


def get_random_samples(dataset, num_samples: int, logger):
    """Get random samples from dataset"""
    logger.info(f"Selecting {num_samples} random samples from {len(dataset)} total samples")
    
    # Get random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    samples = []
    for idx in indices:
        sample = dataset[idx]
        samples.append((idx, sample))
        logger.info(f"Selected sample {idx}")
    
    return samples


def run_inference(model, samples, device, logger):
    """Run inference on samples"""
    results = []
    
    model.eval()
    with torch.no_grad():
        for idx, sample in samples:
            logger.info(f"Running inference on sample {idx}")
            
            # Create batch from single sample
            batch = collate_regulondb_batch([sample])
            
            # Move to device
            sequences = batch['dna'].to(device)
            organism_ids = batch['organism_index'].to(device)
            
            # Get predictions
            model_output = model(sequences, organism_ids)
            
            # Extract predictions from nested structure
            predictions = {}
            organism_key = "E_coli_K12"  # Based on config
            
            # Debug: Print model output structure
            logger.info(f"Model output keys: {list(model_output.keys())}")
            
            if organism_key in model_output:
                organism_predictions = model_output[organism_key]
                logger.info(f"Organism '{organism_key}' prediction keys: {list(organism_predictions.keys())}")
                
                for modality in ['gene_expression', 'gene_density', 'operon_membership']:
                    if modality in organism_predictions:
                        pred_tensor = organism_predictions[modality]
                        predictions[modality] = pred_tensor
                        logger.info(f"Found {modality} prediction with shape: {pred_tensor.shape}")
                    else:
                        logger.warning(f"Missing {modality} in organism predictions")
            else:
                logger.warning(f"Organism '{organism_key}' not found in model output")
                logger.info(f"Available organisms: {list(model_output.keys())}")
            
            # Extract targets (they have 'target_' prefix)
            targets = {}
            target_mapping = {
                'gene_expression': 'target_gene_expression',
                'gene_density': 'target_gene_density', 
                'operon_membership': 'target_operon_membership'
            }
            
            # Debug: Print batch keys
            logger.info(f"Batch keys: {list(batch.keys())}")
            
            for key, target_key in target_mapping.items():
                if target_key in batch:
                    target_tensor = batch[target_key]
                    targets[key] = target_tensor
                    logger.info(f"Found target {key} with shape: {target_tensor.shape}")
                else:
                    logger.warning(f"Missing target {target_key} in batch")
            
            results.append({
                'sample_idx': idx,
                'predictions': predictions,
                'targets': targets,
                'sequences': sequences,
                'organism_ids': organism_ids,
                'batch': batch  # Keep batch for debugging
            })
    
    return results


def visualize_results(results, output_dir: str, logger):
    """Create visualizations for predictions vs ground truth"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for i, result in enumerate(results):
        sample_idx = result['sample_idx']
        predictions = result['predictions']
        targets = result['targets']
        
        logger.info(f"Creating visualizations for sample {sample_idx}")
        
        # Create scatter plot visualization
        create_scatter_plots(sample_idx, predictions, targets, output_dir, logger)
        
        # Create sequence-based line plot visualization
        create_sequence_plots(sample_idx, predictions, targets, output_dir, logger)


def create_scatter_plots(sample_idx, predictions, targets, output_dir, logger):
    """Create scatter plots for predictions vs ground truth"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'BactaGenome Scatter Plots - Sample {sample_idx}', fontsize=16)
    
    # Track plot positions
    plot_idx = 0
    
    # Plot each modality
    for modality in ['gene_expression', 'gene_density', 'operon_membership']:
        if modality in predictions and modality in targets:
            ax = axes[plot_idx // 2, plot_idx % 2]
            
            # Get data
            pred = predictions[modality].cpu().numpy().flatten()
            target = targets[modality].cpu().numpy().flatten()
            
            # Ensure same length
            min_len = min(len(pred), len(target))
            pred = pred[:min_len]
            target = target[:min_len]
            
            # Create scatter plot
            ax.scatter(target, pred, alpha=0.6, s=20)
            
            # Add perfect prediction line
            min_val = min(target.min(), pred.min())
            max_val = max(target.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            # Calculate R²
            if len(target) > 1:
                correlation = np.corrcoef(target, pred)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            else:
                r_squared = 0
            
            ax.set_xlabel(f'Ground Truth {modality}')
            ax.set_ylabel(f'Predicted {modality}')
            ax.set_title(f'{modality.replace("_", " ").title()}\nR² = {r_squared:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Add sequence length info in remaining subplot
    if plot_idx < 4:
        ax = axes[plot_idx // 2, plot_idx % 2]
        seq_len = 98304  # From config
        ax.text(0.5, 0.5, f'Sequence Length: {seq_len:,} bp\nOrganism: E. coli K-12', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Sample Information')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'scatter_sample_{sample_idx}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plot: {plot_path}")


def create_sequence_plots(sample_idx, predictions, targets, output_dir, logger):
    """Create sequence-based line plots showing predictions and targets along genomic positions"""
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    fig.suptitle(f'BactaGenome Sequence Plots - Sample {sample_idx}', fontsize=16)
    
    plot_idx = 0
    modality_info = {
        'gene_expression': {'color': 'blue', 'resolution': 1, 'ylabel': 'Expression Level'},
        'gene_density': {'color': 'green', 'resolution': 128, 'ylabel': 'Gene Count'},
        'operon_membership': {'color': 'red', 'resolution': 1, 'ylabel': 'Operon Probability'}
    }
    
    for modality in ['gene_expression', 'gene_density', 'operon_membership']:
        if modality in predictions and modality in targets:
            ax = axes[plot_idx]
            
            # Get data
            pred = predictions[modality].cpu().numpy().flatten()
            target = targets[modality].cpu().numpy().flatten()
            
            # Ensure same length
            min_len = min(len(pred), len(target))
            pred = pred[:min_len]
            target = target[:min_len]
            
            # Create sequence positions
            resolution = modality_info[modality]['resolution']
            positions = np.arange(len(pred)) * resolution
            
            # Downsample for visualization if too many points
            if len(pred) > 5000:
                step = len(pred) // 5000
                positions = positions[::step]
                pred = pred[::step]
                target = target[::step]
            
            # Plot lines
            color = modality_info[modality]['color']
            ax.plot(positions, target, color=color, alpha=0.7, linewidth=1, label='Ground Truth')
            ax.plot(positions, pred, color=color, alpha=0.7, linewidth=1, linestyle='--', label='Prediction')
            
            # Calculate statistics
            if len(target) > 1:
                correlation = np.corrcoef(target, pred)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
                mse = np.mean((pred - target) ** 2)
            else:
                r_squared = 0
                mse = 0
            
            ax.set_xlabel('Genomic Position (bp)')
            ax.set_ylabel(modality_info[modality]['ylabel'])
            ax.set_title(f'{modality.replace("_", " ").title()} - R² = {r_squared:.3f}, MSE = {mse:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add some spacing between values for clarity
            y_range = target.max() - target.min()
            if y_range > 0:
                ax.set_ylim(target.min() - 0.1 * y_range, target.max() + 0.1 * y_range)
            
            plot_idx += 1
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'sequence_sample_{sample_idx}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved sequence plot: {plot_path}")


def print_summary_stats(results, logger):
    """Print summary statistics for all results"""
    logger.info("=== INFERENCE SUMMARY ===")
    
    # Collect all predictions and targets
    all_predictions = {}
    all_targets = {}
    
    for result in results:
        predictions = result['predictions']
        targets = result['targets']
        
        for modality in ['gene_expression', 'gene_density', 'operon_membership']:
            if modality in predictions and modality in targets:
                if modality not in all_predictions:
                    all_predictions[modality] = []
                    all_targets[modality] = []
                
                pred = predictions[modality].cpu().numpy().flatten()
                target = targets[modality].cpu().numpy().flatten()
                
                # Ensure same length
                min_len = min(len(pred), len(target))
                all_predictions[modality].extend(pred[:min_len])
                all_targets[modality].extend(target[:min_len])
    
    # Calculate summary statistics
    for modality in all_predictions:
        pred = np.array(all_predictions[modality])
        target = np.array(all_targets[modality])
        
        if len(target) > 1:
            correlation = np.corrcoef(target, pred)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            mse = np.mean((pred - target) ** 2)
            mae = np.mean(np.abs(pred - target))
            
            logger.info(f"{modality.replace('_', ' ').title()}:")
            logger.info(f"  R² = {r_squared:.4f}")
            logger.info(f"  MSE = {mse:.4f}")
            logger.info(f"  MAE = {mae:.4f}")
            logger.info(f"  Samples = {len(target)}")
        else:
            logger.info(f"{modality.replace('_', ' ').title()}: Insufficient data")
    
    logger.info("========================")


def main():
    """Main inference function"""
    logger = setup_logging()
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model created with {model.total_parameters:,} parameters")
    
    # Load checkpoint
    model = load_model(model, args.checkpoint, device, logger)
    model = model.to(device)
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = create_dataset(config, args.processed_data_dir, split='val')
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Get random samples
    samples = get_random_samples(dataset, args.num_samples, logger)
    
    # Run inference
    logger.info("Running inference...")
    results = run_inference(model, samples, device, logger)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualize_results(results, args.output_dir, logger)
    
    # Print summary statistics
    print_summary_stats(results, logger)
    
    logger.info(f"Inference completed! Results saved in {args.output_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()