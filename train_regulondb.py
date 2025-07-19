"""
Training script for BactaGenome with real RegulonDB data
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import platform
import argparse
import yaml
import random
import numpy as np
import logging
import math
from pathlib import Path
from datetime import datetime

# MPS fallback for Apple Silicon
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    print("✅ MPS fallback enabled for Apple Silicon.")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # <-- 1. 导入 SummaryWriter
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from bactagenome import BactaGenome, BactaGenomeConfig
from bactagenome.data import RegulonDBDataset, RegulonDBDataLoader, collate_regulondb_batch
from bactagenome.training import BactaGenomeTrainer
from bactagenome.model.heads import RegulonDBLossFunction


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_regulondb.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train BactaGenome with RegulonDB data")
    parser.add_argument("--config", type=str, default="configs/training/phase1_regulondb.yaml",
                        help="Path to training configuration file")
    parser.add_argument("--regulondb-path", type=str, 
                        default="./data/raw/RegulonDB",
                        help="Path to raw RegulonDB BSON files")
    parser.add_argument("--processed-data-dir", type=str,
                        default="./data/processed/regulondb",
                        help="Directory for processed RegulonDB data")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--process-data-only", action="store_true",
                        help="Only process data, don't train")
    parser.add_argument("--test-mode", action="store_true",
                        help="Use limited data for testing")
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
    """Create BactaGenome model from configuration with RegulonDB-based heads"""
    model_config = BactaGenomeConfig(
        dims=tuple(config['model']['dims']),
        context_length=config['model']['context_length'],
        num_organisms=config['model']['num_organisms'],
        transformer_kwargs=config['model'].get('transformer_kwargs', {})
    )
    
    model = BactaGenome(model_config)
    
    # Add RegulonDB-based bacterial heads directly (phase 0)
    phase = config.get('phase', 0)  # Default to phase 0 for RegulonDB training
    for organism_name in config['organisms']:
        model.add_bacterial_heads(organism_name, phase=phase)
    
    return model


def create_regulondb_datasets(config: dict, regulondb_path: str, processed_data_dir: str, max_docs_for_testing: int = None):
    """Create RegulonDB training and validation datasets"""
    
    # Path to E. coli genome FASTA file
    genome_fasta_path = "./data/raw/EcoliGene/U00096_details(1).fasta"
    
    # Create datasets using chromosome-based splits
    train_dataset = RegulonDBDataset(
        data_dir=processed_data_dir,
        seq_len=config['data']['seq_len'],
        num_organisms=config['model']['num_organisms'],
        organism_name="E_coli_K12",
        split='train',
        enable_augmentation=True,
        process_if_missing=True,
        regulondb_raw_path=regulondb_path,
        genome_fasta_path=genome_fasta_path,
        max_docs_per_file=max_docs_for_testing
    )
    
    val_dataset = RegulonDBDataset(
        data_dir=processed_data_dir,
        seq_len=config['data']['seq_len'],
        num_organisms=config['model']['num_organisms'],
        organism_name="E_coli_K12", 
        split='val',
        enable_augmentation=False,
        process_if_missing=False,  # Already processed by train_dataset
        regulondb_raw_path=None,
        genome_fasta_path=genome_fasta_path,
        max_docs_per_file=max_docs_for_testing
    )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: dict):
    """Create data loaders with custom collate function"""
    
    # Disable pin_memory on MPS (Apple Silicon) to avoid warnings
    pin_memory = not torch.backends.mps.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=pin_memory,
        collate_fn=collate_regulondb_batch,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=pin_memory,
        collate_fn=collate_regulondb_batch,
        drop_last=False
    )
    
    return train_loader, val_loader


def main():
    """Main training function"""
    logger = setup_logging()
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Create output directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # <-- 2. 初始化 SummaryWriter
    writer = None
    if config['training'].get('use_tensorboard', False):
        run_name = f"{Path(args.config).stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard_log_dir = Path(config['training']['log_dir']) / "tensorboard" / run_name
        writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    
    # Determine data limits for testing
    max_docs = 1000 if args.test_mode else None
    
    # Create datasets
    logger.info("Creating RegulonDB datasets...")
    train_dataset, val_dataset = create_regulondb_datasets(
        config, args.regulondb_path, args.processed_data_dir, max_docs
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Exit if only processing data
    if args.process_data_only:
        logger.info("Data processing complete. Exiting.")
        return
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model created with {model.total_parameters:,} parameters")
    
    # Print realistic heads info
    for organism in config['organisms']:
        if hasattr(model, 'heads') and organism in model.heads:
            head_manager = model.heads[organism]
            if hasattr(head_manager, 'get_target_info'):
                target_info = head_manager.get_target_info()
                logger.info(f"RegulonDB-based heads for {organism}:")
                for target_name, info in target_info.items():
                    logger.info(f"  • {target_name}: {info['type']} ({info['resolution']}, {info['loss']} loss)")
            else:
                logger.info(f"Organism {organism}: Standard heads (no target info available)")
    
    # Create optimizer - AlphaGenome style parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create loss function
    loss_function = RegulonDBLossFunction(
        loss_weights=config['training'].get('loss_weights', {})
    )
    
    # Setup accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Create learning rate scheduler with warmup
    def create_lr_scheduler(optimizer, config, dataloader):
        """Create AlphaGenome-style learning rate scheduler with warmup and min_lr."""
        warmup_steps = config['training'].get('warmup_steps', 1000)
        num_epochs = config['training']['epochs']
        
        # 【关键修改】: 动态计算 total_steps
        # accelerator.num_processes 会给出分布式训练时的GPU数量
        num_update_steps_per_epoch = math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps)
        total_steps = num_update_steps_per_epoch * num_epochs
        
        # 【新功能】: 添加一个最小学习率，防止学习完全停止
        min_lr_ratio = 0.05 # 学习率最终衰减到峰值的 5%
        peak_lr = optimizer.param_groups[0]['lr'] # 从optimizer获取峰值学习率
        
        logger.info(f"LR Scheduler: Warmup steps={warmup_steps}, Total steps={total_steps}, Peak LR={peak_lr:.2e}, Final LR ~={peak_lr*min_lr_ratio:.2e}")
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # 线性 warmup
                return float(current_step) / float(max(1, warmup_steps))
            
            # 【关键修改】: 改进的 Cosine Decay
            # 确保我们不会衰减到0，而是一个最小学习率
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            
            # Cosine 衰减从 1 到 min_lr_ratio
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed_ratio = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio
            
            return max(0.0, decayed_ratio)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    scheduler = create_lr_scheduler(optimizer, config, train_loader) if config['training'].get('scheduler') == 'cosine_with_warmup' else None
    if scheduler:
        scheduler = accelerator.prepare(scheduler)
        
    device = accelerator.device
    
    # Create trainer
    trainer = BactaGenomeTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        accelerator=accelerator,
        log_interval=config['training'].get('log_interval', 10),
        max_grad_norm=config['training'].get('max_grad_norm'),
        scheduler=scheduler,
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint['epoch']
    
    # Training loop
    logger.info("Starting RegulonDB training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Training
        train_metrics = trainer.train_epoch(train_loader, epoch + 1)
        logger.info(f"=== EPOCH {epoch + 1} TRAINING LOSSES ===")
        logger.info(f"🔢 Total Loss: {train_metrics['total_loss']:.6f}")
        
        modality_losses = {k: v for k, v in train_metrics.items() if k not in ['total_loss', 'samples_processed']}
        if modality_losses:
            logger.info("📊 Individual Modality Losses:")
            for modality, loss in modality_losses.items():
                logger.info(f"   • {modality}: {loss:.6f}")
        else:
            logger.info("⚠️  No individual modality losses found!")
        
        logger.info(f"👥 Samples processed: {train_metrics.get('samples_processed', 'unknown')}")
        logger.info("=" * 40)
        
        # <-- 3. 将训练指标写入 TensorBoard
        if writer and accelerator.is_main_process:
            writer.add_scalar('Loss/train_total', train_metrics['total_loss'], epoch + 1)
            for modality, loss_val in modality_losses.items():
                writer.add_scalar(f'Loss/train_{modality}', loss_val, epoch + 1)
            
            current_lr = trainer.optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, epoch + 1)

        # Validation
        if (epoch + 1) % config['training'].get('val_interval', 5) == 0:
            val_metrics = trainer.validate_epoch(val_loader)
            logger.info(f"=== EPOCH {epoch + 1} VALIDATION LOSSES ===")
            logger.info(f"🔢 Total Validation Loss: {val_metrics['total_loss']:.6f}")
            
            val_modality_losses = {k: v for k, v in val_metrics.items() if k not in ['total_loss', 'samples_processed']}
            if val_modality_losses:
                logger.info("📊 Individual Validation Modality Losses:")
                for modality, loss in val_modality_losses.items():
                    logger.info(f"   • {modality}: {loss:.6f}")
            else:
                logger.info("⚠️  No individual validation modality losses found!")
            
            logger.info(f"👥 Validation samples processed: {val_metrics.get('samples_processed', 'unknown')}")
            logger.info("=" * 40)
            
            # <-- 4. 将验证指标写入 TensorBoard
            if writer and accelerator.is_main_process:
                writer.add_scalar('Loss/validation_total', val_metrics['total_loss'], epoch + 1)
                for modality, loss_val in val_modality_losses.items():
                    writer.add_scalar(f'Loss/validation_{modality}', loss_val, epoch + 1)

            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                best_model_path = os.path.join(config['training']['checkpoint_dir'], 'best_model_regulondb.pt')
                trainer.save_checkpoint(best_model_path, epoch + 1, val_loss=best_val_loss)
                logger.info(f"🏆 New best model saved: {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                config['training']['checkpoint_dir'],
                f'checkpoint_regulondb_epoch_{epoch + 1}.pt'
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1)
        
        # Wait for all processes
        accelerator.wait_for_everyone()
    
    # Save final model
    final_model_path = os.path.join(config['training']['checkpoint_dir'], 'final_model_regulondb.pt')
    trainer.save_checkpoint(final_model_path, config['training']['epochs'])
    logger.info(f"Final model saved: {final_model_path}")
    
    # <-- 5. 训练结束后关闭 writer
    if writer and accelerator.is_main_process:
        writer.close()
        logger.info("TensorBoard writer closed.")
    
    logger.info("RegulonDB training completed!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()