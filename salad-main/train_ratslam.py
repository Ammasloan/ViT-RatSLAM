#!/usr/bin/env python3
"""
Fine-tune SALAD model on RatSLAM bag-extracted data.

Usage:
    python train_ratslam.py \
        --train-csv data/irat_manual/places.csv \
        --pretrained weights/dino_salad.ckpt \
        --batch-size 16 \
        --epochs 4

The script will:
1. Load pretrained SALAD weights
2. Fine-tune on custom bag data
3. Save checkpoints to logs/ratslam_finetune/
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from vpr_model import VPRModel
from dataloaders.BagPlacesDataModule import BagPlacesDataModule


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune SALAD on RatSLAM bag data')
    
    # Data
    parser.add_argument('--train-csv', required=True, help='Path to training places.csv')
    parser.add_argument('--val-csv', default=None, help='Optional validation places.csv')
    
    # Model
    parser.add_argument('--pretrained', default='weights/dino_salad.ckpt',
                        help='Path to pretrained SALAD checkpoint')
    parser.add_argument('--image-size', type=int, default=322,
                        help='Input image size (default: 322)')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (places per batch, default: 16)')
    parser.add_argument('--img-per-place', type=int, default=4,
                        help='Images per place (default: 4)')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs (default: 4)')
    parser.add_argument('--lr', type=float, default=6e-5,
                        help='Learning rate (default: 6e-5)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    
    # Output
    parser.add_argument('--output-dir', default='logs/ratslam_finetune',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment-name', default='salad_ratslam',
                        help='Experiment name for logging')
    
    # Hardware
    parser.add_argument('--precision', default='16-mixed', choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision (default: 16-mixed)')
    parser.add_argument('--accelerator', default='auto', choices=['auto', 'gpu', 'cpu'],
                        help='Accelerator (default: auto)')
    
    return parser.parse_args()


def load_pretrained_model(ckpt_path: str, lr: float) -> VPRModel:
    """
    Load pretrained SALAD model.
    
    We create a new VPRModel with the same architecture and load weights.
    """
    print(f"Loading pretrained weights from: {ckpt_path}")
    
    # Create model with same architecture as official SALAD
    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
        lr=lr,
        optimizer='adamw',
        weight_decay=9.5e-9,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args={
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner',
        miner_margin=0.1,
        faiss_gpu=False
    )
    
    # Load pretrained weights
    if Path(ckpt_path).exists():
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Load weights (strict=False allows missing keys like optimizer state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
        
        print("Pretrained weights loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}, training from scratch")
    
    return model


def main():
    args = parse_args()
    
    # Validate inputs
    if not Path(args.train_csv).exists():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup DataModule
    datamodule = BagPlacesDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.img_per_place,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        random_sample=True,
        use_augmentation=True
    )
    
    # Load model
    model = load_pretrained_model(args.pretrained, args.lr)
    
    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename=f'{args.experiment_name}' + '_{epoch:02d}_{loss:.4f}',
        save_top_k=3,
        save_last=True,
        monitor='loss',
        mode='min'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        default_root_dir=str(output_dir),
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    
    # Print info
    print("\n" + "="*60)
    print("SALAD Fine-tuning on RatSLAM Data")
    print("="*60)
    print(f"Training CSV: {args.train_csv}")
    print(f"Pretrained:   {args.pretrained}")
    print(f"Batch size:   {args.batch_size} places × {args.img_per_place} images")
    print(f"Epochs:       {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output dir:   {output_dir}")
    print("="*60 + "\n")
    
    # Train
    trainer.fit(model=model, datamodule=datamodule)
    
    # Save final model
    final_path = output_dir / 'checkpoints' / f'{args.experiment_name}_final.ckpt'
    trainer.save_checkpoint(str(final_path))
    print(f"\nFinal checkpoint saved to: {final_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"To use the fine-tuned model with RatSLAM:")
    print(f"  1. Copy {final_path} to salad-main/weights/")
    print(f"  2. Update SALAD_CKPT environment variable")
    print(f"  3. Restart salad-svc container")
    print("="*60)


if __name__ == '__main__':
    main()
