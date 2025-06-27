import torch
import os
import pandas as pd
from pathlib import Path
from model import Classifier
import argparse

def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def load_checkpoint_info(checkpoint_path):
    """Load and extract information from a checkpoint file"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if model actually has positional encoding by examining state_dict
        has_positional_encoding = False
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            for key in state_dict.keys():
                if 'positional_encoding' in key:
                    has_positional_encoding = True
                    break
        
        # Handle both old and new normalization parameter names
        # Old format: has_normalization_layer_1 through has_normalization_layer_5
        # New format: has_input_norm, has_pre_attention_norm, has_post_attention_norm, has_post_ffn_norm, has_final_norm
        
        # Check if this is a new format checkpoint
        has_new_format = any(key.startswith('has_input_norm') or key.startswith('has_pre_attention_norm') 
                           for key in checkpoint.keys())
        
        if has_new_format:
            # New format normalization parameters
            norm_layers = [
                checkpoint.get('has_input_norm', False),
                checkpoint.get('has_pre_attention_norm', False),
                checkpoint.get('has_post_attention_norm', False),
                checkpoint.get('has_post_ffn_norm', False),
                checkpoint.get('has_final_norm', False),
            ]
            norm_names = ['inp', 'pre', 'post', 'ffn', 'fin']
        else:
            # Old format normalization parameters
            norm_layers = [
                checkpoint.get('has_normalization_layer_1', False),
                checkpoint.get('has_normalization_layer_2', False),
                checkpoint.get('has_normalization_layer_3', False),
                checkpoint.get('has_normalization_layer_4', False),
                checkpoint.get('has_normalization_layer_5', False),
            ]
            norm_names = ['1', '2', '3', '4', '5']
        
        # Convert to compact format: "inp,pre,post,ffn,fin" or "1,2,3,4,5" where names show True positions
        norm_compact = ','.join([norm_names[i] if norm_layers[i] else 'x' for i in range(5)])
        
        # Extract hyperparameters and metrics
        info = {
            'checkpoint': os.path.basename(checkpoint_path),
            'score': checkpoint.get('score', 0.0),
            'timestamp': checkpoint.get('timestamp', 'Unknown'),
            'epochs': checkpoint.get('epochs', 'Unknown'),
            'learning_rate': checkpoint.get('learning_rate', 'Unknown'),
            'batch_size': checkpoint.get('batch_size', 'Unknown'),
            'patch_kernal_size': checkpoint.get('patch_kernal_size', 'Unknown'),
            'patch_stride': checkpoint.get('patch_stride', 'Unknown'),
            'dim_model': checkpoint.get('dim_model', 'Unknown'),
            'dim_k': checkpoint.get('dim_k', 'Unknown'),
            'dim_v': checkpoint.get('dim_v', 'Unknown'),
            'dropout_rate': checkpoint.get('dropout_rate', 'N/A'),  # New parameter
            'num_encoders': checkpoint.get('num_encoders', 1),
            'has_positional_encoding': has_positional_encoding,  # Actually detected from state_dict
            'normalization': norm_compact,  # Compact representation
            'num_heads': checkpoint.get('num_heads', 1),  # Default to 1 head if not specified
        }
        
        # Add individual normalization flags for model recreation (backward compatibility)
        if has_new_format:
            info.update({
                'has_input_norm': checkpoint.get('has_input_norm', False),
                'has_pre_attention_norm': checkpoint.get('has_pre_attention_norm', False),
                'has_post_attention_norm': checkpoint.get('has_post_attention_norm', False),
                'has_post_ffn_norm': checkpoint.get('has_post_ffn_norm', False),
                'has_final_norm': checkpoint.get('has_final_norm', False),
                # Map to old names for model creation compatibility
                'has_normalization_layer_1': checkpoint.get('has_input_norm', False),
                'has_normalization_layer_2': checkpoint.get('has_pre_attention_norm', False),
                'has_normalization_layer_3': checkpoint.get('has_post_attention_norm', False),
                'has_normalization_layer_4': checkpoint.get('has_post_ffn_norm', False),
                'has_normalization_layer_5': checkpoint.get('has_final_norm', False),
            })
        else:
            info.update({
                'has_normalization_layer_1': checkpoint.get('has_normalization_layer_1', False),
                'has_normalization_layer_2': checkpoint.get('has_normalization_layer_2', False),
                'has_normalization_layer_3': checkpoint.get('has_normalization_layer_3', False),
                'has_normalization_layer_4': checkpoint.get('has_normalization_layer_4', False),
                'has_normalization_layer_5': checkpoint.get('has_normalization_layer_5', False),
            })
        
        # Try to recreate model to count parameters
        try:
            # Use new parameter names for current Classifier constructor
            model_params = {
                'patch_size': info['patch_kernal_size'],
                'dim_model': info['dim_model'],
                'dim_k': info['dim_k'],
                'dim_v': info['dim_v'],
                'dropout_rate': info['dropout_rate'] if info['dropout_rate'] != 'N/A' else 0.0,
                'has_positional_encoding': info['has_positional_encoding'],
                'has_input_norm': info.get('has_input_norm', info['has_normalization_layer_1']),
                'has_pre_attention_norm': info.get('has_pre_attention_norm', info['has_normalization_layer_2']),
                'has_post_attention_norm': info.get('has_post_attention_norm', info['has_normalization_layer_3']),
                'has_post_ffn_norm': info.get('has_post_ffn_norm', info['has_normalization_layer_4']),
                'has_final_norm': info.get('has_final_norm', info['has_normalization_layer_5']),
                'num_encoders': info['num_encoders'],
                'num_heads': checkpoint.get('num_heads', 1)  # Default to 1 head if not specified
            }
            
            model = Classifier(**model_params)
            total_params, trainable_params = count_parameters(model)
            info['total_params'] = total_params
            info['trainable_params'] = trainable_params
        except Exception as e:
            print(f"Error recreating model for {checkpoint_path}: {e}")
            info['total_params'] = 'Error'
            info['trainable_params'] = 'Error'
            
        return info
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def compare_models(checkpoints_dir="checkpoints", sort_by="score", ascending=False):
    """Compare all models in the checkpoints directory"""
    
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory '{checkpoints_dir}' not found!")
        return
    
    # Find all .pth files
    checkpoint_files = list(Path(checkpoints_dir).glob("*.pth"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoints_dir}'!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print("Loading checkpoint information...\n")
    
    # Load information from all checkpoints
    model_info = []
    for checkpoint_path in checkpoint_files:
        info = load_checkpoint_info(checkpoint_path)
        if info:
            model_info.append(info)
    
    if not model_info:
        print("No valid checkpoints found!")
        return
    
    # Create DataFrame for easy comparison
    df = pd.DataFrame(model_info)
    
    # Sort by specified column
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    
    return df

def print_summary(df):
    """Print a summary of the model comparison"""
    print("=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)
    
    # Basic statistics
    print(f"Total models: {len(df)}")
    print(f"Best accuracy: {df['score'].max():.4f}")
    print(f"Worst accuracy: {df['score'].min():.4f}")
    print(f"Average accuracy: {df['score'].mean():.4f}")
    print(f"Standard deviation: {df['score'].std():.4f}")
    print()
    
    # Top 5 performers
    print("TOP 5 PERFORMERS:")
    print("-" * 50)
    available_cols = ['checkpoint', 'score', 'dim_model', 'num_encoders', 'dropout_rate', 'learning_rate', 'has_positional_encoding', 'normalization', 'num_heads', 'total_params']
    display_cols = [col for col in available_cols if col in df.columns]
    top_5 = df.nlargest(5, 'score')[display_cols].copy()
    # Format score as percentage
    top_5['score'] = top_5['score'].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
    # Format parameters with commas
    if 'total_params' in top_5.columns:
        top_5['total_params'] = top_5['total_params'].apply(
            lambda x: f"{x:,}" if isinstance(x, (int, float)) else x
        )
    print(top_5.to_string(index=False))
    print()
    
    # Parameter analysis
    if 'total_params' in df.columns and df['total_params'].dtype != 'object':
        print("PARAMETER ANALYSIS:")
        print("-" * 50)
        print(f"Smallest model: {df['total_params'].min():,} parameters")
        print(f"Largest model: {df['total_params'].max():,} parameters")
        print(f"Average model size: {df['total_params'].mean():,.0f} parameters")
        print()

def print_detailed_comparison(df):
    """Print detailed comparison table"""
    print("DETAILED COMPARISON:")
    print("=" * 150)
    
    # Select columns to display
    display_columns = [
        'checkpoint', 'score', 'dim_model', 'dim_k', 'dim_v', 'num_encoders', 'dropout_rate',
        'learning_rate', 'batch_size', 'patch_kernal_size', 
        'has_positional_encoding', 'normalization', 'num_heads',
        'total_params', 'timestamp'
    ]
    
    # Filter columns that exist in the dataframe
    available_columns = [col for col in display_columns if col in df.columns]
    
    display_df = df[available_columns].copy()
    
    # Format score as percentage
    if 'score' in display_df.columns:
        display_df['score'] = display_df['score'].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
    
    # Format parameters with commas
    if 'total_params' in display_df.columns:
        display_df['total_params'] = display_df['total_params'].apply(
            lambda x: f"{x:,}" if isinstance(x, (int, float)) else x
        )
    
    print(display_df.to_string(index=False))
    print()

def main():
    parser = argparse.ArgumentParser(description='Compare model checkpoints')
    parser.add_argument('--checkpoints-dir', default='checkpoints', 
                       help='Directory containing checkpoint files')
    parser.add_argument('--sort-by', default='score', 
                       help='Column to sort by (default: score)')
    parser.add_argument('--ascending', action='store_true', 
                       help='Sort in ascending order (default: descending)')
    parser.add_argument('--save-csv', type=str, 
                       help='Save comparison to CSV file')
    
    args = parser.parse_args()
    
    # Compare models
    df = compare_models(
        checkpoints_dir=args.checkpoints_dir, 
        sort_by=args.sort_by, 
        ascending=args.ascending
    )
    
    if df is None or df.empty:
        return
    
    # Print results
    print_summary(df)
    print_detailed_comparison(df)
    
    # Save to CSV if requested
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"Comparison saved to {args.save_csv}")

if __name__ == "__main__":
    main() 