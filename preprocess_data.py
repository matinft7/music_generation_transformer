import os
import pandas as pd
import torch
from tqdm import tqdm
from remi_tokenizer import REMITokenizer

def preprocess_maestro(csv_path, base_path, output_dir):
    """
    Reads all MIDI files in MAESTRO, converts them to token IDs, 
    and saves them as PyTorch tensors (.pt files).
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = REMITokenizer()
    
    df = pd.read_csv(csv_path)
    
    print(f"Starting preprocessing of {len(df)} files...")
    
    is_batch_run = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
    for idx, row in tqdm(df.iterrows(), total=len(df), disable=is_batch_run):
        midi_filename = row['midi_filename']
        split = row['split']
        
        if split not in ['train', 'validation']:
            continue
            
        midi_path = os.path.join(base_path, midi_filename)
        
        save_sub_dir = os.path.join(output_dir, os.path.dirname(midi_filename))
        os.makedirs(save_sub_dir, exist_ok=True)
        
        save_name = os.path.basename(midi_filename).replace('.midi', '.pt').replace('.mid', '.pt')
        save_path = os.path.join(save_sub_dir, save_name)
        
        if os.path.exists(save_path):
            continue
            
        try:
            tokens = tokenizer.midi_to_tokens(midi_path)
            ids = tokenizer.vocab.encode(tokens)
            
            # Save as int16 to save disk space
            tensor_ids = torch.tensor(ids, dtype=torch.int16)
            torch.save(tensor_ids, save_path)
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")

if __name__ == "__main__":
    # Example usage for Kaggle
    csv_path = '/kaggle/input/datasets/jackvial/themaestrodatasetv2/maestro-v2.0.0/maestro-v2.0.0.csv'
    base_path = '/kaggle/input/datasets/jackvial/themaestrodatasetv2/maestro-v2.0.0/'
    output_dir = '/kaggle/working/processed_maestro'
    
    if os.path.exists(csv_path):
        preprocess_maestro(csv_path, base_path, output_dir)
    else:
        print("CSV not found. This script should be run in the Kaggle environment.")
