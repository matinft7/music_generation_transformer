import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from remi_tokenizer import REMITokenizer

class MaestroDataset(Dataset):
    def __init__(self, csv_path, processed_dir, split="train", seq_len=1024):
        """
        Args:
            csv_path (str): Path to maestro-v2.0.0.csv
            processed_dir (str): Base path for preprocessed .pt files (/kaggle/working/processed_maestro/)
            split (str): 'train' or 'validation'
            seq_len (int): Sequence length for sliding window
        """
        self.processed_dir = processed_dir
        self.seq_len = seq_len
        
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        
        self.pt_files = []
        for fname in df['midi_filename']:
            pt_name = fname.replace('.midi', '.pt').replace('.mid', '.pt')
            pt_path = os.path.join(processed_dir, pt_name)
            self.pt_files.append(pt_path)
        
        self.token_cache = {}
        self.estimated_sequences_per_file = 20
        
    def __len__(self):
        return len(self.pt_files) * self.estimated_sequences_per_file
        
    def _get_tokens(self, file_idx):
        if file_idx not in self.token_cache:
            pt_path = self.pt_files[file_idx]
            try:
                tokens = torch.load(pt_path).long()
                self.token_cache[file_idx] = tokens
            except Exception as e:
                # If file missing or corrupted
                self.token_cache[file_idx] = torch.tensor([], dtype=torch.long)
        return self.token_cache[file_idx]

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            file_idx = (idx // self.estimated_sequences_per_file) % len(self.pt_files)
            token_ids = self._get_tokens(file_idx)
            
            if len(token_ids) >= self.seq_len + 1:
                break
                
            # If the file is too short or corrupted, randomly pick another index
            idx = torch.randint(0, len(self), (1,)).item()
            attempts += 1
            
        if attempts == max_attempts:
            raise RuntimeError(f"Could not find a valid sequence after {max_attempts} attempts. " 
                               f"Please ensure dataset is preprocessed correctly and files are larger than seq_len={self.seq_len}.")
            
        max_start = len(token_ids) - (self.seq_len + 1)
        start_idx = torch.randint(0, max_start + 1, (1,)).item()
        
        window = token_ids[start_idx : start_idx + self.seq_len + 1]
        
        input_ids = window[:-1]
        target_ids = window[1:]
        
        return input_ids, target_ids

def get_dataloader(csv_path, processed_dir, split="train", batch_size=8, seq_len=1024, num_workers=2):
    dataset = MaestroDataset(csv_path, processed_dir, split, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), 
                            num_workers=num_workers, drop_last=True)
    return dataloader
