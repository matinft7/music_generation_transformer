import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
    
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        
    return logits

def generate_music(model, tokenizer, prompt_tokens=["Bar"], max_length=1024, temperature=1.0, top_k=50, top_p=0.9, save_path="/kaggle/working/generated.mid"):
    """
    Autoregressively generate music sequence.
    """
    device = next(model.parameters()).device
    model.eval()
    
    input_ids = tokenizer.vocab.encode(prompt_tokens)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"Starting generation for {max_length} tokens...")
    
    with torch.no_grad():
        is_batch_run = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
        for _ in tqdm(range(max_length), disable=is_batch_run):
            # Forward pass
            logits = model(input_tensor)
            
            # Take the logits for the last token in the sequence
            next_token_logits = logits[0, -1, :] / (temperature if temperature > 0 else 1.0)
            
            # Filter logits
            filtered_logits = top_k_top_p_filtering(next_token_logits.unsqueeze(0), top_k=top_k, top_p=top_p)
            
            # Sample next token
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            
            # Append next token to the input sequence
            input_tensor = torch.cat([input_tensor, next_token_id], dim=1)
            
            # Stop early if <EOS> token is generated
            if next_token_id.item() == tokenizer.vocab.eos_id:
                print("Generated <EOS> token. Stopping early.")
                break
                
    # Convert IDs back to tokens
    generated_ids = input_tensor[0].cpu().tolist()
    generated_tokens = tokenizer.vocab.decode(generated_ids)
    
    # Save to MIDI
    print(f"Saving generated MIDI to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.tokens_to_midi(generated_tokens, save_path)
    
    return generated_tokens
