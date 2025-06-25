import torch

from data import START_TOKEN, END_TOKEN


def decode_sequence_greedy(model, img_patches, max_len=10):
    """
    Autoregressively decode output sequences from image patches using greedy decoding.
    Supports batch input.
    """
    model.eval()
    device = next(model.parameters()).device

    B = img_patches.size(0)

    with torch.no_grad():
        # Encode image patches once
        memory = model.encode(img_patches.to(device))  # (B, N, D)

        # Initialize decoder input with <START> token
        sequences = torch.full((B, 1), START_TOKEN, dtype=torch.long, device=device)  # (B, 1)
        finished = torch.zeros(B, dtype=torch.bool, device=device)  # track finished sequences

        for _ in range(max_len):
            # Decode current sequence
            logits = model.decode(memory, sequences)   # (B, T, vocab_size)
            next_token = logits[:, -1].argmax(dim=-1)  # (B,)

            # Append predicted token to the sequence
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)  # (B, T+1)

            # Check if any sequences have reached <END>
            finished |= next_token == END_TOKEN
            if finished.all():
                break

        # Remove <START> token (first column)
        result = sequences[:, 1:]  # (B, T)

        # Truncate at first <END> in each sequence
        cleaned = []
        for seq in result:
            end_indices = (seq == END_TOKEN).nonzero(as_tuple=False)
            if len(end_indices) > 0:
                cut = end_indices[0].item()
                seq = seq[:cut]
            cleaned.append(seq)

        # Pad to max_len with END_TOKEN (or you can choose PAD_TOKEN)
        padded = torch.full((B, max_len), END_TOKEN, dtype=torch.long, device=device)
        for i, seq in enumerate(cleaned):
            padded[i, :len(seq)] = seq

        return padded.cpu().tolist()

# TODO Implement beam search decoding

