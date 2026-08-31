import torch

# Greedy decoding

def generate_text_simple(model, idx, max_new_tokens, context_size): # `idx` is (batch, n_tokens) array of indices in current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # Crop current context if exceeding the supported context size
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :] # Focus only on the last time step, so that (batch, n_token, vocab_size) -> (batch, vocab_size)
        probs = torch.softmax(logits, dim=-1) # Of shape (batch, vocab_size)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True) # Of shape (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1) # Append sampled index to running seq, where `idx` has shape (batch, n_tokens + 1)

    return idx