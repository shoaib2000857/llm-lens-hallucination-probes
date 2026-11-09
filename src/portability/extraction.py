"""Batched legacy-aligned generation-step representations, with EOS-safe padding."""
import torch


@torch.inference_mode()
def extract_batch(model, tokenizer, prompts, max_input=2048, max_new=96):
    tokenizer.padding_side = 'left'
    encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_input)
    encoded = {k:v.to(model.device) for k,v in encoded.items()}
    prompt_width = encoded['input_ids'].shape[1]
    output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new,
        return_dict_in_generate=True, output_hidden_states=True, output_logits=True,
        use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    generated = output.sequences[:, prompt_width:]
    steps = len(output.hidden_states)
    batch = len(prompts)
    lengths = torch.full((batch,), steps, dtype=torch.long, device=generated.device)
    if tokenizer.eos_token_id is not None:
        for b in range(batch):
            eos = torch.where(generated[b, :steps] == tokenizer.eos_token_id)[0]
            if len(eos): lengths[b] = eos[0]+1
    # Select last position BEFORE stacking layers: avoids copying full prompt activations.
    gathered = torch.stack([torch.stack([layer[:, -1, :] for layer in step[1:]], dim=1)
                            for step in output.hidden_states], dim=2)
    hidden = torch.zeros((batch, gathered.shape[1], max_new, gathered.shape[-1]), dtype=torch.float16, device='cpu')
    confidence = []; entropies = []
    for t, logits in enumerate(output.logits):
        lp = torch.log_softmax(logits.float(), dim=-1)
        token_lp = lp.gather(1, generated[:, t:t+1]).squeeze(1)
        entropy = -(lp.exp()*lp).sum(dim=-1)
        confidence.append(token_lp);entropies.append(entropy)
    confidence=torch.stack(confidence,dim=1).cpu().tolist()
    entropies=torch.stack(entropies,dim=1).cpu().tolist()
    lengths=lengths.cpu().tolist()
    generated=generated.cpu()
    input_ids=encoded['input_ids'].cpu();attention_mask=encoded['attention_mask'].cpu()
    result=[]
    for b in range(batch):
        length=int(lengths[b]); hidden[b, :, :length] = gathered[b, :, :length].to(device='cpu', dtype=torch.float16)
        ids=generated[b, :length].tolist()
        prompt_ids=input_ids[b][attention_mask[b].bool()].tolist()
        confidence[b]=confidence[b][:length];entropies[b]=entropies[b][:length]
        result.append(dict(hidden=hidden[b].numpy(), generated_text=tokenizer.decode(ids, skip_special_tokens=True),
            generated_token_ids=ids, prompt_token_ids=prompt_ids, n_gen_tokens=length,
            token_logprobs=confidence[b], token_entropies=entropies[b],
            mean_token_logprob=sum(confidence[b])/max(length,1), min_token_logprob=min(confidence[b]),
            mean_token_entropy=sum(entropies[b])/max(length,1)))
    return result
