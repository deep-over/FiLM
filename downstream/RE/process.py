from transformers import AutoTokenizer, AutoModel
# add special tokens '<e1>', '</e1>', '<e2>', '</e2>' to tokenizer
def add_special_tokens(args) :
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModel.from_pretrained(args.model)
    
    crit_token = tokenizer.pad_token

    head_start = crit_token[0] + ('e1' if crit_token[1:-1].islower() else 'e1'.upper()) + crit_token[-1]
    head_end = crit_token[0] + ('/e1' if crit_token[1:-1].islower() else '/e1'.upper()) + crit_token[-1]
    tail_start = crit_token[0] + ('e2' if crit_token[1:-1].islower() else 'e2'.upper()) + crit_token[-1]
    tail_end = crit_token[0] + ('/e2' if crit_token[1:-1].islower() else '/e2'.upper()) + crit_token[-1]

    special_tokens = {
        'head_start' : head_start,
        'head_end' : head_end,
        'tail_start' : tail_start,
        'tail_end' : tail_end
    }

    num_added_tokens = tokenizer.add_tokens([head_start, head_end, tail_start, tail_end], special_tokens=True)
    print("added", num_added_tokens, "special tokens")
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model, special_tokens