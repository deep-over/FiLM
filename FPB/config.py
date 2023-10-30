class Config :
    def __init__(self, model, tokenizer, max_length, batch_size, num_workers, learning_rate) :
        self.model = model
        self.tokenizer = tokenizer
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.learning_rate = learning_rate
