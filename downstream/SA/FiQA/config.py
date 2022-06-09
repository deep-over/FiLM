class Config :
    def __init__(self, model, max_length, batch_size, num_workers, learning_rate, dropout) :
        '''
        model(string) : huggingface transformer 라이브러리 from_pretrained 함수로 부를 수 있는 모델 이름
        max_length(int) : 토크나이징 이후 최대 시퀀스 길이
        batch_size(int)
        num_workers(int) : 데이터 로더 만드는데 사용할 worker 수
        learning_rate(float)
        dropout(float) : dropout 확률
        '''
        self.model = model
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.learning_rate = learning_rate
        self.dropout = dropout
