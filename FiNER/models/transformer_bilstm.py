import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel, RobertaTokenizer

from tf2crf import CRF


class TransformerBiLSTM(tf.keras.Model):
    def __init__(
            self,
            model_name,
            n_classes,
            dropout_rate=0.1,
            crf=False,
            n_layers=1,
            n_units=128,
            tokenizer=None,
            subword_pooling='all'
    ):
        super().__init__()

        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.crf = crf
        self.n_layers = n_layers
        self.n_units = n_units
        self.subword_pooling = subword_pooling

        self.encoder = TFAutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name
        )
        if tokenizer:
            self.encoder.resize_token_embeddings(
                new_num_tokens=len(tokenizer.vocab))

        self.bilstm_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=n_units,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    return_sequences=True,
                    name=f'BiLSTM_{i + 1}'
                )
            ) for i in range(n_layers)
        ]

        if self.crf:
            self.classifier = tf.keras.layers.Dense(
                units=n_classes,
                activation=None
            )

            # Pass logits to a custom CRF Layer
            self.crf_layer = CRF(output_dim=n_classes, mask=True)
        else:
            self.classifier = tf.keras.layers.Dense(
                units=n_classes,
                activation='softmax'
            )

    def call(self, inputs, training=None, mask=None):

        if self.subword_pooling in ['first', 'last']:
            pooling_mask = inputs[1]
            inputs = inputs[0]

        encodings = self.bert_encoder(inputs)[0]
        encodings = tf.keras.layers.SpatialDropout1D(
            rate=self.dropout_rate
        )(encodings, training=training)

        for i, bilstm_layer in enumerate(self.bilstm_layers):
            encodings = bilstm_layer(encodings)
            encodings = tf.keras.layers.SpatialDropout1D(
                rate=self.dropout_rate
            )(encodings, training=training)

        outputs = self.classifier(encodings)

        if self.crf:
            outputs = self.crf_layer(outputs, mask=tf.not_equal(inputs, 0))

        if self.subword_pooling in ['first', 'last']:
            outputs = tf.cast(tf.expand_dims(pooling_mask, axis=-1), dtype=tf.float32) * outputs

        return outputs

    def print_summary(self, line_length=None, positions=None, print_fn=None):
        # Fake forward pass to build graph
        batch_size, sequence_length = 1, 32
        inputs = np.ones((batch_size, sequence_length), dtype=np.int32)

        if self.subword_pooling in ['first', 'last']:
            pooling_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
            inputs = [inputs, pooling_mask]

        self.predict(inputs)
        self.summary(line_length=line_length, positions=positions, print_fn=print_fn)


if __name__ == '__main__':
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Init random seeds
    np.random.seed(1)
    tf.random.set_seed(1)

    model_name = 'nlpaueb/sec-bert-base'

    # Build test model
    model = TransformerBiLSTM(
        model_name=model_name,
        n_classes=10,
        dropout_rate=0.2,
        crf=False,
        n_layers=1,
        n_units=128,
        subword_pooling='all'
    )

    # inputs = pad_sequences(np.random.randint(0, 30000, (5, 32)), maxlen=64, padding='post', truncating='post')
    inputs = [
        'This is the first sentence',
        'This is the second sentence',
        'This is the third sentence',
        'This is the fourth sentence',
        'This is the last sentence, this is a longer sentence']

    tokenizer = RobertaTokenizer.from_pretrained(
        pretrained_model_name_or_path="roberta-base",
        use_fast=True
    )

    inputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=inputs,
        add_special_tokens=False,
        max_length=64,
        padding='max_length',
        return_tensors='tf'
    ).input_ids

    outputs = pad_sequences(np.random.randint(0, 10, (5, 32)), maxlen=64, padding='post', truncating='post')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=5.0)

    if model.crf:
        model.compile(
            optimizer=optimizer,
            loss=model.crf_layer.loss,
            run_eagerly=True
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            run_eagerly=True
        )

    print(model.print_summary(line_length=150))

    model.fit(x=inputs, y=outputs, batch_size=2)
    model.predict(inputs, batch_size=1)
    predictions = model.predict(inputs, batch_size=2)
    print(predictions)
