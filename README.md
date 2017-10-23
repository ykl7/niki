## Model

Simple LSTM with Wor2Vec inputs
Categorical crossentropy loss
Adadelta optimization

## Requirements

```pip install -r requirements.txt```

[*] keras
[*] numpy
[*] scikit-learn
[*] pandas
[*] gensim

preprocessed Word2Vec vectors trained over Google News dataset

trained model weights

## Command sequence

```python data_clean.py```

```python embedding.py```

```python model.py```