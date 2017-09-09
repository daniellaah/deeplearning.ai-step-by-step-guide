# Neural Networks Basics

```python
python3 main.py
```

```python
# output:
# Number of training examples: m_train = 209
# Number of testing examples: m_test = 50
# Height/Width of each image: num_px = 64
# Each image is of size: (64, 64, 3)
# train_set_x shape: (209, 64, 64, 3)
# train_set_y shape: (1, 209)
# test_set_x shape: (50, 64, 64, 3)
# test_set_y shape: (1, 50)
# train_set_x_flatten shape: (12288, 209)
# train_set_y shape: (1, 209)
# test_set_x_flatten shape: (12288, 50)
# test_set_y shape: (1, 50)
# sanity check after reshaping: [17 31 56 22 33]
# Cost after iteration 0: 0.693147
# Cost after iteration 500: 0.303273
# Cost after iteration 1000: 0.214820
# Cost after iteration 1500: 0.166521
# Cost after iteration 2000: 0.135608
# training acc: 0.9904306220095693
# testing acc: 0.7
```