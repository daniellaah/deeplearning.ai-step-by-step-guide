```python
python3 main.py
```
Note: If you want to reproduce the result, be careful of the random seed.

```python
# output
# -----------------------without regularization------------------------
# Cost after iteration 0: 0.655741
# Cost after iteration 10000: 0.163300
# Cost after iteration 20000: 0.138516
# accuracy on train without regularization: 94.79%
# accuracy on test without regularization: 91.50%
# --------------------------l2 regularization--------------------------
# Cost after iteration 0: 0.697448
# Cost after iteration 10000: 0.268492
# Cost after iteration 20000: 0.268092
# accuracy on train without regularization: 93.84%
# accuracy on test without regularization: 93.00%
# -----------------------dropout regularization-------------------------
# Cost after iteration 0: 0.654391
# Cost after iteration 10000: 0.061191
# Cost after iteration 20000: 0.060631
# accuracy on train without regularization: 92.42%
# accuracy on test without regularization: 95.00%
```

![](https://github.com/daniellaah/deeplearning.ai-step-by-step-guide/blob/master/02-Improving-Deep-Neural-Networks/week1/02-regularization/img/none-reg.png)

![](https://github.com/daniellaah/deeplearning.ai-step-by-step-guide/blob/master/02-Improving-Deep-Neural-Networks/week1/02-regularization/img/l2-reg.png)

![](https://github.com/daniellaah/deeplearning.ai-step-by-step-guide/blob/master/02-Improving-Deep-Neural-Networks/week1/02-regularization/img/dropout-reg.png)
