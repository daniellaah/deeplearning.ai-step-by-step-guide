```python
python3 main.py
```
Note: First turn off dropout and turn on gradient checking to make sure the gradient is correct. Then switch.

```python
# output
# ---------------Turn off dropout, Turn on gradient checking-----------------
# Your backward propagation works perfectly fine! difference = 1.86813763508e-08
# Cost after iteration 0: 0.655741
# Your backward propagation works perfectly fine! difference = 2.01415418548e-08
# Your backward propagation works perfectly fine! difference = 1.75673375682e-08
# Your backward propagation works perfectly fine! difference = 1.93813452735e-08
# Your backward propagation works perfectly fine! difference = 1.83391181377e-08
# Your backward propagation works perfectly fine! difference = 1.44548829282e-08
# Your backward propagation works perfectly fine! difference = 1.60431352868e-08
# Your backward propagation works perfectly fine! difference = 1.31567302369e-08
# Your backward propagation works perfectly fine! difference = 1.2674889404e-08
# Your backward propagation works perfectly fine! difference = 1.19131701283e-08
# Your backward propagation works perfectly fine! difference = 1.15320601166e-08
# Your backward propagation works perfectly fine! difference = 1.26298729769e-08
# Your backward propagation works perfectly fine! difference = 1.10348100704e-08
# Your backward propagation works perfectly fine! difference = 1.07930688483e-08
# Your backward propagation works perfectly fine! difference = 9.88720955475e-09
# Your backward propagation works perfectly fine! difference = 1.03708188535e-08
# Your backward propagation works perfectly fine! difference = 9.63672890921e-09
# ...
# ...
# ---------------Turn on dropout, Turn off gradient checking-----------------
# Cost after iteration 0: 0.654391
# Cost after iteration 10000: 0.061191
# Cost after iteration 20000: 0.060631
# accuracy on train without regularization: 92.42%
# accuracy on test without regularization: 95.00%
```
