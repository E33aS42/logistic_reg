# logistic_reg
An exercise to mathematically implement a logistic regression model and explore methods to decrease computational time (bonus).

# Introduction
Based on a dataset of the notes obtained by the students of Hogwarts School of Witchcraft and Wizardry, I built a logistic regression that could classify to which of the 4 houses each student belongs to in function of their notes.

This project was done in 3 parts:
1. A data analysis helped identify which features to keep among the set of 20 different features.
2. A logitistic regression model was trained on our training set and could reach an accuracy above 98% on a provided test set .
3. Implementation of other optimization algorithms (stochastic gradient descent, minibatch GD, loss momentum, SGD+loss momentum) and comparison of the gain in computational time.

# Results
1. 3 relevant and uncorelated features were identified:
![histograms](https://github.com/E33aS42/logistic_reg/assets/66993020/9c467e84-5ff9-4fb9-8182-c747be78a9ef)
![pair_plot_5](https://github.com/E33aS42/logistic_reg/assets/66993020/919e3673-d524-4d27-94fc-0b6213e3f9aa)

2. Classification results:
![predictions_98%](https://github.com/E33aS42/logistic_reg/assets/66993020/01e5492e-708d-45d9-996e-d481a4d063b4)

3. Otimization:
