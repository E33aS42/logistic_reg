# logistic_reg
An exercise to mathematically implement a logistic regression model and explore methods to decrease computational time (bonus).

# Introduction
Based on a dataset of the notes obtained by the students of Hogwarts School of Witchcraft and Wizardry, I built a logistic regression that could classify to which of the 4 houses each student belongs to in function of their notes.

This project was done in 3 parts:
1. A data analysis helped identify which features to keep among the set of 20 different features.
2. A logitistic regression model was trained on our training set and could reach an accuracy above 98% on a provided test set .
3. Implementation of other optimization algorithms (stochastic gradient descent, minibatch GD, loss momentum, SGD+loss momentum) and comparison of the gain in computational time.

# Results
1. 3 relevant and uncorrelated features were identified: ['Astronomy', 'Ancient Runes', 'Charms']
   
![histograms](https://github.com/E33aS42/logistic_reg/assets/66993020/9c467e84-5ff9-4fb9-8182-c747be78a9ef)
![pair_plot_5](https://github.com/E33aS42/logistic_reg/assets/66993020/919e3673-d524-4d27-94fc-0b6213e3f9aa)

3. Classification results:
   
![pred-reg](https://github.com/E33aS42/logistic_reg/assets/66993020/b615fc35-9d82-4d51-a062-1cab1aa10cbc)

4. Optimization:
Applying Loss momentum is by far the most efficient algorithm as it reduces the number of iterations by 10 and the computational time by 7 compared to the basic logistic regression.

![loss_with_batch](https://github.com/E33aS42/logistic_reg/assets/66993020/3d1097e8-b025-4b85-ba0c-927bdc626c39)
![loss_with_minibatch](https://github.com/E33aS42/logistic_reg/assets/66993020/a032433a-95d4-414e-b094-d702eb5d765b)
![loss_with_sgd](https://github.com/E33aS42/logistic_reg/assets/66993020/c0e97bbd-a431-479d-a66e-e09d1df4f051)
![loss_with_momentum](https://github.com/E33aS42/logistic_reg/assets/66993020/c121d0d2-1738-40f5-8ac8-0c89a8bd5a87)
![loss_with_sgdmom](https://github.com/E33aS42/logistic_reg/assets/66993020/0168172a-defa-4561-9733-eeca64aa1290)
