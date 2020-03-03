# rv_bachelor_thesis

## Motivation
The calibration of volatility models poses the bottleneck when pricing derivative contracts. As shown in \cite{Horvath:19, Hernandez:17} this can be lifted using Neural Networks. By going along a similar process I would deepen my knowledge of Stochastic ODEs and Deep Neural Networks, while having to implement a way to sample from the SDE systems, design and train a neural network, and implement a least squares solver. \\

## Content
### Section 1: \\
Deep Learning; Summary of approximation Theorems, prove of Universal Approximation Property (UAP) \\
### Section 2: \\
Short Recap of Black-Scholes Theory + limitations; introduce Heston and SABR models; explain calibration. \\
### Section 3: \\
Calibration with Neural Networks (NN) \\
1. Simulation of Market using different parameters \\
2. Generation of training data using Monte-Carlo, SDE Methods (Euler Maruyama, Milstein), Fast-Fourier-Transform (for Heston) \\
3. Implementation and training of NN using Tensorflow: $$ \hat{w}= argmin_{w \in \Re} \sum_{u=1}^{N_{train}} \sum_{i=1}^{n} (F(\theta_{u},w)_{i}-F^{*}(\theta_{u})_{i})^2$$ \\
4. Perform Calibration by implementing some least square solver such as Levenberg-Marquardt : $$ \hat{\theta} = argmin_{\theta} \sum_{i=0}^{n} (\hat{F}(\theta)- P^{MKT}(\xi_{i}))^2 $$ \\ \\
### Section 4: \\
Performance assessment: Generate synthetic data using parameter set  $ \theta^{*} $ and assess how close  $ \hat{\theta} $ is (probably in RMSE sense to be parameter insensitive). As synthetic data is used as "ground truth" look at convergence of methods when generating it. \\
