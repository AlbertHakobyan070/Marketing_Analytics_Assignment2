# Multi-Armed Bandit: A/B Testing with Epsilon-Greedy and Thompson Sampling

A Marketing Analytics assignment implementing two classic bandit algorithms 
to simulate an A/B testing scenario across four advertisement options.

## Problem Setup

Four advertisements (bandits) with reward means of `[1, 2, 3, 4]` are tested 
over 20,000 trials. The goal is to compare how quickly and efficiently each 
algorithm identifies and exploits the best-performing ad.

## Algorithms

**Epsilon-Greedy** explores with probability `ε = 1/t` (decaying over time) 
and exploits the best known arm otherwise. The decaying epsilon gradually 
shifts the strategy from exploration to exploitation as trials increase.

**Thompson Sampling** uses Bayesian inference with known precision. At each 
step it samples from the posterior distribution of each arm and pulls the one 
with the highest sample — naturally balancing exploration and exploitation 
without a manual epsilon parameter.

## Results

### Learning Process (Cumulative Average Reward)
![Learning Process](Learning%20Process%20plot%20comparison%20(linear%20and%20log).png)

### Cumulative Reward & Regret Comparison
![Cumulative Reward-Regret](Cumulative%20Reward-Regret%20Comparison%20Plots.png)

Thompson Sampling converges to the optimal arm faster and accumulates less 
regret overall. The log-scale plot makes the early-stage exploration difference 
between the two algorithms more visible.

## Project Structure

```
├── Bandit.py                   # Main implementation
├── experiment_results.csv      # Logged results (Bandit, Reward, Algorithm)
├── requirements.txt
└── *.png                       # Output plots
```

## Setup

```bash
pip install -r requirements.txt
python Bandit.py
```

Running the script will print cumulative reward and regret for both algorithms, 
save results to `experiment_results.csv`, and display the comparison plots.

## Dependencies and their purpose

- `numpy` — experiment logic and reward sampling  
- `pandas` — results logging and CSV export  
- `matplotlib` — visualization  
- `loguru` — structured logging  

---

You can copy this directly into your `README.md`. The plot filenames match exactly what's already in your repo so the images will render automatically on GitHub.
