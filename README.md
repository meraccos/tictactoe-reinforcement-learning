# Tabular Reinforcement Learning applied on Tic Tac Toe

Applying valute iteration and MDP to to teach a reinforcement learning agent playing tic-tac-toe. 
The code is written in Python from scratch, and the policy is near-optimal.

The memory folder contains initialized and re-evaluated state-value pairs. (load using Pickle)

![image](https://swdevnotes.com/images/swift/2021/1024/X-to-move-tic-tac-toe.png)

## Guidelines:

**Step 1.** Extract all the possible states and initialize their values:
```Python
python3 state_extractor.py
```

**Step 2.** Run value iteration over all the states until convergence:

```Python
python3 value_iterator.py
```

**Step 3.** Several ways to check the policy

- Method I: AI plays against a randomly playing agent:
```Python
python3 markov_eval.py
```

- Method II: AI plays against itself:
```Python
python3 against_itself.py
```

- Method III: AI plays against human:
```Python
python3 human.py
```
