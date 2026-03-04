## **Fundamentals**
1) *Discrete-time*: Instead of time flowing continuously, we only look at specific checkpoints.
    - Most financial models assume continuous time (prices move every instant).
    - But neural networks work more naturally with discrete steps, so the problem is converted into a grid of decision points.
2) *Finite horizon*: There is a final deadline. Eg., You must stop by time T = 3 years. So if you haven't stopped earlier, you are forced to stop at the last step.
3) *Model-free setting*:
   - One of the most important ideas in the paper [Venkata & Bhattacharya, 2023](https://dl.acm.org/doi/10.5555/3666122.3666658).
   - We do not assume a mathematical formula describing how the prices evolve.
   - Hence, we don't know the true data-generating process and only have sample trajectories.
   - The algorithm learns from these **observed trajectories**, not from a known model.
4) Contrast with *classical approaches*:
   - *Known dynamics*: Traditional finance assumes the exact formula for price movement.
   - *Markov chain*: Future only depends on the current state, not the entire past.
5) **Why the paper avoids this assumption**
   - Markov assumption may fail - there could be path-dependent payoffs
   - Dynamics may be unknown
   - State at (t + 1) may depend on history
   So the paper proposes a **model-free learning** method.
6) Continuous-time problems can be discretized - standard in numerical methods [(Kloeden & Platen, 1992)](https://link.springer.com/book/10.1007/978-3-662-12616-5)
7) However, solving non-Markovian problems is challenging because:
   -  *Curse of dimensionality*
       - Transforming a non-Markovian process to a Markovian process would involve expanding the *state-space* substantially with process history.
       - However, this expansion of *state-space* would make the approximation of the value function highly complex due to high-dimensionality. 
