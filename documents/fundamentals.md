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
   - Known dynamics: Traditional finance assumes the exact formula for price movement.
   - Markov chain: Future only depends on the current state, not the entire past.
