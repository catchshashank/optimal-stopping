**Imports + global experiment constants**
1. What the code does
   - Imports core libraries: `transformers`, `torch`, `datasets`, `pandas`, `numpy`, `sklearn`, etc.
   - Defines key constants for the stopping problem:
     * `COST_PER_UNIT_TIME` (time penalty)
     * `BENEFIT_PER_POSITIVE_OUTCOME` (sale payoff)
     * `DECISION_OPPORTUNITIES = [45, 60]` seconds (the discrete stopping decision times)
   - Adds hardware-stability knobs:
     * `LOW_RAM_MODE`
     * `MAX_SEQ_LEN`
     * `batch sizes`
     * `grad accumulation`

