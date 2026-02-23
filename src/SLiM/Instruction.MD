# Instruction
<li> Search online paper about Hallucination Reduction in LLMs
<li> Since we need to adapt SLiM from the original purpose to a new one, which is Hallucination Reduction, we need to think about how to adapt the original SLiM to our new purpose. We need to think about what kind of data we need to train the SLiM module (we already have as dataset BeliefBankFacts, BeliefBankConstraint and HaluEval), and how to evaluate the performance of the SLiM module in terms of Hallucination Reduction. (src/truthx could be a good place to start, but do not limit ourselves to it, we can search for other metodology)
<li> In src/truthx/train_truthx we have used same pairs for training -> we need to use the same pairs
