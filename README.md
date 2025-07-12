Prompt engineering is noisy, manual, and hard to scale.
Why MIPRO?

According to the paper, it is important to optimize both the prompt instructions and the few-shot examples because they interact in complex ways that significantly affect model performance. Instructions frame the task, while examples guide the model’s reasoning, and optimizing both allows the system to align more precisely with the desired output behavior. This is especially crucial in multi-stage programs, where one module’s output becomes the next module’s input. Optimized examples can correct for model bias, reinforce correct reasoning patterns, and ensure consistency across the pipeline. Empirical results in the paper show that jointly tuning instructions and examples leads to better outcomes than optimizing either component alone.


1. Bootstrapping Demonstrations
The framework begins by generating an initial pool of candidate few-shot demonstrations through bootstrapping, which is essentially just running the original prompt on some of the training set and generating input output traces. If the final output meets or exceeds a predefined performance threshold (based on a task-level metric), the trace is considered successful and retained. By extracting intermediate input-output pairs from these traces, we get naturally aligned demonstrations — grounded in the actual flow and logic of the program. From these successful traces, input-output pairs are extracted, which are stored as candidate demonstrations for future optimization.
We also perform grounding to enrich the context for instruction generation. This step collects structured auxiliary data to help the instruction-proposing language model generate higher-quality, task-aware instructions. The grounding information includes:
* A dataset summary, which describes the nature and structure of the input-output space.
* A set of evaluated module-level traces, including past inputs, outputs, and their performance scores. (These are continually generated)
These elements are assembled as structured inputs to a prompt-generation program that will later be used to propose new instruction candidates.

2. Instruction Proposal and Prompt Configuration Search
With the pool of demonstrations and grounding data established, we then propose candidate instruction strings. I set this Proposal model to a base temperature of 0.7 with a random bit of increase to introduce as much variety as possible in the prompt candidates.
Given the combinatorial size of the search space—spanning instructions and demonstration sets across all modules—MIPRO uses Bayesian optimization to efficiently guide the search toward high-performing prompt configurations.
Specifically, I implemented the Tree-structured Parzen Estimator (TPE), a probabilistic model that models the performance landscape over prompt parameterizations:
* TPE maintains two distributions:
    * One for configurations that have yielded high scores (good region),
    * One for configurations that have yielded lower scores (bad region).
* During each iteration, TPE samples new instruction–demo combinations that are likely to fall in or near the high-performance region, while still exploring uncertain areas of the space to avoid local minima.
* After each trial, the TPE model is updated with the newly observed configuration and its associated performance score.

The task model that actually performs the task we are trying to optimise for is kept at a low temperature, around 0.2, because the decision should be close to deterministic.

3. Mini-Batch Evaluation
To reduce the computational cost of evaluating each prompt configuration, when there’s enough data (there wasn’t any point in minibatching when I ran on the ranked data), I mini-batch evaluation rather than scoring on the entire training set.
* At each iteration, a small batch of input examples are sampled from the training set.
* The average task metric across the batch is computed and used as the configuration’s score.
* This score is then passed to the TPE model to refine its internal estimate of the prompt space, that way we can prune the prompt space a bit. 

4. Iterative Optimization Loop
MIPRO continuously iterates through the following steps:
1. Propose a new instruction and demonstration configuration using TPE.
2. Evaluate it on a mini-batch of examples.
3. Update the TPE model with the new score.
4. Periodically (e.g., every N iterations), evaluate top-scoring configurations on the full training set to validate their generalization.

5. Final Selection and Output
Once all optimization trials are complete, MIPRO selects the highest-performing prompt configuration—so the the set of instructions and demonstrations across that achieved the best average score on the full training set and this configuration is returned as the final, optimized version of the LM program.


DIAGRAM

LLMs often guess or hallucinate when asked to answer complex questions directly.
CoT helps break the problem into smaller parts, increasing interpretability and correctness and in terms of the trace when you have a binary answer like A or B, having that in the trace could really help future iterations learn the required behaviour. 


traditional prompt engineering relies on fragile string concatenation and manual edits. DSPy replaces that with:
* Structured modules defined by clear input/output signatures.
* Declarative prompt templates where variables (like instructions or examples) can be programmatically manipulated and optimized.
Why this helps for optimization:
* Prompts become parameterized objects rather than opaque strings.
* You can systematically define what to optimize (e.g., instruction wording, few-shot examples).
* Optimization algorithms like MIPRO or OPRO can operate over structured variables instead of raw text.

ABLATION study 


