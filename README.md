# Large Language Model for Optimization Problem Modeling and Solving

[![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) <!-- PRs Welcome -->[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

The fusion of **Large Language Models (LLMs)** and **Operations Research (OR)** is transforming how optimization problems are understood, modeled, and solved. This repository provides a curated collection of cutting-edge research that showcases this evolution.

We track the latest papers, code, and resources demonstrating how LLMs are used to:

- **Interpret and Analyze**: Make complex optimization models more understandable and interactive.
- **Automate Formulation**: Translate natural language descriptions directly into solvable mathematical models.
- **Guide Solution Search**: Enhance the performance of solvers by generating heuristics, cuts, or strategies.

## <span style="color: #155ac0ff;">Model Interpretation and Analysis</span>

**[2025/06] EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations, *ICML 2025 Workshop*.** [[paper]](https://arxiv.org/abs/2502.14760) [[dataset]](https://huggingface.co/datasets/humainlab/EquivaFormulation) [[official code]](https://github.com/HumainLab/EquivaMap)

- This study introduces EquivaMap, an LLM-driven framework that leverages a formal definition called "Quasi-Karp Equivalence" to automatically verify the equivalence of optimization problem formulations. EquivaMap achieved 100% accuracy across diverse equivalent transformations, significantly outperforming existing heuristic methods on the newly developed EquivaFormulation dataset.

**[2025/02] Evaluating LLM Reasoning in the Operations Research Domain with ORQA, *AAAI 2025*.** [[paper]](https://arxiv.org/abs/2412.17874) [[dataset]](https://github.com/nl4opt/ORQA)

- The study introduces ORQA, a benchmark evaluating LLMs on Operations Research tasks requiring multistep reasoning. Testing models like LLaMA 3.1 and Mixtral reveals limited performance, highlighting LLMs' challenges in specialized domains.

**[2025/01] Decision information meets large language models: The future of explainable operations research, *ICLR 2025*.** [[paper]](https://openreview.net/forum?id=W2dR6rypBQ) [[official code]](https://github.com/Forrest-Stone/EOR?utm_source=catalyzex.com) [[dataset]](https://github.com/Forrest-Stone/EOR/tree/main/True-labels/benchmark)

- This study present EOR, a novel framework that addresses transparency and interpretability challenges in OR. It introduces the concept of “Decision Information” through what-if analysis and use bipartite graphs to quantify changes in OR models.
- Establish a industrial benchmark for evaluating explanation quality in OR.

**[2025/01] OptiChat: Bridging Optimization Models and Practitioners with Large Language Models, *preprint*.** [[paper]](https://arxiv.org/abs/2501.08406) [[official code]](https://github.com/li-group/OptiChat?tab=readme-ov-file)

- This study introduces OptiChat, a natural language dialogue system that empowers non-expert practitioners to interpret, analyze, and interact with complex optimization models by augmenting a large language model with specialized functional calls and code generation.

**[2024/05] Towards Human-aligned Evaluation for Linear Programming Word Problems, *LREC-COLING 2024*.** [[paper]](https://aclanthology.org/2024.lrec-main.1438/)

- This study introduces a novel metric based on graph edit distance to more accurately evaluate LLM-generated solutions for linear programming word problems (LPWPs) by correctly identifying mathematically equivalent answers.

**[2023/07] Large Language Models for Supply Chain Optimization, *preprint*.** [[paper]](https://arxiv.org/abs/2307.03875) [[official code]](https://github.com/microsoft/OptiGuide)

- This study presents OptiGuide, a framework that leverages large language models (LLMs) to enable supply chain optimization with what-if analysis.

## <span style="color: #155ac0ff;">Automated Optimization Modeling</span>

### Prompt-based Methods

**[2025/08] Guiding Large Language Models in Modeling Optimization Problems via Question Partitioning, *IJCAI 2025*.** [[paper]](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/5589.pdf)

- This study proposes PaMOP, a new framework that uses LLMs to automatically create mathematical models for optimization problems from natural language descriptions. The framework breaks down large problems into smaller parts using a tree structure, guiding the LLM to model each part separately with self-augmented prompts, and then iteratively corrects the final model.

**[2025/07] BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving, *ACL 2025*.** [[paper]](https://aclanthology.org/2025.acl-long.40/) [[official code]](https://github.com/LLM4OR/StructuredOR) [[dataset]](https://huggingface.co/datasets/LLM4OR/StructuredOR)

- This study proposes a new algorithm, BPP-Search, designed to enhance the ability to solve mathematical modeling problems by improving the Tree-of-Thought (ToT) framework. The paper first introduces a new dataset, StructuredOR, which provides detailed annotations of the modeling process.

**[2025/07] ORMind: A Cognitive-Inspired End-to-End Reasoning Framework for Operations Research, *ACL 2025*.** [[paper]](https://aclanthology.org/2025.acl-industry.10/)

- This study introduces ORMind, a cognitive-inspired end-to-end framework that enhances optimization through counterfactual reasoning.

**[2025/06] LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach, *preprint*.** [[paper]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329027)

- This study introduces LEAN-LLM-OPT, a novel framework that uses few-shot learning to guide a team of LLM agents in automatically creating state-of-the-art formulations for large-scale optimization problems from a user's query.

**[2025/05] Autoformulation of Mathematical Optimization Models Using LLMs, *ICML 2025*.** [[paper]](https://openreview.net/forum?id=33YrT1j0O0&noteId=1PhN7tpMJd) [[official code]](https://github.com/jumpynitro/AutoFormulator)

- This study introduces a novel method that combines Large Language Models (LLMs) with Monte-Carlo Tree Search to automatically create optimization models from natural language, using techniques like symbolic pruning and LLM-guided evaluation to efficiently explore and generate correct formulations.

**[2025/01] DRoC: Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constraints, *ICML 2025*.** [[paper]](https://openreview.net/forum?id=s9zoyICZ4k) [[official code]](https://github.com/Summer142857/DRoC)

- This study proposes Decomposed Retrieval of Constraints (DRoC), a novel framework aimed at enhancing large language models (LLMs) in exploiting solvers to tackle vehicle routing problems (VRPs) with intricate constraints.

**[2024/10] CAFA: Coding as Auto-Formulation Can Boost Large Language Models in Solving Linear Programming Problem, *NeurIPS 2024 Workshop MATH-AI*.** [[paper]](https://openreview.net/forum?id=xC2xtBLmri) [[official code]](https://github.com/BlueAsuka/CAFA)

- This study introduces CAFA, a compact prompt guiding the LLMs to formalize the given problem text into lines of code.

**[2024/05] OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models, *ICML 2024*.** [[paper]](https://arxiv.org/pdf/2402.10172) [[official code]](https://github.com/teshnizi/OptiMUS/tree/optimus-v0.2) [[dataset]](https://huggingface.co/datasets/udell-lab/NLP4LP)

- This study introduces OptiMUS, a Large Language Model (LLM)-based agent designed to formulate and solve (mixed integer) linear programming problems from their natural language descriptions. OptiMUS utilizes a modular structure to process problems, allowing it to handle problems with long descriptions and complex data without long prompts.

**[2024/01] Chain-of-Experts: When LLMs Meet Complex Operation Research Problems, *ICLR 2024*.** [[paper]](https://openreview.net/forum?id=HobyL1B9CZ) [[official code]](https://github.com/xzymustbexzy/Chain-of-Experts/tree/main) [[dataset]](https://github.com/xzymustbexzy/Chain-of-Experts/tree/main/dataset) [[poster]](https://iclr.cc/media/PosterPDFs/ICLR%202024/18977.png?t=1714228549.6135468)

- This study introduces Chain-of-Experts (CoE), a multi-agent LLM framework that boosts reasoning in complex operation research problems by integrating domain-specific agents under a conductor's guidance and reflection mechanism.

### Learning-based Methods

**[2025/07] Step-Opt: Boosting Optimization Modeling in LLMs through Iterative
Data Synthesis and Structured Validation, *preprint*.** [[paper]](https://arxiv.org/abs/2506.17637) [[official code]](https://github.com/samwu-learn/Step)

- This study introduces the Step-Opt framework, which enhances LLMs’ ability to generate accurate mathematical optimization models from natural language. By combining an iterative data synthesis process with a rigorous stepwise validation mechanism, Step-Opt enables fine-tuned LLMs to achieve SOTA performance on various benchmarks — especially for complex problem instances.

**[2025/07] Auto-Formulating Dynamic Programming Problems with Large Language Models, *preprint*.** [[paper]](https://arxiv.org/abs/2507.11737) [[official code]](https://github.com/Cardinal-Operations/ORLM)

- This study presents **DPLM**, a state-of-the-art specialized model for dynamic programming, which is enabled by the novel **DualReflect** synthetic data pipeline and validated on **DP-Bench**, the first comprehensive benchmark for this task.

**[2025/05] Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling, *preprint*.** [[paper]](https://arxiv.org/abs/2505.11792) [[official code]](https://github.com/Cardinal-Operations/SIRL)

- This study introduces Solver-Informed Reinforcement Learning (SIRL), a framework that enhances LLM accuracy in mathematical optimization modeling by using professional solvers (like Gurobi) as verifiers to provide high-quality reward signals. This enables LLMs to generate syntactically correct, mathematically feasible optimization code, effectively addressing hallucination and error issues in this domain.

**[2025/05] ORLM: A customizable framework in training large models for automated optimization modeling, *Operations Research*.** [[paper]](https://arxiv.org/abs/2405.17743) [[official code]](https://github.com/Cardinal-Operations/ORLM)

- This study proposes a pathway for training open-source LLMs to automate optimization modeling and solving. It introduces OR-INSTRUCT, a semi-automated data synthesis framework for optimization modeling that allows for customizable enhancements tailored to specific scenarios or model types. Additionally, the research presents IndustryOR, an industrial benchmark designed to evaluate the performance of LLMs in solving practical OR problems.

**[2025/03] LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch, *ICLR 2025*.** [[paper]](https://openreview.net/pdf?id=9OMvtboTJg) [[official code]](https://github.com/caigaojiang/LLMOPT?tab=readme-ov-file)

- LLMOPT introduces a unified framework that effectively bridges natural language optimization problems to mathematical formulations, significantly improving solving accuracy across diverse optimization types.

## <span style="color: #155ac0ff;">LLM-Guided Solution Search</span>

**[2025/08] EvoCut: Strengthening Integer Programs via Evolution-Guided Language Models, *preprint*.** [[paper]](https://arxiv.org/abs/2508.11850) [[official code]](https://github.com/milad1378yz/EvoCut)

- This study introduces EvoCut, a novel framework that accelerates Mixed-Integer Linear Programming (MILP) by injecting problem-specific cutting planes into the LP relaxation. These cuts reduce the feasible set of the LP relaxation and improve solver efficiency.

**[2025/07] Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms, *preprint*.** [[paper]](https://arxiv.org/abs/2503.10968) [[official code]](https://github.com/camilochs/comb-opt-for-all)

- This study shows that LLMs can provide effective suggestions to enhance existing, complete combinatorial optimization algorithms, enabling non-expert programmers to improve both solution quality and computational efficiency of 10 different algorithms for the Traveling Salesman Problem (TSP).

**[2025/05] Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning, *ICLR 2025 Workshop*.** [[paper]](https://claire-labo.github.io/EvoTune/) [[official code]](https://github.com/CLAIRE-Labo/EvoTune)

- This study introduces EvoTune, a method that continuously refines LLMs' generative policies through RL fine-tuning within an evolutionary search loop for algorithm discovery. This approach consistently accelerates the discovery of superior algorithms and enhances search space exploration across various combinatorial optimization and symbolic regression tasks, often outperforming static LLM-based baselines and, in some cases, human experts.

## Survey

**[2025/05] A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions, *IJCAI 2025*.** [[paper]](https://llm4or.github.io/LLM4OR/static/pdfs/LLM4OR_survey.pdf) [[official code]](https://github.com/LLM4OR/LLM4OR?tab=readme-ov-file)

## Vision Paper

**[2025/08] Synergizing Artificial Intelligence and Operations Research: Perspectives from INFORMS Fellows on the Next Frontier, *INFORMS Journal on Data Science*.** [[paper]](https://pubsonline.informs.org/doi/full/10.1287/ijds.2025.0077)

## Other Resources

[Foundation Models for Combinatorial Optimization](https://github.com/ai4co/awesome-fm4co)

- FM4CO contains interesting research papers (1) using Existing Large Language Models for Combinatorial Optimization, and (2) building Domain Foundation Models for Combinatorial Optimization.

[Awesome Multi-Agent Papers](https://github.com/kyegomez/awesome-multi-agent-papers)

- A compilation of the best multi-agent papers by the Swarms Team.
  
[timefold](https://timefold.ai/)

- The open source Solver AI for Java and Kotlin to optimize scheduling and routing with [good explainability](https://docs.timefold.ai/timefold-platform/latest/guides/validating-an-optimized-plan-with-explainable-ai#:~:text=Timefold%20includes%20several%20features%20designed%20for%20explainability%20and,spot-check%20the%20result%20by%20visualizing%20each%20computed%20plan.).
