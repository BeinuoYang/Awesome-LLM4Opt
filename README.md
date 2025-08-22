# Large Language Model for Optimization Problem Modeling and Solving

[![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) <!-- PRs Welcome -->[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A curated list of **Large Language Models (LLMs)** for **optimization problem modeling** and **solving**, with awesome resources (papers, code, applications, reviews, surveys, etc.) across **transportation**, **logistics**, and related domains. This repository aims to systematically summarize recent advances in the field, providing a comprehensive overview of how LLMs solve complex optimization challenges.

## Understanding/Analysis of Optimization Problem

**[2025/02/09] Evaluating LLM Reasoning in the Operations Research Domain with ORQA, AAAI 2025.** [[paper]](https://arxiv.org/abs/2412.17874) [[dataset]](https://github.com/nl4opt/ORQA)

- The study introduces ORQA, a benchmark evaluating LLMs on Operations Research tasks requiring multistep reasoning. Testing models like LLaMA 3.1 and Mixtral reveals limited performance, highlighting LLMs' challenges in specialized domains.

**[2025/01/23] Decision information meets large language models: The future of explainable operations research, ICLR 2025.** [[paper]](https://openreview.net/forum?id=W2dR6rypBQ) [[official code]](https://github.com/Forrest-Stone/EOR?utm_source=catalyzex.com) [[dataset]](https://github.com/Forrest-Stone/EOR/tree/main/True-labels/benchmark)
Interpretability

- This study present EOR, a novel framework that addresses transparency and interpretability challenges in OR. It introduces the concept of “Decision Information” through what-if analysis and use bipartite graphs to quantify changes in OR models.
- Establish a industrial benchmark for evaluating explanation quality in OR.

**[2025/01/14] OptiChat: Bridging Optimization Models and Practitioners with Large Language Models, preprint.** [[paper]](https://arxiv.org/abs/2501.08406) [[official code]](https://github.com/li-group/OptiChat?tab=readme-ov-file)

- This study introduces OptiChat, a natural language dialogue system that empowers non-expert practitioners to interpret, analyze, and interact with complex optimization models by augmenting a large language model with specialized functional calls and code generation.

**[2024/05/17] Towards Human-aligned Evaluation for Linear Programming Word Problems, LREC-COLING 2024.** [[paper]](https://aclanthology.org/2024.lrec-main.1438/)

- This study introduces a novel metric based on graph edit distance to more accurately evaluate LLM-generated solutions for linear programming word problems (LPWPs) by correctly identifying mathematically equivalent answers.

**[2023/07/13] Large Language Models for Supply Chain Optimization, arXiv.** [[paper]](https://arxiv.org/abs/2307.03875) [[official code]](https://github.com/microsoft/OptiGuide)

- This study presents OptiGuide, a framework that leverages large language models (LLMs) to enable supply chain optimization with what-if analysis.

## Automated Modeling and Solving of Optimization Problem

### Prompt-based Methods

**[2025/06/30] LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach, preprint.** [[paper]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329027)

- This study introduces LEAN-LLM-OPT, a novel framework that uses few-shot learning to guide a team of LLM agents in automatically creating state-of-the-art formulations for large-scale optimization problems from a user's query.

**[2025/05/02] Autoformulation of Mathematical Optimization Models Using LLMs, ICML 2025.** [[paper]](https://openreview.net/forum?id=33YrT1j0O0&noteId=1PhN7tpMJd) [[official code]](https://github.com/jumpynitro/AutoFormulator)

- This study introduces a novel method that combines Large Language Models (LLMs) with Monte-Carlo Tree Search to automatically create optimization models from natural language, using techniques like symbolic pruning and LLM-guided evaluation to efficiently explore and generate correct formulations.

**[2025/01/23] DRoC: Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constraints, ICML 2025.** [[paper]](https://openreview.net/forum?id=s9zoyICZ4k) [[official code]](https://github.com/Summer142857/DRoC)

- This study proposes Decomposed Retrieval of Constraints (DRoC), a novel framework aimed at enhancing large language models (LLMs) in exploiting solvers to tackle vehicle routing problems (VRPs) with intricate constraints.

**[2024/05/02] OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models, ICML 2024.** [[paper]](https://arxiv.org/pdf/2402.10172) [[official code]](https://github.com/teshnizi/OptiMUS/tree/optimus-v0.2) [[dataset]](https://huggingface.co/datasets/udell-lab/NLP4LP)

- This study introduces OptiMUS, a Large Language Model (LLM)-based agent designed to formulate and solve (mixed integer) linear programming problems from their natural language descriptions. OptiMUS utilizes a modular structure to process problems, allowing it to handle problems with long descriptions and complex data without long prompts.

**[2024/01/16] Chain-of-Experts: When LLMs Meet Complex Operation Research Problems, ICLR 2024.** [[paper]](https://openreview.net/forum?id=HobyL1B9CZ) [[official code]](https://github.com/xzymustbexzy/Chain-of-Experts/tree/main) [[dataset]](https://github.com/xzymustbexzy/Chain-of-Experts/tree/main/dataset) [[poster]](https://iclr.cc/media/PosterPDFs/ICLR%202024/18977.png?t=1714228549.6135468)

- This study introduces Chain-of-Experts (CoE), a multi-agent LLM framework that boosts reasoning in complex operation research problems by integrating domain-specific agents under a conductor's guidance and reflection mechanism.

### Learning-based Methods

**[2025/07/15] Auto-Formulating Dynamic Programming Problems with Large Language Models, preprint.** [[paper]](https://arxiv.org/abs/2507.11737) [[official code]](https://github.com/Cardinal-Operations/ORLM)

- This study presents **DPLM**, a state-of-the-art specialized model for dynamic programming, which is enabled by the novel **DualReflect** synthetic data pipeline and validated on **DP-Bench**, the first comprehensive benchmark for this task.

**[2025/04/04] ORLM: A customizable framework in training large models for automated optimization modeling, Operations Research.** [[paper]](https://arxiv.org/abs/2405.17743) [[official code]](https://github.com/Cardinal-Operations/ORLM)

- This study proposes a pathway for training open-source LLMs to automate optimization modeling and solving. It introduces OR-INSTRUCT, a semi-automated data synthesis framework for optimization modeling that allows for customizable enhancements tailored to specific scenarios or model types. Additionally, the research presents IndustryOR, an industrial benchmark designed to evaluate the performance of LLMs in solving practical OR problems.

**[2025/03/03] LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch, ICLR 2025.** [[paper]](https://openreview.net/pdf?id=9OMvtboTJg) [[official code]](https://github.com/caigaojiang/LLMOPT?tab=readme-ov-file)

- LLMOPT introduces a unified framework that effectively bridges natural language optimization problems to mathematical formulations, significantly improving solving accuracy across diverse optimization types.

## Survey

**[2025/05] A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions, IJCAI 2025.** [[paper]](https://llm4or.github.io/LLM4OR/static/pdfs/LLM4OR_survey.pdf) [[official code]](https://github.com/LLM4OR/LLM4OR?tab=readme-ov-file)

**[2025/05/17] Why Do Multi-Agent LLM Systems Fail?, preprint.** [[paper]](https://arxiv.org/abs/2503.13657)

- Provide a systematic analysis for why do multiagent systems fail.

**[2025/01/10] Multi-Agent Collaboration Mechanisms: A Survey of LLMs, preprint.** [[paper]](https://arxiv.org/abs/2501.06322)

- This study explores collaboration in Multi-Agent Systems (MASs), proposing a framework categorizing actors, types, structures, strategies, and coordination protocols. It reviews methodologies to enhance MASs for real-world applications in 5G/6G, Industry 5.0, question answering, and social contexts, identifying key insights, challenges, and research directions toward artificial collective intelligence.

**[2024/02/11] Large Language Model based Multi-Agents: A Survey of Progress and Challenges, IJCAI 2024.** [[paper]](https://arxiv.org/abs/2402.01680) [[official code]](https://github.com/taichengguo/LLM_MultiAgents_Survey_Papers)

- This survey is presented to offer an in-depth discussion on the essential aspects and challenges of LLM-based multi-agent (LLM-MA) systems, and provides readers with an in-depth understanding of the domains and settings where LLM-MA systems operate or simulate; the profiling and communication methods of these agents; and the means by which these agents develop their skills.

## Other Resources

[Foundation Models for Combinatorial Optimization](https://github.com/ai4co/awesome-fm4co) |github repo|

- FM4CO contains interesting research papers (1) using Existing Large Language Models for Combinatorial Optimization, and (2) building Domain Foundation Models for Combinatorial Optimization.
