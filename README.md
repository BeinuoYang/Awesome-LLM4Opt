# Large Language Model for Optimization Problem Modeling and Solving

[![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) <!-- PRs Welcome -->[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A curated list of **Large Language Models (LLMs)** for **optimization problem modeling** and **solving**, with awesome resources (papers, code, applications, reviews, surveys, etc.) across **transportation**, **logistics**, and related domains. This repository aims to systematically summarize recent advances in the field, providing a comprehensive overview of how LLMs solve complex optimization challenges.

## LLM for Optimization Problem Modeling and Solving

### Prompt-based Methods

**[2025/02/09] Evaluating LLM Reasoning in the Operations Research Domain with ORQA, AAAI 2025.** [[paper]](https://arxiv.org/abs/2412.17874) [[dataset]](https://github.com/nl4opt/ORQA)

- The study introduces ORQA, a benchmark evaluating LLMs on Operations Research tasks requiring multistep reasoning. Testing models like LLaMA 3.1 and Mixtral reveals limited performance, highlighting LLMs' challenges in specialized domains.

**[2025/01/23] Decision information meets large language models: The future of explainable opera- tions research, ICLR 2025.** [[paper]](https://openreview.net/forum?id=W2dR6rypBQ) [[official code]](https://github.com/Forrest-Stone/EOR?utm_source=catalyzex.com) [[dataset]](https://github.com/Forrest-Stone/EOR/tree/main/True-labels/benchmark)
Interpretability

- The study present EOR, a novel framework that addresses transparency and interpretability challenges in OR. It introduces the concept of “Decision Information” through what-if analysis and use bipartite graphs to quantify changes in OR models.
- Establish a industrial benchmark for evaluating explanation quality in OR.

**[2024/03/15] Chain-of-Experts: When LLMs Meet Complex Operation Research Problems, ICLR 2024.** [[paper]](https://openreview.net/forum?id=HobyL1B9CZ) [[official code]](https://github.com/xzymustbexzy/Chain-of-Experts/tree/main) [[poster]](https://iclr.cc/media/PosterPDFs/ICLR%202024/18977.png?t=1714228549.6135468)

- This paper introduces Chain-of-Experts (CoE), a multi-agent LLM framework that boosts reasoning in complex operation research problems by integrating domain-specific agents under a conductor's guidance and reflection mechanism.

**[2023/07/13] Large Language Models for Supply Chain Optimization, arXiv.** [[paper]](https://arxiv.org/abs/2307.03875) [[official code]](https://github.com/microsoft/OptiGuide)

- This study esigned and implement OptiGuide, a framework that employs LLMs to interpret supply chain optimization solutions.

### Learning-based Methods

**[2025/04/04] ORLM: A customizable framework in training large models for automated optimization modeling, Operations Research.** [[paper]](https://arxiv.org/abs/2405.17743) [[official code]](https://github.com/Cardinal-Operations/ORLM)

- This study proposes a pathway for training open-source LLMs to automate optimization modeling and solving. It introduces OR-INSTRUCT, a semi-automated data synthesis framework for optimization modeling that allows for customizable enhancements tailored to specific scenarios or model types. Additionally, the research presents IndustryOR, an industrial benchmark designed to evaluate the performance of LLMs in solving practical OR problems.

**[2025/03/03] LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch, ICLR 2025.** [[paper]](https://openreview.net/pdf?id=9OMvtboTJg) [[official code]](https://github.com/caigaojiang/LLMOPT?tab=readme-ov-file)

- LLMOPT introduces a unified framework that effectively bridges natural language optimization problems to mathematical formulations, significantly improving solving accuracy across diverse optimization types.

## Multi-Agents

**[2024/06/21] ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs, ACL 2024.** [[paper]](https://arxiv.org/abs/2309.13007) [[official code]](https://github.com/dinobby/ReConcile)

- Improving reasoning by learning to convince other agents in a multi-round discussion among divers LLMs.

**[2024/02/28] Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?, ACL 2024.** [[paper]](https://aclanthology.org/2024.acl-long.331/) [[official code]](https://github.com/HKUST-KnowComp/LLM-discussion)

- This study finds that a single-agent LLM equipped with strong prompts can achieve performance nearly equivalent to the best existing multi-agent discussion methods across various reasoning tasks and backbone models. Multi-agent discussion only demonstrated superior performance over a single agent when prompts lacked demonstrations. The study further reveals common interaction mechanisms among LLMs during the discussion process.

### Survey

**[2025/05/17] Why Do Multi-Agent LLM Systems Fail?, preprint.** [[paper]](https://arxiv.org/abs/2503.13657)

- Provide a systematic analysis for why do multiagent systems fail.

**[2025/01/10] Multi-Agent Collaboration Mechanisms: A Survey of LLMs, preprint.** [[paper]](https://arxiv.org/abs/2501.06322)

- This study explores collaboration in Multi-Agent Systems (MASs), proposing a framework categorizing actors, types, structures, strategies, and coordination protocols. It reviews methodologies to enhance MASs for real-world applications in 5G/6G, Industry 5.0, question answering, and social contexts, identifying key insights, challenges, and research directions toward artificial collective intelligence.

**[2024/02/11] Large Language Model based Multi-Agents: A Survey of Progress and Challenges, IJCAI 2024.** [[paper]](https://arxiv.org/abs/2402.01680) [[official code]](https://github.com/taichengguo/LLM_MultiAgents_Survey_Papers)

- This survey is presented to offer an in-depth discussion on the essential aspects and challenges of LLM-based multi-agent (LLM-MA) systems, and provides readers with an in-depth understanding of the domains and settings where LLM-MA systems operate or simulate; the profiling and communication methods of these agents; and the means by which these agents develop their skills.

### Framework

**[2024/02/11] AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation, ICLR 2024.** [[paper]](https://openreview.net/forum?id=tEAF9LBdgu) [[official code]](https://github.com/ag2ai/ag2)

- AutoGen is an open-source framework that allows developers to build LLM applications using multiple conversable agents and conversation programming. Experiments show benefits on diverse domains.

**[2023/10/13] MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework, ICLR 2024.** [[paper]](https://openreview.net/forum?id=VtmBAGCN7o) [[official code]](https://github.com/geekan/MetaGPT)

- MetaGPT introduces a framework integrating human workflows into multi-agent systems using LLMs. It addresses logic inconsistencies in complex tasks by encoding Standardized Operating Procedures (SOPs) and assigning specialized roles. Evaluations show it outperforms chat-based systems in collaborative problem-solving.

## Other Resources

[Foundation Models for Combinatorial Optimization](https://github.com/ai4co/awesome-fm4co) |github repo|

- FM4CO contains interesting research papers (1) using Existing Large Language Models for Combinatorial Optimization, and (2) building Domain Foundation Models for Combinatorial Optimization.

[ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners)
