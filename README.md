# Large Language Model for Optimization Problem Modeling and Solving

[![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) <!-- PRs Welcome -->[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A curated list of **Large Language Models (LLMs)** for **optimization problem modeling** and **solving**, with awesome resources (papers, code, applications, reviews, surveys, etc.) across **transportation**, **logistics**, and related domains. This repository aims to systematically summarize recent advances in the field, providing a comprehensive overview of how LLMs solve complex optimization challenges.

## LLM for Optimization Problem Modeling and Solving

### Prompt-based Methods

**Evaluating LLM Reasoning in the Operations Research Domain with ORQA, AAAI 2025.** [[paper]](https://arxiv.org/abs/2412.17874)[[dataset]](https://github.com/nl4opt/ORQA)

- The study introduces ORQA, a benchmark evaluating LLMs on Operations Research tasks requiring multistep reasoning. Testing models like LLaMA 3.1 and Mixtral reveals limited performance, highlighting LLMs' challenges in specialized domains.

**Chain-of-Experts: When LLMs Meet Complex Operation Research Problems, ICLR 2024.** [[paper]](https://openreview.net/forum?id=HobyL1B9CZ)[[official code]](https://github.com/xzymustbexzy/Chain-of-Experts/tree/main)[[poster]](https://iclr.cc/media/PosterPDFs/ICLR%202024/18977.png?t=1714228549.6135468)

- This paper introduces Chain-of-Experts (CoE), a multi-agent LLM framework that boosts reasoning in complex operation research problems by integrating domain-specific agents under a conductor's guidance and reflection mechanism.

### Learning-based Methods

**LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch, ICLR 2025.** [[paper]](https://openreview.net/pdf?id=9OMvtboTJg)[[official code]](https://github.com/caigaojiang/LLMOPT?tab=readme-ov-file)

- LLMOPT introduces a unified framework that effectively bridges natural language optimization problems to mathematical formulations, significantly improving solving accuracy across diverse optimization types.

## Multi-Agent Collaborative frameworks

**Multi-Agent Collaboration Mechanisms: A Survey of LLMs, preprint.** [[paper]](https://arxiv.org/abs/2501.06322)

- This study explores collaboration in Multi-Agent Systems (MASs), proposing a framework categorizing actors, types, structures, strategies, and coordination protocols. It reviews methodologies to enhance MASs for real-world applications in 5G/6G, Industry 5.0, question answering, and social contexts, identifying key insights, challenges, and research directions toward artificial collective intelligence.

**MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework, ICLR 2024.** [[paper]](https://openreview.net/forum?id=VtmBAGCN7o)[[official code]](https://github.com/geekan/MetaGPT)

- MetaGPT introduces a framework integrating human workflows into multi-agent systems using LLMs. It addresses logic inconsistencies in complex tasks by encoding Standardized Operating Procedures (SOPs) and assigning specialized roles. Evaluations show it outperforms chat-based systems in collaborative problem-solving.

## Other Resources
