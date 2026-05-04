# Valura AI — Team Lead Assignment (Submission)

This document serves as the **single source of truth** for this submission. It includes setup, environment configuration, architecture, **design decisions and tradeoffs**, cost/latency evaluation, testing instructions, and the defence video link.

The file [`ASSIGNMENT.md`](ASSIGNMENT.md) contains the original project specification. All implementation reasoning intended for reviewers is documented here.

---

## 🚀 Overview

This project implements a production-style LLM system for financial query handling with:

- Deterministic safety guard (no LLM)
- Single-call intent classification
- Agent-based routing architecture
- Streaming responses via Server-Sent Events (SSE)

The system is designed for **reliability, low latency, and testability**, with full CI coverage and no external dependency required for tests.

---

## 🎥 Defence Video (Required)

Upload an **unlisted video (≤10 minutes)** within **24 hours** of your final push.

Cover:
- System architecture and request flow  
- One non-obvious decision and why  
- One improvement with more time  

**Defence video:**  
`https://www.youtube.com/watch?v=REPLACE_ME`

---

## 📦 Submission Structure

```text
README.md
src/
tests/
fixtures/
requirements.txt
.env.example
