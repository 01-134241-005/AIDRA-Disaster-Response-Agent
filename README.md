# 🚑 AIDRA: Adaptive Intelligent Disaster Response Agent

A fully integrated hybrid AI system for disaster victim rescue using search algorithms, constraint satisfaction, machine learning, fuzzy logic, and dynamic replanning.

---

## 📌 Overview

AIDRA coordinates ambulances in a 5×5 grid disaster environment. It integrates:

- **5 search algorithms** – BFS, DFS, Greedy, A* (with risk penalty), Hill Climbing
- **CSP** – MRV heuristic, forward checking, backtrack counting (0 backtracks achieved)
- **Machine Learning** – Custom KNN (94.2% accuracy) and Naïve Bayes (86.7% accuracy) for severity prediction
- **Fuzzy Logic** – Mamdani inference for dynamic priority (risk + time → priority)
- **Dynamic Replanning** – 50% chance of random road blockage, automatic A* rerouting
- **Performance Metrics** – Victims saved, avg time, risk exposure, optimality ratio, resource utilization + 3 graphs

The system successfully rescues **5/5 victims** (2 Critical, 2 Moderate, 1 Minor) with 2 ambulances (capacity 3 each) under dynamic blockages.

---

## 🚀 Features

| Module | Techniques & Performance |
|--------|--------------------------|
| Search | BFS, DFS, Greedy, A*, Hill Climbing |
| CSP | MRV, forward checking, backtrack count = 0 |
| ML | KNN (k=5, distance-weighted), Naïve Bayes (Gaussian) |
| Fuzzy | Mamdini, 4 rules, centroid defuzzification, priority ∈ [3.3–3.7] |
| Replanning | On‑the‑fly A* after random blockages (50% per victim) |
| Metrics | 3 graphs: algorithm comparison, time‑risk trade‑off, optimality ratio |

---

## 🧠 System Architecture



*Data flow: Environment → ML → Fuzzy → CSP → Search → Decision → Replanning → Metrics.*

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/AIDRA-Disaster-Response-Agent.git
   cd AIDRA-Disaster-Response-Agent
