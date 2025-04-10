<p align="center">
  <!-- Optional: Replace with a real logo URL if you have one -->
  <!-- <img src="https://raw.githubusercontent.com/YourUsername/YourRepo/main/docs/machinedream_logo.png" alt="Machine Dream Engine Logo" width="150"/> -->
  <h1>⚙️ Machine Dream Engine (v1.1.1)</h1>
</p>

**Conceptual Origin:** Gregory L. Magnusson | **Code Implementation:** PYTHAI (Based on Concepts)
**Licenses:** MIT (Concept) / BSD v3 (Code Implementation)

---

## Sparking Insight: Introducing Machine Dream

`machinedream` **ignites** the exploration of internal knowledge refinement within AI, mirroring the enigmatic process of biological dreaming. This Python simulation models an engine dedicated to processing accumulated experiences ("data") during crucial "offline" phases. It distills raw information into abstract, **symbolic insights** ("aha moments"), actively combats information overload through intelligent memory management, and generates dynamic feedback signals for continuous, autonomous self-improvement.

> 🚀 **Explore the Unconscious:** This script acts as a conceptual sandbox, employing heuristics to simulate complex cognitive functions. Dive into internal state management, knowledge consolidation, and adaptive self-tuning for advanced AI architectures. Version **1.1.1** sharpens the focus by integrating simulated utility feedback and refining assessment metrics.

---

## Taming the Data Deluge: Why Machines Must Dream

Modern AI, especially architectures integrating **Agentic** actors (`OpenMind`) and **Augmentic** data enhancers, grapples with an exponential influx of information. Real-time interactions and synthetic data generation create a high-velocity, high-dimensional data tsunami.

Attempting to learn directly from this raw stream invites inefficiency and instability:

*   🐌 **Learning Crawls:** Signal extraction becomes a struggle amidst overwhelming noise.
*   🧠 **Knowledge Fades:** Past learnings vanish under the weight of new, potentially conflicting data (catastrophic forgetting).
*   🎯 **Context Lacks:** Systems overfit to specifics, failing to grasp generalizable, underlying principles.

**Machine Dreaming emerges as the vital internal consolidation & abstraction phase.** Like biological sleep, this simulated "offline" crucible empowers the system to:

*   ✨ **Aggregate & Filter:** Ingest data batches, implicitly focusing on significant patterns.
*   💡 **Synthesize Symbolic Insights:** Forge high-level, abstract "aha moments" – metaphorical truths beyond mere statistics.
*   📚 **Curate Memory:** Intelligently prune less vital memories based on age, calculated importance, and perceived utility, preventing cognitive overload.
*   🧭 **Generate Tuning Feedback:** Critically assess its own insight generation process to produce structured `tuning_data` for external **autotuning** systems, enabling the entire architecture to master the art of learning itself.

By transmuting raw experience into potent symbolic wisdom, Machine Dreaming cultivates more robust, efficient, and truly adaptive autonomous intelligence.

---

## Peeking Inside: Simulating the Dream Cycle

The engine pulses through a refined cycle (`run_dream_cycle`):

*   **Assess Current State:** Before incorporating new data, `assess_state` analyzes recent memory, computing metrics (importance, novelty, utility, diversity, variance), identifying parameter oscillation, and classifying the system's operational mode (e.g., "Exploring", "Stabilizing"). This yields structured `tuning_data`.
*   **Preprocess Input:** Conceptually refines incoming simulated data based on the engine's current `abstraction_level`.
*   **Aggregate Insight:** The core `_simulate_symbolic_aggregation` generates a `dream_result`, featuring a metaphorical `symbolic_insight` phrase and related `key_themes`, shaped by `dream_complexity` and `abstraction_level`.
*   **Score Insight:** Calculates `theme_novelty` relative to recent memory and assigns an `importance_score`, factoring in synthesis level, theme richness, novelty, and a chance for a simulated "breakthrough".
*   **Store Insight:** Commits the complete insight record (context, result, scores, parameters, utility placeholder) to the `self.memory`.
*   **Apply Tuning:** `apply_tuning` adjusts internal parameters (`dream_complexity`, `abstraction_level`, `tuning_level`) guided by the `tuning_data`, incorporating logic to dampen oscillation.
*   **Prune Memory:** `prune_memory` discards the lowest-value insights if `max_memory_size` is breached, using a combined score reflecting age-weighted importance and recorded utility.
*   **Persist State:** `save_memory` serializes the updated memory and engine parameters to JSON, ensuring continuity across sessions.

---

## Unveiling Capabilities (v1.1.1)

✅ **Symbolic Insight Generation:** Crafts abstract textual insights & themes via parameter-influenced templates.
📊 **Rich Metric Calculation:** Simulates Importance, Novelty, Synthesis Level, Theme Diversity, & externally-fed Utility, including averages & standard deviations.
🧭 **Normalized Assessment:** Delivers `tuning_data` with metrics normalized (-1 to +1) against target ranges and classifies the system's operational `system_mode`.
📈 **Detailed Tuning Suggestions:** Provides specific adjustment deltas & factors in `tuning_data` for precise external autotuner integration.
⛓️ **Oscillation Detection & Damping:** Identifies parameter flip-flopping and dynamically adjusts tuning step sizes to promote stability.
🗑️ **Utility-Aware Pruning:** Integrates simulated external feedback (`utility_score`) with age-weighted importance for intelligent memory curation.
🔧 **Deep Configurability:** Allows fine-tuning of initial state, memory limits, tuning behavior (steps, factors, penalties, damping), and metric targets via CLI / `__init__`.
🔄 **Dual Modes:** Supports interactive manual control and automated batch execution of dream cycles.
💾 **State Persistence:** Reliably saves and loads the complete engine state (memory, parameters, history) using JSON.

---

## Activating the Engine: Usage Guide

Execute `machinedream_v1.1.1.py` from your terminal.

**🚀 Automated Mode (Run N Cycles)**

```bash
# Example: Run 50 cycles with defaults
python3 machinedream_v1.1.1.py -n 50

# Example: Run 20 cycles, custom settings
python3 machinedream_v1.1.1.py -n 20 --max-memory 50 --init-complexity 7 --age-penalty 0.01

# Example: Custom targets via JSON string
python3 machinedream_v1.1.1.py -n 10 --target-ranges '{"utility": [0.5,0.9]}'
Use code with caution.
Markdown
▶️ Interactive Mode (Manual Control)
# Start interactive session
python3 machinedream_v1.1.1.py
Use code with caution.
Bash
Interactive Commands:
MachineDream> help  # (Conceptual - list commands below)
Use code with caution.
dream [id] [sz] [comp] : Trigger a dream cycle (args optional).
status : View current parameters, memory stats, metrics, mode, history.
insight [id] : Show latest high-importance insight or details for a specific ID.
assess : Manually run assessment & display full tuning_data.
tune_data : Display JSON data from the last assessment.
api [prompt] : Simulate external API call using latest insight context; records utility feedback.
set <param> <value> : Manually adjust complexity, abstraction, or tuning.
clear : Wipe memory & assessment history (requires confirmation).
exit : Quit the session.
Tailoring the Simulation: Configuration
Fine-tune the engine's behavior via:
Command Line Arguments:
-n/--num_loops: Automated cycles (0 = interactive).
--max-memory: Pruning threshold.
--memory-file: State file path.
--init-*: Override initial complexity, abstraction, tuning.
--abs-step: Base step for abstraction tuning.
--age-penalty: Factor for age weighting in pruning.
--damping-factor: Strength of step-size reduction during oscillation.
--target-ranges: JSON defining ideal metric ranges (e.g., '{"importance": [0.5,0.9]}').
--svg: Generate optional memory hash visualization on exit.
Internal Constants: Modify script constants like MIN_MEMORY_FOR_ASSESSMENT, RECOMMENDATION_HISTORY_LENGTH, UTILITY_WEIGHT_IN_PRUNING, parameter bounds (MIN/MAX_*).
.env File: Optionally use a .env file for configuration variables (requires python-dotenv).
Bridging Worlds: Integration Points
Connect machinedream to a larger system:
Ingest Data: Feed input_data_metadata dictionaries into run_dream_cycle to trigger refinement based on external events or data batches.
Export Tuning Data: Consume the structured JSON from assess_state (via _last_assessment_data or tune_data command) in an external autotuning system. Leverage normalized metrics, system mode, and suggested adjustments.
Share Insights: Use get_latest_insight to retrieve symbolic context (symbolic_insight, key_themes) for other AI modules (e.g., injecting context into prompts for LLMs like Gemini via api.py).
Import Utility Feedback: Call record_insight_utility(dream_id, score) from external components after using an insight, providing a 0-1 score reflecting its perceived usefulness, influencing subsequent pruning.
Grounding Expectations: Acknowledging Limits
❗ Crucially, this is a Simulation: The engine employs abstract heuristics and randomness. It does not perform real data analysis, semantic understanding, or AI training. Insights, metrics, and tuning effects are simulated constructs.
Heuristic Dependency: Behavior hinges on simulation rules, thresholds, weights, and target ranges lacking empirical validation.
Insight Validity Unverified: Generated insights aren't checked for correctness or real-world meaning. "Breakthroughs" are simulated chance events.
Partial Utility Loop: Utility feedback influences pruning but doesn't yet adapt the insight generation or assessment logic.
Conceptual Integration: Links to external systems are placeholders demonstrating potential, not implemented connections.
Scalability Constraints: Single-instance, file-based design limits use in large-scale or distributed systems.
Legal Notices
Concept: MIT License (Gregory L. Magnusson 2024)
Code Implementation: BSD 3-Clause License (PYTHAI 2024)
