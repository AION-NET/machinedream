# Machine Dream Engine - Technical Documentation (v1.1.1)

## Overview

**Version:** v1.1.1

`machinedream` simulates an internal knowledge refinement engine inspired by the concept of dreaming
It models how a system might process accumulated experiences ("data") during an "offline" phase to extract abstract, symbolic insights ("aha moments"), assess the quality and novelty of these insights using internal metrics, generate structured tuning data based on this assessment, apply self-tuning adjustments to its own parameters, allow manual control, simulate insight utility feedback influencing pruning, and manage a memory of past dreams using utility/age-weighted importance pruning
The core purpose is to model a self-regulating internal process within a larger AI system that refines raw experience into abstracted, symbolic knowledge and produces feedback signals (tuning data) suitable for guiding an external autotuning mechanism or informing other AI components
This version focuses on incorporating simulated utility feedback into memory management and refining the assessment metrics and tuning data output

## Core Concepts

- **Dream Cycle:** This is the main operational loop (`run_dream_cycle`) encompassing state assessment, data preprocessing, symbolic aggregation, importance scoring, insight storage, parameter tuning application, memory pruning, and state persistence
- **Symbolic Insight:** This represents the primary output of the aggregation process (`_simulate_symbolic_aggregation`), usually a dictionary (`dream_result`) containing a short, templated textual phrase representing the core "message" of the dream and associated `key_themes`, simulating an emergent understanding
- **Importance Score:** A simulated metric assigned to each dream insight, reflecting its perceived significance based on synthesis level, theme count, novelty, and a chance for a "breakthrough" score, used for pruning
- **Theme Novelty:** A simulated metric measuring how different an insight's themes are compared to recent insights using Jaccard distance, used in importance scoring and assessment
- **Synthesis Level:** A simulated metric estimating the degree of abstraction or integration achieved during the symbolic aggregation process, influencing the importance score
- **Utility Score:** A simulated value (0-1) recorded externally reflecting the perceived usefulness of an insight when applied, influencing pruning decisions
- **Assessment (`assess_state`):** The "self-healing" phase where the engine analyzes recent dream history metrics like average/stdev of importance, novelty, utility, theme diversity, memory pressure, and parameter oscillation to evaluate its performance
- **Tuning Data:** The structured JSON output of `assess_state`, containing current metrics (raw and normalized), state status (`stable`, `needs_tuning`, etc), system mode (`Exploring`, `Stabilizing`, etc), and specific tuning suggestions (`suggested_adjustments` with deltas/factors), designed for external autotuner consumption
- **Oscillation:** The state where tuning recommendations repeatedly flip-flop between increasing and decreasing a parameter, which the system attempts to detect and dampen
- **Utility/Age-Weighted Pruning:** The memory management strategy (`prune_memory`) that removes the least valuable insights when memory limits are exceeded, where value combines base importance, an age penalty, and the recorded utility score

## Architecture & Components

The system comprises the central `MachineDream` class, which encapsulates all state and logic for the dream engine
This class handles initialization, core simulation methods for data processing and insight generation, self-assessment and tuning logic, memory management including pruning and persistence, manual control methods, and insight retrieval
A placeholder function `call_external_multimodal_model` represents where interaction with an external model like Gemini would occur, optionally using generated insights
The Command-Line Interface (CLI), managed by `main` and `run_interactive_loop` using `argparse`, provides user interaction for automated cycles, manual control, parameter configuration, and optional visualization

## Key Parameters

- `dream_complexity` (int): Controls simulated effort/depth of the symbolic aggregation, influences concept count and synthesis, tuned based on assessment
- `abstraction_level` (float, 00-10): Controls generalization vs detail focus, influences insight templates and themes, tuned based on assessment
- `tuning_level` (float): A general factor representing system confidence or exploration propensity, influences synthesis and tuning magnitude, tuned based on assessment
- `max_memory_size` (int): Maximum dream insights stored before pruning
- `abs_tuning_step` (float): Base step size for adjusting `abstraction_level`
- `tune_factor_range` (Tuple[float, float]): Min/max random factor for adjusting `tuning_level`
- `age_penalty_factor` (float): Factor controlling how much age reduces effective importance during pruning
- `oscillation_damping_factor` (float): Factor controlling step size reduction when oscillation is detected
- `target_ranges` (Dict): Defines ideal operating ranges for metrics used in normalization

## Simulation Logic

The script simulates complex processes via simplified heuristics and randomness
Preprocessing reduces simulated data size based on `abstraction_level` and input `complexity`
Insight Generation selects text templates based on parameters, formats them with `data_type`, selects `key_themes`, and calculates a `synthesis_level`
Novelty is calculated using Jaccard distance between current and recent themes
Importance combines `synthesis_level`, theme count, `theme_novelty`, and a chance for a "breakthrough"
Assessment compares recent metrics against `target_ranges` and internal thresholds to determine system state and generate recommendations, including oscillation detection
Tuning Application modifies parameters based on assessment recommendations, adjusting step sizes if damping is active

*Note: This is purely a simulation; outputs do not correspond to real-world AI behavior without external validation*

## Self-Tuning Mechanism

The engine employs a closed internal loop
First, `assess_state` analyzes recent dream history based on multiple metrics and detects oscillation
Second, it generates `tuning_data` containing metrics, status, system mode, and specific recommendations (`suggested_adjustments`)
Third, `apply_tuning` modifies the instance's parameters according to these suggestions, incorporating damping logic
Fourth, the next `run_dream_cycle` uses the newly tuned parameters, influencing subsequent insights and metrics
Fifth, the `tuning_data` produced by `assess_state` serves as the primary output for potential consumption by an external autotuning system

## Memory Management

Dream insights are stored chronologically as dictionaries in the `self.memory` list
When memory exceeds `max_memory_size`, `prune_memory` calculates a combined score for each insight using its `importance_score` (penalized by age via `age_penalty_factor`) and its recorded `utility_score` (defaulting to neutral if unavailable)
It removes the `N` insights with the lowest combined scores
The entire `self.memory` list and key metadata are persisted to a JSON file via `save_memory` and reloaded by `load_memory`
