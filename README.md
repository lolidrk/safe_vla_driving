# Safe-VLA: Safety-Filtered Vision-Language-Action Pipeline for Autonomous Driving

**Bridging Multimodal Foundation Model Reasoning with Control Barrier Function (CBF) Safety Guarantees**

## Overview
Vision-Language-Action (VLA) models excel at high-level semantic reasoning in complex, long-tail driving scenarios, but unconstrained neural trajectory outputs often violate vehicle kinematics or safety margins. 

**Safe-VLA** is a hybrid autonomy pipeline that decouples **System 2 semantic reasoning** from **System 1 safety-critical control**:
1. **Multimodal VLA Backbone:** Fine-tunes a quantized 3B Vision-Language Model (`Qwen2.5-VL-3B-Instruct`) via QLoRA on **nuScenes / DriveLM** scenarios to jointly output a Chain-of-Thought (CoT) driving rationale and a 3-second future ego-waypoint trajectory ($x, y, v$ at 2 Hz).
2. **MPC + CBF-QP Safety Filter:** Formulates a kinematic trajectory tracker with Discrete-Time Control Barrier Functions (CBF) in `CVXPY` to minimally modify the VLA's predicted waypoints while enforcing hard kinematic and obstacle-avoidance constraints in Bird's-Eye-View (BEV) space.
3. **Real-Time ROS Visualization:** Publishes camera feeds, CoT rationales, 3D obstacle bounding boxes, and raw vs. safety-filtered trajectories for closed-loop interpretability.

---

## System Architecture

```text
[Front Cam + Ego Kinematics] ──► [Quantized Qwen2.5-VL (QLoRA)]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        [Chain-of-Thought Rationale]           [Unconstrained Waypoints (3s)]
        ("Yield to merging vehicle")                        │
                    │                                       ▼
                    │                          [MPC / CBF-QP Safety Filter] ◄── [3D Obstacle Boxes]
                    │                                       │
                    └──────────────► [ROS / BEV Viz] ◄──────┘
                                            │
                                            ▼
                                [Safe Trajectory (x, y, v, δ, a)]
