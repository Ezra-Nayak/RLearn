# Simulation-Based Crossy Road RL Training Pipeline (DirectML & Vectorized PPO)

This repository contains a high-throughput, parallelized reinforcement learning (RL) training pipeline designed to train a neural agent to navigate an infinite procedural terrain. The project is modeled mathematically and structurally on *Crossy Road*, utilizing a decoupled pipeline consisting of a **Split-Brain Spatial VQ-VAE** for visual feature extraction and a **CoordConv Actor-Critic Proximal Policy Optimization (PPO)** network for control logic.

---

## 1. Architectural Blueprint & Design Rationale

```
       +-----------------------------------------------------------+
       |             Simulated Environment Subprocesses             |
       |  N x AsyncVectorEnv[CrossyGymEnv -> FrameStackWrapper]    |
       +-----------------------------------------------------------+
                                    ||  (Observations: 160x160x4 Frame Stack)
                                    \/  (Device: Host CPU Memory)
       +-----------------------------------------------------------+
       |                    Parent Process (GPU)                    |
       |             DirectML Device (privateuseone:0)             |
       |                                                           |
       |             +-------------------------------+             |
       |             |   Pretrained SpatialVQVAE     |             |
       |             |   (Freezes Weights post-Phase2) |             |
       |             +-------------------------------+             |
       |                //                       \\                |
       |               //                         \\               |
       |       [64-ch Context Latent]     [64-ch Trend Latent]     |
       |               \\                         //               |
       |                \\                       //                |
       |                Concat: [128, 20, 20] Latent Map           |
       +-----------------------------------------------------------+
                                    ||
                                    \/  (Transfer to Host CPU Memory)
       +-----------------------------------------------------------+
       |                    Parent Process (CPU)                    |
       |                     PPO Execution Core                    |
       |                                                           |
       |             +-------------------------------+             |
       |             |     CoordConv Grid Injection  |             |
       |             |     - Insert Absolute X & Y    |             |
       |             |     - Result: [130, 20, 20]   |             |
       |             +-------------------------------+             |
       |                             ||                            |
       |                             \/                            |
       |                     [Actor-Critic Head]                   |
       |                     - Disjoint MLP paths                  |
       |                     - Action Mask Evaluation              |
       |                             ||                            |
       |                             \/                            |
       |           Decisions & Advantage Computations (GAE)        |
       +-----------------------------------------------------------+
```

### The Bottleneck & The Vectorized Shift
In live-game RL projects, the agent is heavily bound to a single Windows execution process. Frame grabs (`mss`), OS window hooks (`win32gui`), and keyboard injection APIs (`pydirectinput`) limit execution speeds to a rigid 10–15 Hz. 

This project bypasses this throughput ceiling by shifting to a highly optimized, vectorized simulation environment. Visual states are programmatically drawn directly to memory inside separate CPU subprocesses. By stacking multiple workers inside an asynchronous Gym vectorized environment, the training pipeline generates experiences at **$300\text{--}400+$ FPS**, fully saturating host hardware.

### Visual Architecture: Split-Brain Spatial VQ-VAE
Rather than passing raw visual pixels directly to the RL network, the visual input of shape `(4, 160, 160)` is encoded into a discrete latent representation using a custom **Spatial VQ-VAE** (`SpatialVQVAE`).
* **Encoder**: Standard convolution layers map the $160 \times 160$ input space down to a compact $20 \times 20$ grid of spatial embeddings containing $128$ channels.
* **Decoupled Latent Channels**: The $128$-channel spatial map is split physically down the middle:
  * **Context Brain ($z_c$)**: $64$ channels passed through a discrete codebook (`vq_c`) containing $512$ vectors of dimension $64$. It is decoded to reconstruct the present static frame ($t$).
  * **Trend Brain ($z_t$)**: $64$ channels passed through a second discrete codebook (`vq_t`) of identical size. It is decoded to predict the upcoming future frame ($t+1$).
* **Vector Quantization (Spherical & Dead-Code Revival)**: Standard Euclidean quantization often suffers from codebook collapse, where only a few codebook vectors are ever activated. This model implements:
  * **Spherical VQ (Cosine Similarity)**: Inputs and codebook weights are L2-normalized prior to distance matching. Quantization pathing is decided purely by spatial patterns rather than color/intensity magnitude.
  * **Dead-Code Revival**: During training, code vectors that do not receive updates are dynamically overwritten by random slices of the active input feature map.
* **The Emergent Chicken-Deletion Property**: An unexpected but crucial mathematical behavior of this decoupled architecture is that **the predictive Trend model completely deletes the player's model from its $t+1$ prediction**. Because the VAE receives no action inputs, the player's movements are highly unpredictable. Under L1 and Sobel optimization, the model minimizes loss by dropping the unpredictable entity (the player) and devoting 100% of its representation budget to tracking the deterministic elements of the world (moving cars). The Context model preserves the current position of the player, meaning the combined latent map represents a perfectly clean separation of "Self" and "Environment Dynamics."

### Policy Architecture: Actor-Critic with CoordConv
To convert the spatial latents into discrete motor actions, the $128$-channel, $20 \times 20$ latent representation is processed by a highly customized policy:
* **CoordConv Injection**: Convolutional networks are spatially translation-invariant, which prevents standard CNNs from identifying absolute grid positions. To correct this, the policy procedurally generates linear normalized $X$ and $Y$ coordinate grids and appends them to the latent map. This expands the input from $128$ to $130$ channels, providing the Conv2D layers with precise spatial coordinates.
* **Disjoint Networks**: To prevent destructive gradient interference between policy and value predictions, the Actor and Critic heads utilize entirely disjoint linear-layer pathings.
* **Action Masking**: The environments calculate explicit masks to restrict lateral boundaries. These masks are added directly to the raw actor logits before evaluating standard categorical distributions, preventing out-of-bounds exploration.
* **Joint Policy-Value Imitation Bootstrapping (Behavioral Cloning)**: To bypass the early exploration barrier of reinforcement learning, the network undergoes supervised pretraining. It jointly learns the Actor's optimal action distribution (via CrossEntropy) and the Critic's value estimation of states (via MSE over normalized episodic returns). To prevent unstable state-value variance from destabilizing representation learning, the feature extractor gradients are detached from the Critic head during supervised pretraining.

---

## 2. Detailed Module Breakdown

### `crossy_gym_env.py`
The absolute center of the training pipeline, containing the complete mathematical implementation of the procedural simulator.
* **State Logic**: Keeps track of logical coordinates $x_p \in [-4, 4]$ and $z_p \in [0, \infty)$ for the player.
* **Procedural Generation**: Constructs terrains in $L=50$ row chunks. It maps rows as Grass or Roads. Spawning of static obstacles (Trees/Rocks) is capped at a probability of $0.15$.
* **BFS Connectivity Guarantee**: To prevent generating unnavigable maps, every procedural chunk must pass an internal Breadth-First Search (BFS) validation. If the BFS cannot find a continuous path from the bottom of the chunk to the top around the static obstacles, the chunk is discarded and regenerated.
* **Continuous Car Physics**: Simulates continuous horizontal coordinates $x_{car, i} \in [-7.0, 7.0]$ for cars on roads. Velocity vectors are determined per road lane. Cars spawn continuously and recycle smoothly across the boundary edge.
* **Scrolling Camera Mechanics (Balloon-String Analogy)**: Establishes a smooth camera tracking system that resolves the idle-progression relationship:
  * **Constant Breeze**: The camera line $z_{camera}$ moves up the Z-axis continuously at $1.0$ rows/sec, scaling up to $2.5$ rows/sec as the player progresses.
  * **Elastic Spring Catch-Up**: When the player progresses ahead of the target viewport center ($z_{player} - z_{camera} > 3.0$), an elastic spring pull of $(z_{target} - z_{camera}) \times 1.5 \times DT$ accelerates the camera smoothly toward them.
  * **Taut String Limit (Hard Cap)**: If the player moves too rapidly and approaches the upper third of the screen ($z_{player} - z_{camera} > 8.0$), the virtual string tightens, instantly dragging the camera up to maintain containment.
  * **Eagle Death**: If the player falls behind the moving viewport bottom edge ($z_{player} \leq z_{camera}$), they are immediately eliminated.
* **Viewport Projection**: To preserve visual fluidity, `_render_canvas()` anchors the viewport bottom strictly to `int(self.camera_z)`, removing visual dead zones and enabling continuous scroll.
* **NumPy Grayscale Rendering**: Translates the logical grid coordinate layout into a precise $160 \times 160$ grayscale canvas. Different physical layers are written into the matrices as flat constant values: Road ($0.10$), Grass ($0.30$), Obstacle ($0.50$), Car ($0.70$), and Player ($0.90$).

### `gym_wrappers.py`
Provides preprocessing wrappers that hook the active environment step methods.
* **Temporal Frame Stacking**: Subclasses `gym.ObservationWrapper`. It intercepts raw frames from the simulator and manages an internal deque of size 4.
* **Reset/Step Mapping**: On initialization, it pads the stack with the first frame. On step, it pushes the newest observation, returning a concatenated numpy array of shape `(4, 160, 160)`.

### `collect_sim_data.py`
The data harvesting utility used to generate balanced offline visual samples for VAE pretraining.
* **Oracle Autoplay Collection**: Employs the real-time A*/BFS lookahead planner to generate high-quality trajectory frames with optimal paths, rather than relying on noisy random explorations.
* **Data Balancing Triggers**:
  * **Progression Cap**: Resets the active run immediately once a score of 100 is achieved to prevent a single high-scoring trajectory from overrepresenting the dataset.
  * **Grass Block Idle-Spam**: To feed the predictive Trend VQ-VAE model with sufficient examples of stagnation and camera-creep death, there is a 10% chance upon reaching a score of 50 on any grass block to disable the pathfinder and spam `Idle` until eagle death is triggered.
* **Alignment Logic**: Manages a 5-step frame history. It saves the stacked visual timeline `[t-3, t-2, t-1, t]` as the training inputs, and the future frame `[t+1]` as the target.
* **Buffered Savings**: Writes chunks of $1,000$ samples into compact `.npy` files inside `/sim_data` to minimize RAM footprint.

### `train_vision_sim.py`
The visual pretraining suite for the **Spatial VQ-VAE**.
* **DirectML Compatibility (`DMLAdam`)**: Replaces standard Adam optimizer calls. Because the default PyTorch Adam execution depends on `.lerp_` (which can crash on Windows DirectML drivers during backward passes), `DMLAdam` recalculates exponential moving averages via basic multiplication and addition operations.
* **Foveated Motion-Attention Mask**: Rather than calculating raw L1 loss over the whole canvas, the training loop computes a dynamic spatial loss mask:
  $$\text{Motion Diff} = |x_t - x_{t-1}|$$
  Regions showing movement (such as passing cars) are multiplied by a weight of $5.0$. A static vertical bounding box around the player's center-zone is multiplied by $1.5$. This focuses the model's capacity on capturing fine-grained edges of active hazards rather than static background grass.
* **Custom Edge Preserving Sobel Loss**: Combines standard pixel-level L1 loss with an edge-detection filter to prevent the network from producing blurred/smudged reconstructions.

### `verify_vae_sim.py`
A live graphical diagnostic utility for inspecting the health of the trained VAE.
* **Dashboard Visualizer**: Captures live frames from the simulator and displays an interactive, real-time grid:
  1. *Input Frame (t)*
  2. *VQ-VAE Reconstruction*
  3. *Trend Prediction (t+1)* (visualizing the projected path of traffic while confirming player deletion)
  4. *Absolute L1 Error Heatmap* (using a JET colormap overlay)
* **Mathematical Telemetry**: Computes and displays the Mean Absolute Error (MAE) and the Variance Preservation Ratio (indicating if reconstructed object edges are sharp or smudged).

### `train_ppo_sim.py`
The primary reinforcement learning engine that runs PPO.
* **Asynchronous Multi-Processing**: Spawns $N$ instances of `CrossyGymEnv` using `gymnasium.vector.AsyncVectorEnv`. Each environment instance runs in its own subprocess on the CPU.
* **Decoupled Compute Devices**: 
  * The batched frames `(N, 4, 160, 160)` are pushed to the GPU/DirectML device for high-speed VAE feature extraction.
  * The compressed features are then transferred back to the CPU, where policy evaluation, backward passes, and gradient optimizations occur. This hybrid model prevents DirectML multi-process lockups and maintains absolute gradient stability.
* **Manual Trajectory Logging**: Collects running step scores and rewards across all parallel environments. When an individual environment terminates, it reads the final metadata and appends the final episode metrics to a rolling 100-episode history deque.
* **State-Optimizer Checkpoint Bundling**: Periodically writes full `.pth` state packages containing model parameters, optimizer momentum parameters, global step indices, and the historical best score. Old checkpoints are automatically deleted to prevent storage bloat.

### `vector_memory.py`
The batched rollout memory and GAE advantage calculator.
* **Batched Buffering**: Collects parallel step traces of shape `(rollout_steps, num_envs, ...)` instead of single-dimensional lists.
* **Vectorized GAE**: Computes Generalized Advantage Estimations across the entire environment axis simultaneously using matrix algebra.
* **Flattening for Minibatches**: Flattens the completed step matrices from $T \times N$ to a single-dimension tensor of size $TN$. This allows standard randomized shuffling and minibatch slicing during policy updates.

### `play_sim_manual.py`
An interactive diagnostic tool designed to let you manually play the simulation.
* **Opencv Window Hooks**: Listens to keyboard triggers directly on the display window, eliminating external Windows input dependencies.
* **State Diagnostics**: Overlays a visual dashboard showing real-time metrics, active scores, cumulative rewards, step numbers, and collision warnings.

### `play_oracle.py`
An automated solver script showcasing optimal pathing decisions via real-time search.
* **Time-Expanded Grid Search**: Simulates paths in a time-expanded search tree. By collapsing search states down to `(player_x, player_z)` keys per depth level, the planner prunes $4^d$ tree branches to $\approx 1,400$ nodes for a 12-step lookahead.
* **State Isolation**: Captures and restores the environment and pseudo-random seed state during branching simulations, guaranteeing deterministic lookahead predictions without disrupting the main visual rollout.
* **Multi-Objective Survival Planner**: Evaluates branches based on forward progress, minimal time-to-reach, and falls back to a maximum-survival path if a collision is unavoidable.
* **Boundary-Masking Safety Override**: Enforces coordinate-level masking directly inside the solver. If the lookahead search yields a lateral movement that exceeds the grid boundary (which would act as a stationary "Idle" move but conflict with the environment's active action mask), it collapses the action to `3` (Idle) to prevent training-loss gradient explosions during supervised pretraining.

### `train_bc.py`
A high-throughput, joint policy-value supervised pretraining suite (Behavioral Cloning).
* **Parallel CPU Harvesting**: Distributes independent raw trajectory generation across a multiprocessing pool (`ProcessPoolExecutor`), bypassing single-threaded Python bottlenecks and capturing thousands of expert transitions per minute.
* **Batched GPU Latent Encoding**: Decouples search simulation from neural rendering. Once CPU workers harvest raw environment frames, the main process gathers them into batches of 128 and processes them through the frozen `SpatialVQVAE` on the GPU in seconds.
* **Joint Policy-Value Supervised Objective**: Minimizes the combined error of the Actor (CrossEntropy action classification) and the Critic (Mean Squared Error over normalized returns). Detaches CNN features from the Critic's backpropagation path to keep policy representations stable.
* **Warm Start Resuming**: Generates the initialized checkpoint `checkpoints/ppo_sim_bc.pth`. When standard vectorized PPO is executed, the core loop detects and loads these weights to bootstrap training performance.

### `decode_ppo_sim.py`
A diagnostics tool that serves as a simulated fMRI brain scanner for your trained agent.
* **Attention Saliency Maps**: Performs backpropagation from the actor's strongest decision logit back to the VAE's spatial latents, constructing a real-time attention heatmap showing what visual patterns the model is focused on.
* **Interactive Synaptic Connectivity Graph**: Downsamples the actor's high-dimensional linear weights and layer activations into a simplified $8 \to 12 \to 12 \to 4$ node layout. Connection pathways are rendered as Bézier curves with thicknesses and colors (Cyan for positive activations, Magenta for negative ones) indicating relative signal strength.
* **Recordings**: Step decisions, critic value estimations, and policy distributions are compiled and saved directly to disk as diagnostic frame logs.

---

## 3. Project Configuration & Parameters

The table below outlines the default training configurations and data structures used across the simulation pipeline:

| Module / Class | Setting | Parameter / Shape | Purpose |
| :--- | :--- | :--- | :--- |
| **Global Pipeline** | `NUM_ENVS` | `32` | Scales visual execution batches on DirectML GPU |
| **Global Pipeline** | `ROLLOUT_STEPS` | `128` | Timesteps collected per environment before PPO updates |
| **Global Pipeline** | `MINIBATCH_SIZE` | `64` | Sub-sample size for policy optimization epochs |
| **CrossyGymEnv** | State Dimension | `Dict("image", "scalars")` | Exact observation structures passed to wrappers |
| **CrossyGymEnv** | `self.GRID_MIN_X` / `GRID_MAX_X` | `[-4, 4]` | Logical lateral limits of the grid |
| **CrossyGymEnv** | `self.camera_speed` | `1.0` (scaled to `2.5` max) | Progression rate of the trailing death line |
| **FrameStackWrapper** | Output Shape | `(4, 160, 160)` | Stacked historical frame format for visual encoding |
| **SpatialVQVAE** | Latent Output Shape | `(128, 20, 20)` | Concatenated spatial context and trend embeddings |
| **SpatialVQVAE** | Codebook Size | `512` embeddings of `64` dim | Discretized representation capacity of each head |
| **ActorCritic** | Conv Layer Input | `130` channels | 128 VAE channels + 2 coordinate grids (CoordConv) |
| **ActorCritic** | Output Layer | `4` discrete logits | [Up, Left, Right, Idle] action choices |
| **play_oracle.py** | `lookahead_steps` | `12` | Planning horizon depth for optimal pathing decisions |
| **train_bc.py** | `TARGET_STEPS` | `15,000` | Sample threshold of optimal transitions collected in memory |
| **train_bc.py** | `EPOCHS` | `25` | Supervised training epochs for policy-value bootstrapping |