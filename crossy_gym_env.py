# --- START OF FILE crossy_gym_env.py ---

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque


class CrossyGymEnv(gym.Env):
    """
    A lightweight, vectorized-ready Gymnasium environment simulating Crossy Road.
    Generates 160x160 programmatic grayscale canvases and outputs normalized
    proprioceptive scalars. Runs on the CPU to enable high-speed parallel rollouts.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, chunk_size=50):
        super(CrossyGymEnv, self).__init__()

        self.chunk_size = chunk_size

        # Define Spaces
        # Action Space: 0: Up, 1: Left, 2: Right, 3: Idle
        self.action_space = spaces.Discrete(4)

        # Observation Space matching the VAE + Proprioception requirements
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(160, 160), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })

        # Grayscale Intensity Constants
        self.INTENSITY_ROAD = 0.10
        self.INTENSITY_GRASS = 0.30
        self.INTENSITY_OBSTACLE = 0.50
        self.INTENSITY_CAR = 0.70
        self.INTENSITY_PLAYER = 0.90

        # Physical/Kinematic Constants
        self.GRID_MIN_X = -4
        self.GRID_MAX_X = 4
        self.CAR_LENGTH = 1.5
        self.CONTROL_FREQUENCY_HZ = 10.0
        self.DT = 1.0 / self.CONTROL_FREQUENCY_HZ

        # Global Map buffers
        self.terrain_map = {}  # Maps logical row Z -> 'G' (Grass) or 'R' (Road)
        self.obstacle_map = {}  # Maps (X, Z) -> True if blocked by static obstacle
        self.road_parameters = {}  # Maps road Z -> {'speed': float, 'direction': int}
        self.active_cars = {}  # Maps road Z -> list of float X center-points

        self.highest_generated_z = -1

    def _verify_connectivity(self, start_z, end_z, local_obstacles, local_terrain):
        """
        Runs a Breadth-First Search (BFS) to guarantee a traversable path
        exists from the beginning to the end of a generated terrain chunk.
        """
        # Collect unblocked entry nodes on the starting row
        queue = deque()
        visited = set()

        for col in range(self.GRID_MIN_X, self.GRID_MAX_X + 1):
            if not local_obstacles.get((col, start_z), False):
                node = (col, start_z)
                queue.append(node)
                visited.add(node)

        if not queue:
            return False

        # BFS Loop
        while queue:
            curr_x, curr_z = queue.popleft()

            if curr_z == end_z:
                return True

            # Explore orthogonal neighbors: Up, Down, Left, Right
            for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, nz = curr_x + dx, curr_z + dz

                if self.GRID_MIN_X <= nx <= self.GRID_MAX_X and start_z <= nz <= end_z:
                    neighbor = (nx, nz)
                    if neighbor not in visited and not local_obstacles.get(neighbor, False):
                        visited.add(neighbor)
                        queue.append(neighbor)

        return False

    def _generate_chunk(self):
        """
        Procedurally generates a new chunk of terrain, guaranteeing path traversability
        using iterative BFS rejection sampling before committing to global state.
        """
        start_z = self.highest_generated_z + 1
        end_z = start_z + self.chunk_size - 1

        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            local_terrain = {}
            local_obstacles = {}
            local_road_params = {}

            # Maintain consistency and prevent long repeating row blocks
            consec_grass = 0
            consec_road = 0

            for z in range(start_z, end_z + 1):
                # Ensure the very first rows of the game map are grass safe-zones
                if z < 5:
                    local_terrain[z] = 'G'
                    consec_grass += 1
                    continue

                # Determine terrain type probabilistically
                prob_grass = 0.40
                if consec_grass >= 3:
                    chosen_terrain = 'R'
                elif consec_road >= 4:
                    chosen_terrain = 'G'
                else:
                    chosen_terrain = 'G' if random.random() < prob_grass else 'R'

                local_terrain[z] = chosen_terrain

                if chosen_terrain == 'G':
                    consec_grass += 1
                    consec_road = 0

                    # Spawn static obstacles in cols [-4, 4]
                    for x in range(self.GRID_MIN_X, self.GRID_MAX_X + 1):
                        # The central spawning row/column of the player start is always kept clear
                        if z == 0 or (z < 5 and x == 0):
                            continue
                        # REDUCED OBSTACLE DENSITY: Lower chance of traps forming
                        if random.random() < 0.15:
                            local_obstacles[(x, z)] = True
                else:
                    consec_road += 1
                    consec_grass = 0

                    # Generate speed (columns/sec) and direction (-1: Left, 1: Right)
                    direction = 1 if random.random() > 0.5 else -1
                    speed = random.uniform(1.5, 4.0)
                    local_road_params[z] = {'speed': speed, 'direction': direction}

            # Verify connectivity of the newly constructed block
            if self._verify_connectivity(start_z, end_z, local_obstacles, local_terrain):
                # BFS validated a clean traversal, commit to global dictionary maps
                self.terrain_map.update(local_terrain)
                self.obstacle_map.update(local_obstacles)
                self.road_parameters.update(local_road_params)

                # Initialize starting car queues for committed road rows
                for z, params in local_road_params.items():
                    self.active_cars[z] = []
                    # REDUCED TRAFFIC DENSITY: Fewer cars per lane, larger gaps
                    num_initial_cars = random.randint(1, 2)
                    last_x = -7.0
                    for _ in range(num_initial_cars):
                        spawn_x = last_x + random.uniform(4.0, 7.5)
                        if spawn_x < 7.0:
                            self.active_cars[z].append(spawn_x)
                            last_x = spawn_x

                self.highest_generated_z = end_z
                return

            attempts += 1

        raise RuntimeError(f"Failed to generate a connected chunk after {max_attempts} iterations.")

    def _update_physics(self):
        """
        Updates kinematics of active vehicles, including spawning and boundary recycling.
        """
        # Only simulate lanes near the active player viewport range
        range_min = max(0, self.player_z - 10)
        range_max = self.player_z + 20

        for z in range(range_min, range_max + 1):
            if self.terrain_map.get(z) != 'R':
                continue

            params = self.road_parameters[z]
            speed = params['speed']
            direction = params['direction']

            # Step position
            updated_cars = []
            for car_x in self.active_cars[z]:
                next_x = car_x + (direction * speed * self.DT)
                # Keep active if inside logical recycling boundaries
                if -7.0 <= next_x <= 7.0:
                    updated_cars.append(next_x)

            # REDUCED TRAFFIC DENSITY: Needs bigger gap to respawn, lower probability
            if direction == 1:  # Moving Right, Spawns on left (-7.0)
                if not updated_cars or min(updated_cars) > -2.5:
                    if random.random() < 0.12:
                        updated_cars.append(-7.0)
            else:  # Moving Left, Spawns on right (7.0)
                if not updated_cars or max(updated_cars) < 2.5:
                    if random.random() < 0.12:
                        updated_cars.append(7.0)

            self.active_cars[z] = sorted(updated_cars)

    def _render_canvas(self):
        """
        Constructs and renders the 160x160 programmatic grayscale image.
        Uses a fixed palette to represent the player, road, grass, and obstacles.
        Fills all 160 vertical pixels with 16 rows (each 10px high) to maximize VAE capacity.
        """
        canvas = np.zeros((160, 160), dtype=np.float32)

        # Anchor the viewport strictly to the smooth-scrolling camera.
        z_bottom = int(self.camera_z)
        z_top = z_bottom + 15  # 16 rows total (z_bottom to z_bottom + 15)

        # Iterate over viewport row positions (0 to 15 from top to bottom)
        for v_row in range(16):
            logical_z = z_top - v_row
            row_start = v_row * 10
            row_end = row_start + 10

            # Check map boundary limit
            if logical_z < 0:
                # Out-of-bounds terrain rendered as Road Surface
                canvas[row_start:row_end, :] = self.INTENSITY_ROAD
                continue

            terrain_type = self.terrain_map.get(logical_z, 'G')
            base_intensity = self.INTENSITY_GRASS if terrain_type == 'G' else self.INTENSITY_ROAD
            canvas[row_start:row_end, :] = base_intensity

            # Draw static obstacles (Trees/Rocks)
            if terrain_type == 'G':
                for col_idx, col_val in enumerate(range(self.GRID_MIN_X, self.GRID_MAX_X + 1)):
                    if self.obstacle_map.get((col_val, logical_z), False):
                        col_start = 3 + col_idx * 17
                        col_end = col_start + 17
                        canvas[row_start:row_end, col_start:col_end] = self.INTENSITY_OBSTACLE

            # Draw cars
            elif terrain_type == 'R':
                for car_x in self.active_cars.get(logical_z, []):
                    # Project continuous float X bounds [-4.5, 4.5] to canvas pixels [3, 156]
                    car_start_x = car_x - (self.CAR_LENGTH / 2.0)
                    car_end_x = car_x + (self.CAR_LENGTH / 2.0)

                    px_start = int(3 + (car_start_x + 4.5) * 17)
                    px_end = int(3 + (car_end_x + 4.5) * 17)

                    px_start = max(3, min(156, px_start))
                    px_end = max(3, min(156, px_end))

                    if px_end > px_start:
                        canvas[row_start:row_end, px_start:px_end] = self.INTENSITY_CAR

            # Overlay player
            if logical_z == self.player_z:
                col_idx = self.player_x - self.GRID_MIN_X
                col_start = 3 + col_idx * 17
                col_end = col_start + 17
                canvas[row_start:row_end, col_start:col_end] = self.INTENSITY_PLAYER

        return canvas

    def _get_action_mask(self):
        """
        Builds a discrete action mask to restrict out-of-bound lateral moves.
        """
        mask = np.zeros(4, dtype=np.float32)
        if self.player_x <= self.GRID_MIN_X:
            mask[1] = -1e8  # Restrict moving Left
        if self.player_x >= self.GRID_MAX_X:
            mask[2] = -1e8  # Restrict moving Right
        return mask

    def _check_collisions(self):
        """
        Checks if player's bounding box intersects any active vehicle.
        """
        if self.terrain_map.get(self.player_z) != 'R':
            return False

        # NARROW CHICKEN: Halved the collision width box to allow squeezing
        p_left, p_right = self.player_x - 0.2, self.player_x + 0.2

        for car_x in self.active_cars.get(self.player_z, []):
            c_left = car_x - (self.CAR_LENGTH / 2.0)
            c_right = car_x + (self.CAR_LENGTH / 2.0)

            # Simple 1D interval overlap check
            if max(p_left, c_left) < min(p_right, c_right):
                return True

        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Flush maps
        self.terrain_map.clear()
        self.obstacle_map.clear()
        self.road_parameters.clear()
        self.active_cars.clear()
        self.highest_generated_z = -1

        # Spawn initial terrain
        self._generate_chunk()

        # Initialize Player Variables
        self.player_x = 0
        self.player_z = 0

        # CAMERA LOGIC: Initialize scrolling eagle camera
        self.camera_z = -3.0
        self.camera_speed = 1.0  # rows per second

        # Metric tracking
        self.score_milestone = 10
        self.current_score = 0

        self._update_physics()
        image = self._render_canvas()

        obs = {
            "image": image,
            "scalars": np.array([0.0, 0.0], dtype=np.float32)  # [norm_x, eagle_threat]
        }

        info = {
            "action_mask": self._get_action_mask(),
            "score": 0
        }

        return obs, info

    def step(self, action):
        # Identify if the direct path forward is blocked before movement
        forward_blocked = self.obstacle_map.get((self.player_x, self.player_z + 1), False)

        prev_z = self.player_z
        prev_x = self.player_x

        target_x = self.player_x
        target_z = self.player_z

        # 1. Map discrete action
        if action == 0:  # Up
            target_z += 1
        elif action == 1:  # Left
            target_x = max(self.GRID_MIN_X, self.player_x - 1)
        elif action == 2:  # Right
            target_x = min(self.GRID_MAX_X, self.player_x + 1)
        # action == 3: Idle (remains target_x, target_z)

        # 2. Check Static Obstacles (Tactile Response Check)
        obstacle_hit = self.obstacle_map.get((target_x, target_z), False)

        if obstacle_hit:
            # Action blocked: revert movement destination
            target_x = self.player_x
            target_z = self.player_z
            tactile_penalty = -0.5
        else:
            self.player_x = target_x
            self.player_z = target_z
            tactile_penalty = 0.0

        # 3. Simulate Vehicle Positions
        self._update_physics()

        # CAMERA PHYSICS: Apply a constant breeze pushing the balloon forward
        self.camera_z += self.camera_speed * self.DT
        # Speed up the baseline camera slightly as the agent progresses (temporarily set to constant 1)
        self.camera_speed = 1.0

        # Smooth Catchup (Spring/Elastic String centering pull):
        # Target camera position centers the player about 3 rows from the viewport bottom
        target_camera_z = self.player_z - 3.0
        if self.camera_z < target_camera_z:
            self.camera_z += (target_camera_z - self.camera_z) * 1.5 * self.DT

        # Tight String Limit (Hard Cap):
        # If the player is near the top of the viewport (e.g., 8 rows ahead), the string gets tight
        # and pulls the camera forward immediately.
        max_dist = 8.0
        if self.player_z - self.camera_z > max_dist:
            self.camera_z = self.player_z - max_dist

        # 4. Procedural Generation Check
        # Generate next map chunk if player approaches the boundary
        if self.highest_generated_z - self.player_z < 30:
            self._generate_chunk()

        # 5. Evaluate Termination Conditions
        done = False
        termination_reward = 0.0

        # Check Eagle Strike / Camera OOB
        if self.player_z <= self.camera_z:
            done = True
            termination_reward = -15.0
        # Check Vehicle Collision
        elif self._check_collisions():
            done = True
            termination_reward = -15.0

        # 6. Calculate Reward Shaping
        reward_base = 0.0

        if action == 0:
            reward_base = 0.1  # Base reward for attempting to go forward

        # THE JAILBREAK BONUS:
        # If blocked by a rock, reward moving Left/Right.
        # If blocked and idling, apply a penalty to discourage freezing.
        if forward_blocked:
            if action in [1, 2] and not obstacle_hit:
                reward_base = 0.25  # Strategic value for lateral exit
            elif action == 3:
                reward_base = -0.15  # Penalty for "giving up" in jail
        else:
            # Normal movement logic
            if action in [1, 2]:
                reward_base = -0.02  # Tiny penalty to keep movement purposeful
            elif action == 3:
                reward_base = -0.05  # Tiny idle penalty

        # Progress Tracking Reward
        reward_progress = 1.0 * (self.player_z - prev_z)

        # Milestone Bonus Reward shaping implementation
        milestone_bonus = 0.0
        if self.player_z >= self.score_milestone:
            milestone_bonus = 5.0
            self.score_milestone += 10

        # Combine rewards
        reward = reward_base + reward_progress + tactile_penalty + milestone_bonus + termination_reward

        # Update metrics
        self.current_score = max(self.current_score, self.player_z)

        # 7. Package Outputs
        image_obs = self._render_canvas()

        # Normalize scalars for network processing
        norm_x = self.player_x / 5.0
        distance_from_camera = self.player_z - self.camera_z
        eagle_threat = max(0.0, min(1.0, 1.0 - (distance_from_camera / 6.0)))
        scalars_obs = np.array([norm_x, eagle_threat], dtype=np.float32)

        obs = {
            "image": image_obs,
            "scalars": scalars_obs
        }

        info = {
            "action_mask": self._get_action_mask(),
            "score": self.current_score,
            "tactile_bump": obstacle_hit
        }

        return obs, reward, done, False, info

    def render(self):
        """
        Returns the raw pixel matrix for standard Gymnasium visualization.
        """
        raw_gray = self._render_canvas()
        # Rescale the 0.0-1.0 float canvas back to 0-255 uint8 channel output
        scaled_gray = (raw_gray * 255.0).astype(np.uint8)
        # Duplicate channels to match standard RGB formatting
        return np.stack([scaled_gray, scaled_gray, scaled_gray], axis=-1)