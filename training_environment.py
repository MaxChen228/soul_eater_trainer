# soul_eater_trainer/training_environment.py
import pygame
import numpy as np
import random
import os
import sys # ⭐️ 新增 sys

# ⭐️ 新增：為了載入 BugDQNAgent (將作為我們的板子 AI 代理)
current_env_dir = os.path.dirname(os.path.abspath(__file__))
if current_env_dir not in sys.path:
    sys.path.append(current_env_dir)
from dqn_definitions import BugDQNAgent, QNet # ⭐️ 假設對手板 AI 也使用 QNet

# --- PlayerState (保持不變) ---
class PlayerState:
    def __init__(self, initial_x=0.5, initial_paddle_width_normalized=0.15, initial_lives=3, player_identifier="player"):
        self.x = initial_x
        self.prev_x = initial_x # ⭐️ 新增 prev_x 以計算板子速度 (如果需要)
        self.paddle_width_normalized = initial_paddle_width_normalized
        self.lives = initial_lives
        self.identifier = player_identifier

# --- SimplifiedSoulEaterBugSkill (保持不變) ---
class SimplifiedSoulEaterBugSkill:
    def __init__(self, mock_env, owner_player_state, target_player_state, params):
        self.env = mock_env
        self.owner = owner_player_state
        self.target_player_state = target_player_state
        self.bug_x_rl_move_speed = params.get("bug_x_rl_move_speed", 0.03)
        self.bug_y_rl_move_speed = params.get("bug_y_rl_move_speed", 0.03)
        self.base_y_speed = params.get("base_y_speed", 0.015)
        self.duration_ms = params.get("duration_ms", 8000)
        self.bug_visual_radius_norm = params.get("bug_visual_radius_norm", 0.05)
        self.activated_time = 0
        self.active = False

        # ⭐️ 新增：從 params 讀取蟲的初始位置隨機範圍
        self.initial_bug_x_min = params.get("initial_bug_x_min", 0.4) # 提供預設值
        self.initial_bug_x_max = params.get("initial_bug_x_max", 0.6)
        self.initial_bug_y_min = params.get("initial_bug_y_min", 0.75)
        self.initial_bug_y_max = params.get("initial_bug_y_max", 0.85)


    def activate(self):
        self.active = True
        self.activated_time = pygame.time.get_ticks()
        
        # ⭐️ 修改：隨機化蟲的初始位置
        self.env.ball_x = random.uniform(self.initial_bug_x_min, self.initial_bug_x_max)
        self.env.ball_y = random.uniform(self.initial_bug_y_min, self.initial_bug_y_max)
        # 確保初始位置在邊界內 (雖然隨機範圍應該已經考慮，但多一層保護)
        self.env.ball_x = np.clip(self.env.ball_x, self.bug_visual_radius_norm, 1.0 - self.bug_visual_radius_norm)
        self.env.ball_y = np.clip(self.env.ball_y, self.bug_visual_radius_norm, 1.0 - self.bug_visual_radius_norm)
        
        self.env.trail.clear()

    def deactivate(self, hit_paddle=False, scored=False):
        self.active = False

    def _get_bug_observation(self):
        bug_x_norm = self.env.ball_x
        bug_y_norm = self.env.ball_y
        target_paddle_x_norm = self.target_player_state.x
        target_paddle_half_width_norm = self.target_player_state.paddle_width_normalized / 2.0
        
        bug_y_distance_to_goal_line = bug_y_norm # Y=0 是目標線

        observation = [
            bug_x_norm, bug_y_norm, target_paddle_x_norm,
            target_paddle_half_width_norm, bug_x_norm - target_paddle_x_norm,
            bug_y_distance_to_goal_line
        ]
        return np.array(observation, dtype=np.float32)

    def _apply_movement_and_constrain_bounds(self, delta_x_norm, delta_y_norm):
        self.env.ball_x += delta_x_norm * self.env.time_scale
        self.env.ball_y += delta_y_norm * self.env.time_scale
        self.env.ball_x = np.clip(self.env.ball_x, self.bug_visual_radius_norm, 1.0 - self.bug_visual_radius_norm)
        self.env.ball_y = np.clip(self.env.ball_y, self.bug_visual_radius_norm, 1.0 - self.bug_visual_radius_norm)

    def _update_trail(self):
        self.env.trail.append((self.env.ball_x, self.env.ball_y))
        if len(self.env.trail) > self.env.max_trail_length:
             self.env.trail.pop(0)

    def _check_bug_scored(self):
        target_goal_line_y_norm = self.bug_visual_radius_norm 
        if self.env.ball_y <= target_goal_line_y_norm:
            self.target_player_state.lives -= 1
            self.env.round_concluded_by_skill = True
            self.deactivate(scored=True)
            return True
        return False

    def _check_bug_hit_paddle(self):
        target_paddle = self.target_player_state
        target_paddle_x_min = target_paddle.x - target_paddle.paddle_width_normalized / 2
        target_paddle_x_max = target_paddle.x + target_paddle.paddle_width_normalized / 2
        
        bug_x_min = self.env.ball_x - self.bug_visual_radius_norm
        bug_x_max = self.env.ball_x + self.bug_visual_radius_norm
        bug_y_min = self.env.ball_y - self.bug_visual_radius_norm 
        bug_y_max = self.env.ball_y + self.bug_visual_radius_norm 

        paddle_y_surface_min = 0 
        paddle_y_surface_max = self.env.paddle_height_normalized 

        x_overlap = bug_x_max >= target_paddle_x_min and bug_x_min <= target_paddle_x_max
        y_overlap_with_paddle_face = (bug_y_min <= paddle_y_surface_max and \
                                      bug_y_min >= paddle_y_surface_min) 

        if x_overlap and y_overlap_with_paddle_face:
            self.env.round_concluded_by_skill = True
            self.deactivate(hit_paddle=True)
            return True
        return False


# --- BugSkillTrainingEnv (修改版) ---
class BugSkillTrainingEnv:
    def __init__(self, render_training=False, skill_params=None, opponent_ai_config=None, reward_shaping_config=None, config=None): # 參數列表保持不變
        pygame.init()

        self.skill_owner_is_player1 = True 
        self.render_size = 400
        self.paddle_height_normalized = 10 / self.render_size # 對手板子的高度
        self.time_scale = 1.0
        self.max_trail_length = 15

        # 技能擁有者 (P1, 假設在螢幕下方，未使用其板子)
        self.p1_skill_owner = PlayerState(player_identifier="p1_skill_owner", initial_x=0.5)
        # 技能目標對手 (P2, 板子在螢幕上方 Y=0 附近)
        self.p2_target_opponent = PlayerState(
            player_identifier="p2_target", 
            initial_paddle_width_normalized=60/self.render_size # 對手板子寬度
        )
        
        self.mock_env_for_skill = self._create_mock_env_for_skill()

        # 準備技能參數，包含新的初始位置範圍
        default_skill_params = {
            "bug_x_rl_move_speed": 0.03, 
            "bug_y_rl_move_speed": 0.03,
            "base_y_speed": 0.015,          # 蟲的基礎Y軸移動速度
            "duration_ms": 8000,            # 技能持續時間
            "bug_visual_radius_norm_factor": 0.0375, # 用於計算蟲的視覺/碰撞半徑
            "max_steps_for_impatience": 700, # 用於急躁懲罰計算的最大步數
            # ⭐️ 新增：蟲初始位置隨機範圍的預設值
            "initial_bug_x_min": 0.4,       # X軸最小隨機起始位置
            "initial_bug_x_max": 0.6,       # X軸最大隨機起始位置
            "initial_bug_y_min": 0.75,      # Y軸最小隨機起始位置 (離下方較遠)
            "initial_bug_y_max": 0.85       # Y軸最大隨機起始位置 (離下方較近)
        }
        self.current_skill_params = default_skill_params.copy() # 從預設值開始
        if skill_params is not None:
            self.current_skill_params.update(skill_params) # 使用傳入的 skill_params 覆蓋預設值

        # 根據 factor 計算實際的 bug_visual_radius_norm，並確保它在 current_skill_params 中
        # 這個半徑將傳遞給 SimplifiedSoulEaterBugSkill 用於碰撞和邊界檢測
        self.current_skill_params["bug_visual_radius_norm"] = self.current_skill_params.get("bug_visual_radius_norm_factor", 0.0375)
        
        # 實例化蟲技能，傳入包含所有必要參數（包括新位置參數）的 current_skill_params
        self.bug_skill_instance = SimplifiedSoulEaterBugSkill(
            self.mock_env_for_skill, self.p1_skill_owner, self.p2_target_opponent, self.current_skill_params
        )
        
# In training_environment.py
# Within BugSkillTrainingEnv class, inside __init__ method:

        # ... (之前的 skill_params 和 bug_skill_instance 初始化) ...
        
        # 初始化對手板子 AI
        self.opponent_agent = None
        self.use_dynamic_opponent = False
        self.opponent_paddle_move_speed = 0.03 
        if opponent_ai_config and opponent_ai_config.get("use_dynamic_opponent", False):
            self.use_dynamic_opponent = True
            
            model_filename_config_value = opponent_ai_config.get("model_filename") # e.g., "level2.pth" or "*RANDOM*"
            opponent_model_dir_name = opponent_ai_config.get("opponent_model_directory", "opponent_models/")
            
            self.opponent_paddle_move_speed = opponent_ai_config.get("paddle_move_speed", 0.03)
            
            chosen_model_filename = None # 將要實際載入的模型檔案名

            if model_filename_config_value == "*RANDOM*":
                # ⭐️ 實現隨機選取模型的邏輯
                try:
                    # current_env_dir 是 training_environment.py 所在的目錄 (soul_eater_trainer/)
                    # opponent_model_dir_path 是 soul_eater_trainer/opponent_models/
                    opponent_model_dir_path = os.path.join(current_env_dir, opponent_model_dir_name) # current_env_dir 在檔案頂部定義
                    
                    if os.path.exists(opponent_model_dir_path) and os.path.isdir(opponent_model_dir_path):
                        # 列出所有 .pth 檔案
                        pth_files = [f for f in os.listdir(opponent_model_dir_path) if f.endswith(".pth")]
                        if pth_files:
                            chosen_model_filename = random.choice(pth_files)
                            print(f"Randomly selected opponent AI model: {chosen_model_filename} from {opponent_model_dir_path}")
                        else:
                            print(f"WARNING: No .pth files found in opponent model directory: {opponent_model_dir_path}. Opponent will be static or random fallback.")
                            self.use_dynamic_opponent = False # 無法選取，則不使用動態對手
                    else:
                        print(f"WARNING: Opponent model directory not found: {opponent_model_dir_path}. Opponent will be static or random fallback.")
                        self.use_dynamic_opponent = False # 目錄不存在，則不使用動態對手
                except Exception as e:
                    print(f"Error during random opponent model selection: {e}. Opponent will be static or random fallback.")
                    self.use_dynamic_opponent = False
            else:
                # 如果不是 "*RANDOM*"，則使用配置中指定的檔案名
                chosen_model_filename = model_filename_config_value

            # 只有在成功選擇了模型檔案名 (且 use_dynamic_opponent 仍為 True) 時才繼續載入
            if self.use_dynamic_opponent and chosen_model_filename:
                # 組合完整的模型路徑 (與第二點修改的邏輯相同)
                # full_model_path_to_load 是 "opponent_models/chosen_model_filename.pth"
                full_model_path_to_load = os.path.join(opponent_model_dir_name, chosen_model_filename)

                opp_qnet_input_dim = opponent_ai_config.get("qnet_input_dim", 7) 
                opp_qnet_output_dim = opponent_ai_config.get("qnet_output_dim", 3) 
                
                self.opponent_agent = BugDQNAgent(state_size=opp_qnet_input_dim, action_size=opp_qnet_output_dim, seed=random.randint(0,10000))
                
                if self.opponent_agent.load(full_model_path_to_load): 
                    print(f"Successfully loaded opponent paddle AI model from: {full_model_path_to_load}")
                else:
                    print(f"WARNING: Could not load opponent paddle AI model from '{full_model_path_to_load}'. Opponent will be static or random fallback.")
                    self.opponent_agent = None 
                    self.use_dynamic_opponent = False 
            elif self.use_dynamic_opponent and not chosen_model_filename: # 如果設定為動態但最終沒有選到模型
                print(f"WARNING: 'use_dynamic_opponent' is true, but no opponent model could be chosen (model_filename: {model_filename_config_value}). Opponent will be static or random fallback.")
                self.use_dynamic_opponent = False

        # 讀取並儲存獎勵塑造參數 (這部分來自之前的修改，保持完整)
        self.rsc = reward_shaping_config if reward_shaping_config else {} 
        self.step_penalty = self.rsc.get('step_penalty', -0.01)
        self.score_reward = self.rsc.get('score_reward', 10.0)
        self.hit_paddle_penalty = self.rsc.get('hit_paddle_penalty', -5.0)
        self.duration_expired_penalty = self.rsc.get('duration_expired_penalty', -1.0)

        self.enable_movement_reward = self.rsc.get('enable_movement_reward', False)
        self.movement_reward_factor = self.rsc.get('movement_reward_factor', 0.005)
        
        self.enable_impatience_penalty = self.rsc.get('enable_impatience_penalty', False)
        self.impatience_m = self.rsc.get('impatience_penalty_m', 0.001)
        self.impatience_n = self.rsc.get('impatience_penalty_n', 0.05)
        self.impatience_k = self.rsc.get('impatience_penalty_k', 2.0)
        # max_steps_for_impatience 從 current_skill_params 獲取 (因為它是從 train.py 傳入 skill_params 的)
        self.max_steps_for_impatience = self.current_skill_params.get('max_steps_for_impatience', 700)

        self.enable_baseline_progress_reward = self.rsc.get('enable_baseline_progress_reward', False)
        self.baseline_progress_reward_factor = self.rsc.get('baseline_progress_reward_factor', 0.3)

        self.enable_paddle_avoidance_penalty = self.rsc.get('enable_paddle_avoidance_penalty', False)
        self.paddle_avoidance_penalty_factor = self.rsc.get('paddle_avoidance_penalty_factor', -0.2) 
        self.paddle_danger_zone_y_threshold = self.rsc.get('paddle_danger_zone_y_threshold', 0.15)

        # 用於計算獎勵的先前狀態變數
        # 這些值將在 reset() 方法中，於 bug_skill_instance.activate() 之後被正確更新
        self.previous_bug_x = self.current_skill_params.get("initial_bug_x_min", 0.5) 
        self.previous_bug_y = self.current_skill_params.get("initial_bug_y_max", 0.8) 
        self.current_steps_in_episode = 0

        # 渲染相關設定
        self.render_training = render_training
        self.screen = None
        self.clock = None
        self.font = None
        self.bug_image_render = None # 渲染用的蟲圖片表面
        if self.render_training:
            self.screen = pygame.display.set_mode((self.render_size, self.render_size + 100)) # 遊戲區域 +下方資訊區域
            pygame.display.set_caption("Bug Skill Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24) # 使用預設字體
            
            # 嘗試載入蟲的圖片用於渲染
            try:
                bug_image_path = "placeholder_bug.png" # 確保此圖片存在於 training_environment.py 同目錄下
                full_image_path = os.path.join(current_env_dir, bug_image_path) # current_env_dir 在檔案開頭定義
                if os.path.exists(full_image_path):
                    raw_bug_surf = pygame.image.load(full_image_path).convert_alpha()
                    # 使用 current_skill_params 中的 bug_visual_radius_norm (不是 factor) 計算渲染直徑
                    render_diameter = int(self.current_skill_params["bug_visual_radius_norm"] * 2 * self.render_size)
                    if render_diameter > 0:
                         self.bug_image_render = pygame.transform.smoothscale(raw_bug_surf, (render_diameter, render_diameter))
                    else:
                        print(f"Warning: Calculated bug render_diameter is {render_diameter}. Bug image not scaled.")
                else:
                    print(f"Warning: Bug image '{bug_image_path}' not found at '{full_image_path}'. Will use fallback rendering.")
            except Exception as e:
                print(f"Error setting up bug image: {e}. Will use fallback rendering.")


    def _create_mock_env_for_skill(self):
        mock_env = type('MockEnv', (object,), {})()
        mock_env.player1 = self.p1_skill_owner 
        mock_env.opponent = self.p2_target_opponent
        mock_env.render_size = self.render_size
        mock_env.paddle_height_normalized = self.paddle_height_normalized
        mock_env.time_scale = self.time_scale
        mock_env.max_trail_length = self.max_trail_length
        mock_env.ball_x = 0.5 # 蟲的 X (由技能控制)
        mock_env.ball_y = 0.5 # 蟲的 Y (由技能控制)
        # ⭐️ 新增：模擬遊戲球的速度和旋轉，供板子 AI 觀察 (蟲技能本身不使用這些)
        mock_env.ball_vx = 0.0 
        mock_env.ball_vy = 0.0
        mock_env.spin = 0.0
        mock_env.trail = []
        mock_env.round_concluded_by_skill = False
        return mock_env

    def reset(self):
        self.p2_target_opponent.x = random.uniform(
            self.p2_target_opponent.paddle_width_normalized / 2, 
            1.0 - self.p2_target_opponent.paddle_width_normalized / 2
        )
        self.p2_target_opponent.prev_x = self.p2_target_opponent.x
        self.p2_target_opponent.lives = 1 
        
        self.mock_env_for_skill.round_concluded_by_skill = False
        self.bug_skill_instance.activate() # ⭐️ 這一步會隨機設定 mock_env_for_skill.ball_x 和 ball_y

        self.mock_env_for_skill.ball_vx = 0.0
        self.mock_env_for_skill.ball_vy = 0.0 
        self.mock_env_for_skill.spin = random.uniform(-1, 1)

        # ⭐️ MODIFIED: 更新 previous_bug_x/y 為 activate() 後的隨機值
        self.previous_bug_x = self.mock_env_for_skill.ball_x
        self.previous_bug_y = self.mock_env_for_skill.ball_y
        self.current_steps_in_episode = 0
        
        return self.bug_skill_instance._get_bug_observation()

    def _get_opponent_paddle_observation(self):
        """為目標板子 (p2_target_opponent) 建構觀察。它在上方。"""
        # 從 p2_target_opponent (上方板子) 的視角來看：
        # 球的 Y 座標需要反轉
        obs_ball_y = 1.0 - self.mock_env_for_skill.ball_y
        # 球的 Y 速度需要反轉
        obs_ball_vy = -self.mock_env_for_skill.ball_vy # 使用 mock_env 的球速
        
        observation = [
            self.mock_env_for_skill.ball_x,       # 球 X
            obs_ball_y,                           # 球 Y (相對 p2)
            self.mock_env_for_skill.ball_vx,      # 球 VX
            obs_ball_vy,                          # 球 VY (相對 p2)
            self.p2_target_opponent.x,            # 我的板子 X (p2 的 X)
            self.p1_skill_owner.x,                # 對手板子 X (p1 的 X)
            self.mock_env_for_skill.spin          # 球旋轉
        ]
        return np.array(observation, dtype=np.float32)

    def step(self, bug_action_index):
        # --- 更新目標板子 (p2_target_opponent) 的位置 ---
        self.p2_target_opponent.prev_x = self.p2_target_opponent.x
        if self.use_dynamic_opponent and self.opponent_agent:
            opp_obs = self._get_opponent_paddle_observation()
            opp_action = self.opponent_agent.act(opp_obs, eps=0.05) # 給一點探索性
            # opp_action: 0=左, 1=不動, 2=右
            if opp_action == 0: # 左
                self.p2_target_opponent.x -= self.opponent_paddle_move_speed * self.time_scale
            elif opp_action == 2: # 右
                self.p2_target_opponent.x += self.opponent_paddle_move_speed * self.time_scale
            self.p2_target_opponent.x = np.clip(self.p2_target_opponent.x, 
                                                self.p2_target_opponent.paddle_width_normalized / 2, 
                                                1.0 - self.p2_target_opponent.paddle_width_normalized / 2)
        elif self.use_dynamic_opponent: # 如果設定要動態但模型載入失敗，可以隨機移動或追球
            # 簡易追球邏輯 (如果沒有AI模型)
            if self.mock_env_for_skill.ball_x < self.p2_target_opponent.x - 0.02:
                 self.p2_target_opponent.x -= self.opponent_paddle_move_speed * self.time_scale * 0.5
            elif self.mock_env_for_skill.ball_x > self.p2_target_opponent.x + 0.02:
                 self.p2_target_opponent.x += self.opponent_paddle_move_speed * self.time_scale * 0.5
            self.p2_target_opponent.x = np.clip(self.p2_target_opponent.x, 
                                                self.p2_target_opponent.paddle_width_normalized / 2, 
                                                1.0 - self.p2_target_opponent.paddle_width_normalized / 2)


        # --- 更新蟲的狀態 ---
        if not self.bug_skill_instance.active:
            obs = self.bug_skill_instance._get_bug_observation()
            return obs, -1, True, {'result': 'skill_inactive_timeout'}

        delta_x_norm, delta_y_norm = 0.0, 0.0
        y_direction_sign_to_target = -1.0 

        if bug_action_index == 0:
            delta_y_norm = y_direction_sign_to_target * self.bug_skill_instance.bug_y_rl_move_speed
        elif bug_action_index == 1:
            delta_y_norm = -y_direction_sign_to_target * self.bug_skill_instance.bug_y_rl_move_speed
        elif bug_action_index == 2:
            delta_x_norm = -self.bug_skill_instance.bug_x_rl_move_speed
        elif bug_action_index == 3:
            delta_x_norm = self.bug_skill_instance.bug_x_rl_move_speed
        
        if self.bug_skill_instance.base_y_speed != 0.0 and bug_action_index in [2, 3, 4]:
            if delta_y_norm == 0.0:
                delta_y_norm = y_direction_sign_to_target * self.bug_skill_instance.base_y_speed
        
        self.bug_skill_instance._apply_movement_and_constrain_bounds(delta_x_norm, delta_y_norm)
        self.bug_skill_instance._update_trail()

        # ⭐️ 更新模擬的球速 (讓板子AI的觀察有點變化，這裡是很簡化的模擬)
        # 實際上蟲的移動不直接產生球速，但為了讓板子AI能動，我們模擬一下
        self.mock_env_for_skill.ball_vx = delta_x_norm / self.time_scale if self.time_scale else 0
        self.mock_env_for_skill.ball_vy = delta_y_norm / self.time_scale if self.time_scale else 0
        # 模擬一點隨機旋轉
        if random.random() < 0.1:
            self.mock_env_for_skill.spin += random.uniform(-0.5, 0.5)
            self.mock_env_for_skill.spin = np.clip(self.mock_env_for_skill.spin, -10, 10)


        reward = -0.01 
        done = False
        info = {}

        if self.bug_skill_instance._check_bug_scored():
            reward += 10.0
            done = True
            info['result'] = 'scored'
        elif self.bug_skill_instance._check_bug_hit_paddle():
            reward -= 5.0 # 撞到板子懲罰更高
            done = True
            info['result'] = 'hit_paddle'
        
        if not done and (pygame.time.get_ticks() - self.bug_skill_instance.activated_time) >= self.bug_skill_instance.duration_ms:
            reward -= 1.0 
            done = True
            info['result'] = 'duration_expired'
            self.bug_skill_instance.deactivate()
        
        if not self.bug_skill_instance.active and not done:
            done = True
            if 'result' not in info: info['result'] = 'skill_deactivated_internally'

        next_observation_for_bug = self.bug_skill_instance._get_bug_observation() if self.bug_skill_instance.active else np.zeros_like(self.bug_skill_instance._get_bug_observation())
        
        return next_observation_for_bug, reward, done, info

    def render(self, agent_action=None, current_reward=0, episode_score=0, episode_num=0):
        if not self.render_training:
            return
        self.screen.fill((30,30,30))
        target_paddle = self.p2_target_opponent
        tp_x_px = int(target_paddle.x * self.render_size)
        tp_w_px = int(target_paddle.paddle_width_normalized * self.render_size)
        tp_h_px = int(self.paddle_height_normalized * self.render_size)
        tp_y_px = 0 
        pygame.draw.rect(self.screen, (200,0,0), (tp_x_px - tp_w_px//2, tp_y_px, tp_w_px, tp_h_px))

        if self.bug_image_render:
            bug_center_x_px = int(self.mock_env_for_skill.ball_x * self.render_size)
            bug_center_y_px = int(self.mock_env_for_skill.ball_y * self.render_size)
            bug_rect = self.bug_image_render.get_rect(center=(bug_center_x_px, bug_center_y_px))
            self.screen.blit(self.bug_image_render, bug_rect)
        else:
            bug_radius_px = int(self.bug_skill_instance.bug_visual_radius_norm * self.render_size)
            pygame.draw.circle(self.screen, (100,0,100), 
                               (int(self.mock_env_for_skill.ball_x * self.render_size), 
                                int(self.mock_env_for_skill.ball_y * self.render_size)), 
                               bug_radius_px)
        if len(self.mock_env_for_skill.trail) > 1:
            scaled_trail = [(int(x*self.render_size), int(y*self.render_size)) for x,y in self.mock_env_for_skill.trail]
            pygame.draw.lines(self.screen, (0,100,100), False, scaled_trail, 2)

        info_y_start = self.render_size + 10
        if agent_action is not None:
            action_map = {0: "FWD", 1: "BCK", 2: "LFT", 3: "RGT", 4: "STAY"}
            text_surface = self.font.render(f"Ep: {episode_num} Action: {action_map.get(agent_action, '?')} Rew: {current_reward:.2f} Score: {episode_score:.2f}", True, (255,255,255))
            self.screen.blit(text_surface, (10, info_y_start))
        
        target_lives_text = f"Target Lives: {self.p2_target_opponent.lives}"
        text_surface_score = self.font.render(target_lives_text, True, (255,255,255))
        self.screen.blit(text_surface_score, (10, info_y_start + 25))

        time_left_ms = self.bug_skill_instance.duration_ms - (pygame.time.get_ticks() - self.bug_skill_instance.activated_time)
        time_left_s = max(0, time_left_ms / 1000.0)
        time_text = f"Time Left: {time_left_s:.1f}s"
        text_surface_time = self.font.render(time_text, True, (255,255,255))
        self.screen.blit(text_surface_time, (10, info_y_start + 50))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_training:
            pygame.quit()