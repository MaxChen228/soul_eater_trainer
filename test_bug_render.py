# soul_eater_trainer/test_bug_render.py
import pygame
import sys
import os
import numpy as np
import yaml # ⭐️ 新增 yaml

# 確保可以 import 同目錄下的模組
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from training_environment import BugSkillTrainingEnv

# --- ⭐️ 新增：載入設定檔 ---
def load_config(config_path="config/training_config.yaml"):
    full_config_path = os.path.join(current_dir, config_path)
    if not os.path.exists(full_config_path):
        print(f"WARNING: Config file not found at {full_config_path}. Using default parameters for opponent AI.")
        return {} # 返回空字典，讓後續代碼使用預設值
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

def run_render_test():
    # ⭐️ 從 CONFIG 讀取環境和對手 AI 設定
    env_cfg = CONFIG.get('environment_settings', {})
    opponent_ai_cfg = env_cfg.get('opponent_ai_settings', {"use_dynamic_opponent": False}) # 提供預設

    # 技能參數 (可以優先從設定檔讀取，若無則使用預設)
    skill_params_from_config = env_cfg.get('skill_params', {}).copy()
    skill_params_for_testing = { # 先給定一些預設值
        "bug_x_rl_move_speed": 0.04,
        "bug_y_rl_move_speed": 0.04,
        "base_y_speed": 0.02,
        "duration_ms": 20000, 
        "bug_visual_radius_norm_factor": 0.0375 
    }
    skill_params_for_testing.update(skill_params_from_config) # 用設定檔的值覆蓋預設
    # 確保 bug_visual_radius_norm 被設定
    if "bug_visual_radius_norm_factor" in skill_params_for_testing and "bug_visual_radius_norm" not in skill_params_for_testing:
        skill_params_for_testing["bug_visual_radius_norm"] = skill_params_for_testing.get("bug_visual_radius_norm_factor", 0.0375)


    # ⭐️ 初始化環境時傳入 opponent_ai_cfg
    env = BugSkillTrainingEnv(
        render_training=True, 
        skill_params=skill_params_for_testing,
        opponent_ai_config=opponent_ai_cfg
    )
    
    if not pygame.display.get_init():
        pygame.display.init()
        pygame.font.init()
        env.screen = pygame.display.set_mode((env.render_size, env.render_size + 100))
        env.clock = pygame.time.Clock()
        if env.font is None:
             env.font = pygame.font.Font(None, 24)
    pygame.display.set_caption(f"Manual Bug Render Test (Opponent AI: {opponent_ai_cfg.get('use_dynamic_opponent')})")


    running = True
    current_action = 4 
    obs = env.reset()
    episode_score = 0
    episode_num = 1
    current_reward_display = 0

    print("Manual Bug Render Test Controls:")
    print("  UP_ARROW:    Move Bug Forward")
    print("  DOWN_ARROW:  Move Bug Backward")
    print("  LEFT_ARROW:  Move Bug Left")
    print("  RIGHT_ARROW: Move Bug Right")
    print("  SPACE:       Bug Stays")
    print("  R:           Reset Environment")
    print("  ESC:         Quit")
    print(f"Opponent AI dynamic: {env.use_dynamic_opponent}")
    if env.use_dynamic_opponent and env.opponent_agent:
        print(f"Opponent AI model: {opponent_ai_cfg.get('model_filename', 'N/A')} for paddle control.")


    while running:
        manual_action_taken_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                manual_action_taken_this_frame = True
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_UP: current_action = 0
                elif event.key == pygame.K_DOWN: current_action = 1
                elif event.key == pygame.K_LEFT: current_action = 2
                elif event.key == pygame.K_RIGHT: current_action = 3
                elif event.key == pygame.K_SPACE: current_action = 4
                elif event.key == pygame.K_r:
                    obs = env.reset(); episode_score = 0; episode_num += 1; current_reward_display = 0
                    current_action = 4
                    continue 

        next_obs, reward, done, info = env.step(current_action)
        episode_score += reward
        current_reward_display = reward
        obs = next_obs

        env.render(
            agent_action=current_action, 
            current_reward=current_reward_display, 
            episode_score=episode_score, 
            episode_num=episode_num
        )

        if done:
            print(f"Episode {episode_num} finished. Reason: {info.get('result', 'Unknown')}. Final score: {episode_score:.2f}")
            wait_for_input = True
            while wait_for_input and running:
                for event_done in pygame.event.get():
                    if event_done.type == pygame.QUIT: running = False; wait_for_input = False
                    if event_done.type == pygame.KEYDOWN:
                        if event_done.key == pygame.K_ESCAPE: running = False; wait_for_input = False
                        elif event_done.key == pygame.K_r:
                            obs = env.reset(); episode_score = 0; episode_num +=1; current_reward_display = 0
                            current_action = 4
                            wait_for_input = False
                env.clock.tick(30) 
    env.close()
    pygame.quit()
    print("Manual render test finished.")

if __name__ == '__main__':
    run_render_test()