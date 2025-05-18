# soul_eater_trainer/test_trained_bug_effect.py
import pygame
import sys
import os
import numpy as np
import torch
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from training_environment import BugSkillTrainingEnv
from dqn_definitions import BugDQNAgent

# --- 載入設定檔 ---
def load_config(config_path="config/training_config.yaml"):
    full_config_path = os.path.join(current_dir, config_path)
    if not os.path.exists(full_config_path):
        print(f"ERROR: Config file not found at {full_config_path}")
        sys.exit(1)
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

def run_evaluation():
    eval_cfg = CONFIG.get('evaluation_settings', {}) # 使用 .get 提供預設空字典
    env_cfg = CONFIG.get('environment_settings', {})
    opponent_ai_cfg = env_cfg.get('opponent_ai_settings', {"use_dynamic_opponent": False}) # ⭐️ 讀取對手AI設定

    model_filename = eval_cfg.get('model_to_evaluate', "bug_agent_ep2500.pth")
    
    skill_params_from_config = env_cfg.get('skill_params', {}).copy()
    skill_params_for_eval = skill_params_from_config
    skill_params_for_eval["duration_ms"] = eval_cfg.get('skill_duration_ms_eval', skill_params_from_config.get("duration_ms", 8000))
    # 確保 bug_visual_radius_norm 被設定
    if "bug_visual_radius_norm_factor" in skill_params_for_eval and "bug_visual_radius_norm" not in skill_params_for_eval :
        skill_params_for_eval["bug_visual_radius_norm"] = skill_params_for_eval.get("bug_visual_radius_norm_factor", 0.0375)


    # ⭐️ 初始化環境時傳入 opponent_ai_cfg
    env = BugSkillTrainingEnv(
        render_training=True, 
        skill_params=skill_params_for_eval,
        opponent_ai_config=opponent_ai_cfg 
    )
    
    if not pygame.display.get_init():
        pygame.display.init()
        pygame.font.init()
        env.screen = pygame.display.set_mode((env.render_size, env.render_size + 100))
        pygame.display.set_caption(f"Trained Bug Evaluation (Opponent AI: {opponent_ai_cfg.get('use_dynamic_opponent')})")
        env.clock = pygame.time.Clock()
        if env.font is None:
             env.font = pygame.font.Font(None, 24)

    initial_obs_for_size = env.reset()
    state_size = initial_obs_for_size.shape[0]
    action_size = 5
    
    agent_hyperparams_cfg = CONFIG.get('agent_hyperparameters', {})
    agent = BugDQNAgent(
        state_size=state_size, 
        action_size=action_size, 
        seed=0,
        lr=float(agent_hyperparams_cfg.get('lr', 5e-4)),
    )

    print(f"Attempting to load bug agent model: {model_filename}")
    if agent.load(model_filename):
        print(f"Successfully loaded trained bug agent model: {model_filename}")
    else:
        print(f"ERROR: Could not load bug agent model '{model_filename}'.")
        env.close(); pygame.quit(); return

    print(f"Opponent AI dynamic: {env.use_dynamic_opponent}")
    if env.use_dynamic_opponent and env.opponent_agent:
        print(f"Opponent AI model: {opponent_ai_cfg.get('model_filename', 'N/A')} for paddle control.")


    running = True
    obs = env.reset() 
    episode_score = 0
    episode_num = 1
    current_reward_display = 0
    
    print("\nEvaluating trained bug model...")
    print("  R:           Reset Environment")
    print("  ESC:         Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset(); episode_score = 0; episode_num += 1; current_reward_display = 0
                    continue 

        action = agent.act(obs, eps=0.0) 
        next_obs, reward, done, info = env.step(action)
        episode_score += reward
        current_reward_display = reward
        obs = next_obs

        env.render(
            agent_action=action, 
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
                            wait_for_input = False
                env.clock.tick(30)
    env.close()
    pygame.quit()
    print("Trained model evaluation finished.")

if __name__ == '__main__':
    eval_cfg = CONFIG.get('evaluation_settings', {})
    model_to_evaluate_from_config = eval_cfg.get('model_to_evaluate', "bug_agent_ep3200.pth")
    
    trainer_dir = os.path.dirname(os.path.abspath(__file__))
    model_path_to_check = os.path.join(trainer_dir, "trained_models", model_to_evaluate_from_config)

    if not os.path.exists(model_path_to_check) and not eval_cfg.get("allow_no_model_for_manual_test", False):
        print(f"ERROR: Bug agent model file '{model_to_evaluate_from_config}' (from config) not found.")
    else:
        run_evaluation()