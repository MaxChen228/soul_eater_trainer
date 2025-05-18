# soul_eater_trainer/train.py
import torch
import numpy as np
from collections import deque
import os
import sys
import yaml 

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dqn_definitions import BugDQNAgent
from training_environment import BugSkillTrainingEnv # BugSkillTrainingEnv 現在接受 opponent_ai_config

def load_config(config_path="config/training_config.yaml"):
    full_config_path = os.path.join(current_dir, config_path)
    if not os.path.exists(full_config_path):
        print(f"ERROR: Config file not found at {full_config_path}")
        sys.exit(1)
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

def train_dqn_agent():
    train_cfg = CONFIG['training_settings']
    agent_cfg = CONFIG['agent_hyperparameters']
    env_cfg = CONFIG['environment_settings']
    
    # ⭐️ 獲取對手 AI 設定
    opponent_ai_cfg = env_cfg.get('opponent_ai_settings', {"use_dynamic_opponent": False}) # 提供預設值以防萬一

    n_episodes = train_cfg['n_episodes']
    max_t_per_episode = train_cfg['max_t_per_episode']
    eps_start = float(train_cfg['eps_start'])
    eps_end = float(train_cfg['eps_end'])
    eps_decay = float(train_cfg['eps_decay'])
    load_checkpoint = train_cfg['load_checkpoint']
    checkpoint_to_load = train_cfg['checkpoint_to_load']
    save_checkpoint_every = train_cfg['save_checkpoint_every']
    final_model_name = train_cfg['final_model_name']
    render_each_n_episodes = train_cfg.get('render_each_n_episodes', 0)

    skill_params_from_config = env_cfg['skill_params'].copy()
    skill_params_from_config["bug_visual_radius_norm"] = skill_params_from_config.get("bug_visual_radius_norm_factor", 0.0375)

    # ⭐️ 傳入 opponent_ai_cfg 給環境
    env = BugSkillTrainingEnv(
        render_training=False, 
        skill_params=skill_params_from_config,
        opponent_ai_config=opponent_ai_cfg 
    )

    initial_obs = env.reset()
    state_size = initial_obs.shape[0]
    action_size = 5
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Using dynamic opponent: {env.use_dynamic_opponent}")
    if env.use_dynamic_opponent and env.opponent_agent:
        print(f"Opponent AI model loaded for paddle control.")


    agent = BugDQNAgent(state_size=state_size, action_size=action_size, seed=0,
                        lr=float(agent_cfg['lr']), 
                        buffer_size=int(agent_cfg['buffer_size']), 
                        batch_size=int(agent_cfg['batch_size']), 
                        gamma=float(agent_cfg['gamma']), 
                        tau=float(agent_cfg['tau']),
                        update_every=int(agent_cfg['update_every']), 
                        target_update_every=int(agent_cfg['target_update_every']))
    
    if load_checkpoint:
        if agent.load(checkpoint_to_load):
            print(f"Loaded checkpoint: {checkpoint_to_load}")
        else:
            print(f"Could not load checkpoint {checkpoint_to_load}, starting from scratch.")

    scores_deque = deque(maxlen=100)
    all_scores = []
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        current_episode_render = (render_each_n_episodes > 0 and i_episode % render_each_n_episodes == 0)
        if current_episode_render and not env.render_training:
            env.render_training = True
            if not hasattr(env, 'screen') or env.screen is None:
                 print("Re-initializing env for rendering this episode...")
                 env = BugSkillTrainingEnv(
                     render_training=True, 
                     skill_params=skill_params_from_config,
                     opponent_ai_config=opponent_ai_cfg # ⭐️ 重新初始化時也要傳入
                    )
        
        state = env.reset()
        episode_score = 0
        
        for t in range(max_t_per_episode):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_score += reward
            
            if current_episode_render:
                env.render(agent_action=action, current_reward=reward, episode_score=episode_score, episode_num=i_episode)
            
            if done:
                break
        
        scores_deque.append(episode_score)
        all_scores.append(episode_score)
        eps = max(eps_end, eps_decay * eps)

        avg_score_100 = np.mean(scores_deque)
        print(f'\rEpisode {i_episode}\tScore: {episode_score:.2f}\tAvg Score (100 ep): {avg_score_100:.2f}\tEpsilon: {eps:.3f}', end="")
        
        if i_episode % save_checkpoint_every == 0:
            print(f'\rEpisode {i_episode}\tScore: {episode_score:.2f}\tAvg Score (100 ep): {avg_score_100:.2f}\tEpsilon: {eps:.3f}')
            agent.save(f"bug_agent_ep{i_episode}.pth")
        
        if current_episode_render and env.render_training:
            env.render_training = False

    agent.save(final_model_name)
    print(f"\nTraining complete. Final model saved as {final_model_name} in trained_models/ folder.")
    
    env.close()
    return all_scores

if __name__ == '__main__':
    scores_history = train_dqn_agent() # ⭐️ 直接呼叫，不傳遞任何參數
