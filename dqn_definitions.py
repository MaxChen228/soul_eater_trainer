# soul_eater_trainer/dqn_definitions.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os
import math

# --- QNet (從 pong-soul/game/ai_agent.py 複製並簡化) ---
# NoisyLinear 和 QNet 的定義 (如果您的 QNet 使用 NoisyLinear)
# 如果您的 QNet 是一個簡單的 nn.Sequential，則直接使用那個定義。
# 根據您提供的 game/ai_agent.py，QNet 結構如下：
class NoisyLinear(nn.Module): # 從您的 game/ai_agent.py 複製
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        def scale_noise(size):
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign().mul_(x.abs().sqrt_())
        eps_in  = scale_noise(self.in_features)
        eps_out = scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)

class QNet(nn.Module): # 從您的 game/ai_agent.py 複製
    def __init__(self, input_dim=7, output_dim=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fc_V = NoisyLinear(64, 1)
        self.fc_A = NoisyLinear(64, output_dim)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        h = self.features(x)
        V = self.fc_V(h)
        A = self.fc_A(h)
        return V + (A - A.mean(dim=1, keepdim=True))

# --- ReplayBuffer (從 pong-soul/rl_training/train_bug_rl.py 複製) ---
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- BugDQNAgent (從 pong-soul/rl_training/train_bug_rl.py 複製並修改 save/load 路徑) ---
class BugDQNAgent:
    def __init__(self, state_size, action_size, seed, lr=5e-4, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, update_every=4, target_update_every=100):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.qnetwork_local = QNet(state_size, action_size).to(self.device)
        self.qnetwork_target = QNet(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.target_update_every = target_update_every
        
        self.t_step = 0
        self.target_update_step = 0

        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, next_state, reward, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences, self.gamma)

        self.target_update_step = (self.target_update_step + 1) % self.target_update_every
        if self.target_update_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if hasattr(self.qnetwork_local, 'reset_noise'): # For NoisyNet
            self.qnetwork_local.reset_noise()
            self.qnetwork_target.reset_noise()


        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, next_states, rewards, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename="bug_agent_checkpoint.pth"):
        # 儲存到相對於此腳本的 'trained_models' 子目錄
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_dir = os.path.join(current_script_dir, "trained_models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_save_path = os.path.join(model_save_dir, filename)
        
        torch.save({
            'model_state_dict': self.qnetwork_local.state_dict(),
        }, model_save_path)
        print(f"Model saved to {model_save_path}")


    def load(self, filename_or_path="bug_agent_checkpoint.pth"):
        current_script_dir = os.path.dirname(os.path.abspath(__file__)) # dqn_definitions.py 所在的目錄
        trainer_project_root = os.path.dirname(current_script_dir) # 退回到 soul_eater_trainer/

        model_load_path = ""

        # ⭐️ 修改：判斷傳入的是否包含路徑
        if os.path.sep in filename_or_path or (os.altsep and os.altsep in filename_or_path):
            # 如果包含路徑分隔符，則認為是相對於專案根目錄的路徑
            # 例如 "opponent_models/level2.pth"
            model_load_path = os.path.join(trainer_project_root, filename_or_path)
            print(f"DEBUG: Loading model using provided path relative to project root: {model_load_path}")
        else:
            # 如果不包含路徑分隔符，則預設在 "trained_models/" 目錄下查找 (原行為)
            # 例如 "bug_agent_ep1000.pth"
            model_load_path = os.path.join(trainer_project_root, "trained_models", filename_or_path)
            print(f"DEBUG: Loading model from default 'trained_models' directory: {model_load_path}")

        # 移除舊的 alt_model_load_path 邏輯，因為現在路徑判斷更清晰了
        # if not os.path.exists(model_load_path):
        #     # ... (舊的 alt path 檢查) ...
        #     return False

        if not os.path.exists(model_load_path):
            print(f"Model file not found at {model_load_path}")
            # ⭐️ 新增一個嘗試：如果 filename_or_path 是一個絕對路徑且存在
            if os.path.isabs(filename_or_path) and os.path.exists(filename_or_path):
                print(f"DEBUG: Attempting to load model from absolute path: {filename_or_path}")
                model_load_path = filename_or_path
            else:
                # 嘗試另一種可能：如果 filename_or_path 是相對於 dqn_definitions.py 所在目錄的 trained_models 或 opponent_models
                # 這種情況比較複雜，優先級較低。主要依賴 trainer_project_root。
                # 為了簡潔，我們先不加過於複雜的備用路徑查找。
                # 上面的邏輯應該能覆蓋主要情況。
                return False


        if os.path.exists(model_load_path):
            try:
                checkpoint = torch.load(model_load_path, map_location=self.device)
                
                state_dict_to_load = None
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict_to_load = checkpoint['model_state_dict']
                        # print(f"Found state_dict under 'model_state_dict' key in {filename_or_path}")
                    elif 'modelB' in checkpoint: 
                        state_dict_to_load = checkpoint['modelB']
                        # print(f"Found state_dict under 'modelB' key in {filename_or_path}")
                    elif 'model' in checkpoint: 
                        state_dict_to_load = checkpoint['model']
                        # print(f"Found state_dict under 'model' key in {filename_or_path}")
                    else:
                        # 檢查是否 checkpoint 本身就是 state_dict
                        is_likely_state_dict = True
                        for key in checkpoint.keys():
                            if not isinstance(checkpoint[key], torch.Tensor):
                                is_likely_state_dict = False
                                break
                        if is_likely_state_dict:
                            state_dict_to_load = checkpoint
                            # print(f"Checkpoint for {filename_or_path} appears to be a raw state_dict itself.")
                        else:
                            print(f"ERROR: Expected keys ('model_state_dict', 'modelB', 'model') not found in checkpoint dictionary for {filename_or_path}. Keys found: {list(checkpoint.keys())}")
                            return False
                elif isinstance(checkpoint, (torch.Tensor, OrderedDict)): # type: ignore # (OrderedDict often from older torch saves)
                    # 假設 checkpoint 本身就是 state_dict
                    state_dict_to_load = checkpoint
                    # print(f"Checkpoint for {filename_or_path} appears to be a raw state_dict itself (OrderedDict or Tensor).")

                else:
                    print(f"ERROR: Checkpoint for {filename_or_path} is not a dictionary or recognized state_dict format. Type: {type(checkpoint)}")
                    return False

                if state_dict_to_load:
                    is_new_architecture = any(k.startswith(("features.", "fc_V.", "fc_A.")) for k in state_dict_to_load.keys())
                    
                    if is_new_architecture:
                        # print(f"Loading NEW QNet architecture for {filename_or_path} (features, fc_V, fc_A).")
                        self.qnetwork_local.load_state_dict(state_dict_to_load, strict=True)
                    else: 
                        # print(f"Detected OLD QNet architecture for {filename_or_path} (fc.0, fc.2, fc.4). Performing key mapping.")
                        mapped_state_dict = {}
                        has_fc4 = "fc.4.weight" in state_dict_to_load and "fc.4.bias" in state_dict_to_load

                        for k, v in state_dict_to_load.items():
                            if k.startswith("fc.0."): mapped_state_dict[k.replace("fc.0.", "features.0.")] = v
                            elif k.startswith("fc.2."): mapped_state_dict[k.replace("fc.2.", "features.2.")] = v
                        
                        if has_fc4:
                            mapped_state_dict["fc_A.weight_mu"] = state_dict_to_load["fc.4.weight"]
                            mapped_state_dict["fc_A.bias_mu"] = state_dict_to_load["fc.4.bias"]
                            mapped_state_dict["fc_V.weight_mu"] = state_dict_to_load["fc.4.weight"].mean(dim=0, keepdim=True)
                            mapped_state_dict["fc_V.bias_mu"] = state_dict_to_load["fc.4.bias"].mean().unsqueeze(0)
                            # print(f"Mapped fc.4 from {filename_or_path} to Dueling heads (fc_A, fc_V).")
                        else:
                            print(f"WARNING: Old architecture state_dict for {filename_or_path} is missing 'fc.4.weight' or 'fc.4.bias'. Cannot map to Dueling heads if it was intended for them.")
                        
                        self.qnetwork_local.load_state_dict(mapped_state_dict, strict=False)
                        # print(f"Loaded mapped state_dict for {filename_or_path} with strict=False.")

                    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict()) 
                    self.qnetwork_local.to(self.device)
                    self.qnetwork_target.to(self.device)
                    print(f"Model {filename_or_path} (loaded from {model_load_path}) successfully loaded and mapped to device: {self.device}")
                    return True
                else:
                    print(f"ERROR: Could not extract state_dict from checkpoint for {filename_or_path}.")
                    return False

            except Exception as e:
                print(f"Error loading model from {model_load_path}: {e}")
                import traceback
                traceback.print_exc()
                return False
        else: 
            # 這個分支理論上應該在更早的 os.path.exists(model_load_path) 檢查中被捕獲
            print(f"Model file not found at {model_load_path} (final check in load method).")
            return False