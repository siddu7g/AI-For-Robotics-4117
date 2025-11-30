# policy_grad.py
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def discount_returns(rewards, masks, gamma, last_value):
    """
    Compute discounted returns R_t = r_t + gamma * R_{t+1} (backwards).
    masks[t] should be 1.0 if the next state exists (no terminal), 0.0 if terminal.
    """
    returns = []
    R = float(last_value)
    # iterate reversed lists
    for r, m in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * m
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def compute_losses(R, ep_rewards, ep_values, ep_action_log_probs, masks, gamma, tau, algo):

    # Keep rewards/masks as new tensors (safe – they are not from the model)
    rewards = torch.tensor(ep_rewards, dtype=torch.float32)
    masks   = torch.tensor(masks,   dtype=torch.float32)

    # BUT DO NOT CREATE NEW TENSORS FOR VALUES OR LOG PROBS
    values = torch.stack(ep_values)                # <-- KEEP grad
    log_probs = torch.stack(ep_action_log_probs)   # <-- KEEP grad

    # 1. Compute discounted returns
    returns = discount_returns(rewards.tolist(), masks.tolist(), gamma, R)

    # 2. Compute advantages
    advantages = returns - values.detach()         # baseline detached, not the values used in loss

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 3. Policy loss (must use original log_probs with grad)
    policy_loss = -(log_probs * advantages).sum()

    # 4. Value loss (use model-connected values)
    value_loss = F.mse_loss(values, returns)

    return policy_loss, value_loss
