import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict
from operator import itemgetter
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os

# Helper function for RoPE (updated)
def rotary_emb(x, freqs_cis):
    """Applies RoPE to the input tensor using complex multiplication."""
    # x: [..., seq_len, dim]
    # freqs_cis: [seq_len, dim // 2] (complex) - should be on the same device as x
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2) # [..., seq_len, dim // 2, 2]
    x_complex = torch.view_as_complex(x_reshaped)      # [..., seq_len, dim // 2]

    # Ensure freqs_cis is on the correct device
    freqs_cis = freqs_cis.to(x_complex.device)

    # Perform complex multiplication (element-wise)
    x_rotated_complex = x_complex * freqs_cis # [..., seq_len, dim // 2]

    # Convert back to real and reshape
    x_rotated_real = torch.view_as_real(x_rotated_complex) # [..., seq_len, dim // 2, 2]
    x_out = x_rotated_real.flatten(-2)                     # [..., seq_len, dim]

    return x_out.type_as(x)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: torch.device = 'cpu'):
    """Precomputes frequency components for RoPE and returns as complex numbers."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs) # Shape: [end, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Shape: [end, dim // 2], complex64
    return freqs_cis

class SentenceRewardModel(nn.Module):
    def __init__(
        self,
        base_model : PreTrainedModel,
        tokenizer : PreTrainedTokenizer,
        r_embed_dim : int = 128,
        causal_sentence_mask : bool = False,
        weighted_sum : bool = True,
        penalty_coef : float = 0.0,
        use_rope: bool = True, # Add flag to control RoPE
        rope_theta: float = 10000.0, # RoPE parameter
        max_rope_sentences: int = 100 # Max sentences for precomputation
    ):
        super().__init__()
        self.config = base_model.config
        self.hidden_size = None
        if hasattr(self.config, 'word_embed_proj_dim'):
            self.hidden_size = self.config.word_embed_proj_dim
        elif hasattr(self.config, 'hidden_size'):
            self.hidden_size = self.config.hidden_size
        else:
            self.hidden_size = self.config.n_embed

        self.tokenizer = tokenizer
        self.r_embed_dim = r_embed_dim
        self._causal_rm_mask = causal_sentence_mask
        self._weighted_sum = weighted_sum
        self.penalty_coef = penalty_coef
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_rope_sentences = max_rope_sentences

        # self._causal_rm_mask = True
        self.r_num_head = 1
        self.base_model = base_model

        # Reward attention
        if self._weighted_sum:
            self.q_proj = nn.Linear(self.hidden_size, self.r_embed_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.r_embed_dim)
            self.r_proj = nn.Linear(self.hidden_size, 1)
        else:
            self.r_proj = nn.Linear(self.hidden_size, 1)

        self.r_out_fn = nn.Identity()

        # Precompute RoPE frequencies if enabled
        if self.use_rope and self._weighted_sum:
            # Detach helps if base_model is already on GPU but we compute this on CPU first
            freqs_cis = precompute_freqs_cis(
                self.r_embed_dim, self.max_rope_sentences, self.rope_theta
            ).detach()
            # Register as buffer to automatically handle device placement
            self.register_buffer('rope_freqs_cis', freqs_cis, persistent=False)
        else:
            self.register_buffer('rope_freqs_cis', None, persistent=False)

        if False: # Keep your existing mu/var init logic
            self.init_mu_var('cpu')

    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()

    def _get_rewards(self, hidden_states : torch.Tensor, split_masks : torch.Tensor):
        dtype = hidden_states.dtype
        device = hidden_states.device # Get device from input tensor

        if self._weighted_sum:
            q_all : torch.Tensor = self.q_proj(hidden_states)
            k_all : torch.Tensor = self.k_proj(hidden_states)
        else:
             q_all = torch.zeros_like(hidden_states[..., :self.r_embed_dim])
             k_all = torch.zeros_like(hidden_states[..., :self.r_embed_dim])

        v : torch.Tensor = self.r_proj(hidden_states)
        v = self.r_out_fn(v)

        bs, seq_len, _ = hidden_states.shape

        sentence_rewards = []
        sequence_rewards = []
        sentence_nums = []

        # RoPE frequencies are now precomputed in self.rope_freqs_cis (on the correct device)

        for i in range(bs):
            split_tokens = torch.where(split_masks[i] == 1)[0]
            num_sentences = split_tokens.shape[0]

            if num_sentences == 0:
                 vs = v[i, -1:]
                 seq_reward = vs.sum().unsqueeze(0)
                 sentence_rewards.append(vs.reshape((-1,)))
                 sequence_rewards.append(seq_reward)
                 sentence_nums.append(1) # Or 0 depending on desired behavior
                 continue

            # Ensure we don't exceed precomputed RoPE length
            if self.use_rope and self._weighted_sum and num_sentences > self.max_rope_sentences:
                 raise ValueError(
                    f"Number of sentences ({num_sentences}) exceeds precomputed RoPE length "
                    f"({self.max_rope_sentences}). Increase max_rope_sentences during init."
                 )

            vs = v[i, split_tokens]

            if self._weighted_sum:
                q_split = q_all[i, split_tokens] # (num_sentences, r_embed_dim)
                k_split = k_all[i, split_tokens] # (num_sentences, r_embed_dim)

                if self.use_rope:
                    # Select precomputed frequencies for the current number of sentences
                    # self.rope_freqs_cis should already be on the correct device
                    freqs_cis_i = self.rope_freqs_cis[:num_sentences] # Select first num_sentences freqs
                    q_split = rotary_emb(q_split, freqs_cis_i)
                    k_split = rotary_emb(k_split, freqs_cis_i)

                sequence_query = q_split[-1]
                sentence_keys = k_split

                sentence_reward_diffs = vs[1:] - vs[:-1]
                sentence_reward = torch.cat([vs[:1], sentence_reward_diffs], dim=0)

                assert sentence_reward.shape[0] == num_sentences
                assert sentence_keys.shape[0] == num_sentences

                attn_weights = torch.matmul(sequence_query, sentence_keys.transpose(0, 1)) / (q_split.shape[-1] ** 0.5)

                if dtype == torch.float16:
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
                else:
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                seq_reward = torch.matmul(attn_weights.unsqueeze(0), sentence_reward).squeeze(0)

            else:
                sentence_reward = vs
                seq_reward = torch.sum(vs).unsqueeze(0)

            sentence_rewards.append(sentence_reward.reshape((-1,)))
            sequence_rewards.append(seq_reward)
            sentence_nums.append(num_sentences)

        return dict(
            sentence_rewards = sentence_rewards,
            sequence_rewards = torch.stack(sequence_rewards).squeeze(1),
            sentence_nums = sentence_nums
        )

    def _prepare(self, batch : Dict, dtype : torch.dtype, attention_heads : int = 1):
        assert attention_heads == 1, f'Only support one head attention'
        # Make sure 'split_mask' from the batch is renamed or assigned to 'splitted_mask' if needed later
        # For example, if your batch dictionary uses 'split_mask', you might need:
        # splitted_mask = batch.get('split_mask') or batch.get('splitted_mask')
        input_ids, attention_mask, splitted_mask = itemgetter(
            'input_ids', 'attention_mask', 'splitted_mask' # Ensure 'splitted_mask' exists in the batch dict passed here
        )(batch)
        assert len(input_ids.shape) == 2
        # The original _prepare returned split_mask (now splitted_mask) and None for masks.
        # Keeping that structure unless sentence_masks are needed elsewhere.
        return splitted_mask, None # Return the splitted_mask needed by forward/forward_value

    def forward(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        splitted_mask : torch.Tensor, # Ensure this is passed correctly
        past_key_values : torch.Tensor = None,
        position_ids : torch.Tensor = None,
        head_mask : torch.Tensor = None,
        inputs_embeds : torch.Tensor = None,
        use_cache : bool = False,
    ) -> Dict:
        # Train RM
        hidden_states = self.base_model(
            input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache
        )[0]

        # Prepare the masks - ensure 'splitted_mask' is passed in the dict
        split_token_masks, _ = self._prepare( # Assuming _prepare returns (split_mask, None)
            batch = dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                splitted_mask = splitted_mask # Pass the input splitted_mask here
            ),
            dtype = hidden_states.dtype
        )

        # Rest of the forward method using split_token_masks...
        bs = input_ids.shape[0] // 2
        results = self._get_rewards(hidden_states, split_token_masks)
        # ... (loss calculation etc.) ...
        rewards, sentence_nums = itemgetter('sequence_rewards', 'sentence_nums')(results)
        chosen_rewards, reject_rewards = rewards[ : bs], rewards[bs : ]
        chosen_sentence_nums, reject_sentence_nums = sentence_nums[ : bs], sentence_nums[bs : ]

        loss = 0.0
        total_penalty = 0.0
        penalty_ratio = 0.0
        max_sentence_nums = 0
        chosen_scores, reject_scores = [], []

        for i in range(bs):
            chosen_score, reject_score = chosen_rewards[i], reject_rewards[i]
            chosen_sentence_num, reject_sentence_num = chosen_sentence_nums[i], reject_sentence_nums[i]
            if not self._weighted_sum:
                # Ensure chosen_sentence_num and reject_sentence_num are valid numbers > 0 if used in denominator
                denom = max(chosen_sentence_num, reject_sentence_num)
                alpha = 1.0 / (denom + 1e-6) if denom > 0 else 1.0
            else:
                alpha = 1.0

            bt_loss = -torch.nn.functional.logsigmoid(alpha * (chosen_score - reject_score)).mean()

            if self.penalty_coef > 0.0:
                penalty_term = (chosen_score ** 2 + reject_score ** 2) * self.penalty_coef
            else:
                penalty_term = torch.zeros_like(bt_loss).detach()

            total_penalty += penalty_term.item()
            current_loss_item = bt_loss.item()
            penalty_ratio += (penalty_term.item() / (current_loss_item + penalty_term.item() + 1e-6)) if (current_loss_item + penalty_term.item()) != 0 else 0.0
            loss += (bt_loss + penalty_term)

            chosen_scores.append(chosen_score)
            reject_scores.append(reject_score)
            max_sentence_nums = max(max_sentence_nums, chosen_sentence_num, reject_sentence_num)

        loss = loss / bs
        total_penalty = total_penalty / bs
        penalty_ratio = penalty_ratio / bs

        return dict(
            loss = loss,
            chosen_rewards = torch.stack(chosen_scores),
            reject_rewards = torch.stack(reject_scores),
            total_penalty = total_penalty,
            penalty_ratio = penalty_ratio,
            N = max_sentence_nums
        )

    @torch.no_grad()
    def forward_value(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        splitted_mask : torch.Tensor, # Ensure this is passed correctly
        past_key_values : torch.Tensor = None,
        position_ids : torch.Tensor = None,
        head_mask : torch.Tensor = None,
        inputs_embeds : torch.Tensor = None,
        use_cache : bool = False,
    ) -> Dict:
        # evaluate RM
        hidden_states = self.base_model(
            input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache
        )[0]

        # Prepare the masks - ensure 'splitted_mask' is passed in the dict
        split_token_masks, _ = self._prepare(
            batch = dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                splitted_mask = splitted_mask # Pass the input splitted_mask here
            ),
            dtype = hidden_states.dtype
        )

        # Get rewards using the prepared masks
        results = self._get_rewards(hidden_states, split_token_masks)
        rewards, sentence_rewards = itemgetter('sequence_rewards', 'sentence_rewards')(results)

        return dict(
            rewards = rewards,
            scores = rewards,
            setence_rewards = sentence_rewards
        )

    def init_mu_var(self, device):
        """
        Args:
            device (int or torch.device)
        """
        if isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        # Define max_N if not already defined (e.g., based on max_rope_sentences or a fixed value)
        self.max_N = getattr(self, 'max_N', 100) # Default to 100 if not set
        if not hasattr(self, 'mu'):
            self.register_buffer('mu', torch.zeros(self.max_N, device=device), persistent=False)
        if not hasattr(self, 'var'):
            self.register_buffer('var', torch.ones(self.max_N, device=device), persistent=False)
        if not hasattr(self, 'n_updates'):
            self.register_buffer('n_updates', torch.zeros(1, dtype=torch.long, device=device), persistent=False)

    def _update_relative_rms(self, sentence_rewards: List[torch.Tensor]):
        # Ensure mu and var are initialized
        if not hasattr(self, 'mu') or not hasattr(self, 'var'):
             print("Warning: mu/var not initialized. Skipping stats update.")
             # Or initialize them here: self.init_mu_var(next(self.parameters()).device)
             return

        for sentence_reward in sentence_rewards:
            if not isinstance(sentence_reward, torch.Tensor) or sentence_reward.numel() == 0:
                continue # Skip empty or invalid tensors

            assert len(sentence_reward.shape) == 1, f"Expected 1D tensor, got {sentence_reward.shape}"
            T = sentence_reward.shape[0]
            if T == 0: continue # Skip if no sentences

            src_pos = np.arange(T)
            relative_pos = np.arange(1, T + 1) / T # Normalized position [1/T, ..., T/T]
            # Map normalized position to the bins of mu/var
            target_pos_indices = (self.mu.shape[0] * relative_pos - 1e-6).astype(int) # Map to [0, max_N-1] indices
            target_pos_indices = np.clip(target_pos_indices, 0, self.mu.shape[0] - 1)

            # Aggregate rewards per target position bin using scatter_add_ for efficiency
            device = sentence_reward.device
            target_pos_tensor = torch.tensor(target_pos_indices, device=device, dtype=torch.long)

            summed_rewards = torch.zeros_like(self.mu)
            summed_rewards.scatter_add_(0, target_pos_tensor, sentence_reward)

            target_pos_counts = torch.zeros_like(self.mu, dtype=torch.int64)
            target_pos_counts.scatter_add_(0, target_pos_tensor, torch.ones_like(sentence_reward, dtype=torch.int64))

            # Calculate mean reward per bin, handling division by zero
            mean_rewards_in_bin = torch.where(
                target_pos_counts > 0,
                summed_rewards / target_pos_counts,
                torch.zeros_like(self.mu) # Or maybe self.mu?
            )

            # === Welford's online algorithm for mean and variance ===
            self.n_updates += 1
            delta = mean_rewards_in_bin - self.mu
            self.mu.add_(delta / self.n_updates) # Update mean: M_n = M_{n-1} + (x_n - M_{n-1}) / n

            delta2 = mean_rewards_in_bin - self.mu # Use the *updated* mean
            # Update variance: S_n = S_{n-1} + (x_n - M_{n-1}) * (x_n - M_n)
            # Note: self.var stores M2/(n-1) or S/n depending on implementation.
            # Assuming self.var stores the variance S/n
            # This update needs careful implementation based on definition of self.var
            # Simplified update (might need adjustment based on exact variance definition):
            if self.n_updates > 1:
                 self.var.copy_(((self.n_updates - 2) * self.var + delta * delta2) / (self.n_updates - 1))
            # Handle first update separately if variance is unbiased (n-1 denominator)
            # Welford's M2 update: M2_n = M2_{n-1} + (x_n - M_{n-1}) * (x_n - M_n)
            # Variance = M2_n / (n-1) or M2_n / n

    @torch.no_grad()
    def update_mean_std(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        splitted_mask : torch.Tensor, # Ensure this is passed correctly
        past_key_values : torch.Tensor = None,
        position_ids : torch.Tensor = None,
        head_mask : torch.Tensor = None,
        inputs_embeds : torch.Tensor = None,
        use_cache : bool = False,
    ):
        # get reward
        hidden_states = self.base_model(
            input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache
        )[0]
        split_token_masks, _ = self._prepare(
            batch = dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                splitted_mask = splitted_mask # Pass the input splitted_mask here
            ),
            dtype = hidden_states.dtype
        )
        results = self._get_rewards(hidden_states, split_token_masks)
        # Use 'sentence_rewards' list from the results
        sentence_rewards_list = results['sentence_rewards']

        # update mean and std using position calculation
        self._update_relative_rms(sentence_rewards_list) # Pass the list of tensors

    def export_stats(self, path: str, include_model_info: bool = False):
        """
        增强统计量导出功能

        参数：
            include_model_info : 是否包含模型基础信息用于校验
        """
        if not hasattr(self, 'mu') or not hasattr(self, 'var'):
            print("Warning: mu/var not initialized. Cannot export stats.")
            return

        stats = {
            'mu': self.mu.cpu().clone(),
            'var': self.var.cpu().clone(),
            'config': {
                'max_N': self.max_N, # Use the actual max_N used
                'model_type': self.__class__.__name__,
                'stats_version': 1.1
            },
            'n_updates': self.n_updates.cpu().item() if hasattr(self, 'n_updates') else 0
        }

        if include_model_info:
            # Ensure self.config exists and has to_dict method
            if hasattr(self, 'config') and callable(getattr(self.config, 'to_dict', None)):
                 stats['model_config'] = self.config.to_dict()
            else:
                 stats['model_config'] = str(getattr(self, 'config', 'N/A'))

        torch.save(stats, path)
        print(f"Stats exported to {path}")

    def load_stats(self, path: str, strict: bool = True):
        """
        加载预计算的统计量数据

        参数：
            path : 统计量文件路径
            strict : 是否严格检查维度匹配（建议True）

        数学验证：
            检查 max_N 维度一致性：
                E[dim] = E_saved[dim]
                Var[dim] = Var_saved[dim]
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Statistics file {path} not found")

        stats = torch.load(path, map_location='cpu')

        if 'config' not in stats:
            raise ValueError("Invalid statistics format: missing config")

        saved_max_N = stats['config'].get('max_N', None)
        if saved_max_N is None:
            raise KeyError("Missing 'max_N' in config")

        # Initialize mu/var if they don't exist, using loaded max_N if strict=False
        current_max_N = getattr(self, 'max_N', saved_max_N) # Use saved_max_N if not set
        if not hasattr(self, 'mu') or not hasattr(self, 'var'):
            print(f"Initializing mu/var buffers with max_N={current_max_N}")
            self.max_N = current_max_N # Set max_N attribute
            self.init_mu_var(next(self.parameters()).device) # Initialize on current device

        if strict and (saved_max_N != self.max_N):
            raise RuntimeError(
                f"Dimension mismatch: Current max_N={self.max_N}, "
                f"Loaded max_N={saved_max_N}. Set strict=False to allow loading."
            )
        elif not strict and saved_max_N != self.max_N:
             print(f"Warning: Dimension mismatch (strict=False). Current max_N={self.max_N}, Loaded max_N={saved_max_N}. Resizing/truncating stats.")
             # Handle resize/truncation if needed, or re-init buffer
             self.max_N = saved_max_N
             self.init_mu_var(next(self.parameters()).device) # Re-initialize with loaded size

        # Ensure buffers exist before copying data
        if not hasattr(self, 'mu') or not hasattr(self, 'var') or not hasattr(self, 'n_updates'):
             raise RuntimeError("Buffers mu, var, or n_updates not initialized correctly.")

        device = next(self.parameters()).device

        self.mu.data.copy_(stats['mu'].to(device=device))
        self.var.data.copy_(stats['var'].to(device=device))
        if 'n_updates' in stats:
             self.n_updates.data.copy_(torch.tensor(stats['n_updates'], dtype=torch.long, device=device))

        self.validate_stats()
        print(f"Successfully loaded stats from {path} (n_updates={self.n_updates.item()})")

    def validate_stats(self):
        """
        执行统计量完整性校验
        """
        if not hasattr(self, 'mu') or not hasattr(self, 'var'):
            print("Warning: mu/var not initialized. Skipping validation.")
            return

        # 维度检查
        assert self.mu.shape == (self.max_N,), \
            f"mu shape {self.mu.shape} != expected ({self.max_N},)"
        assert self.var.shape == (self.max_N,), \
            f"var shape {self.var.shape} != expected ({self.max_N},)"

        # 数值有效性检查
        assert not torch.isnan(self.mu).any(), "NaN values in mu"
        assert not torch.isinf(self.var).any(), "Inf values in var"
        # Variance can be slightly negative due to floating point errors in online algorithms
        # assert (self.var >= -1e-6).all(), f"Significant negative variance values found: {self.var[self.var < 0]}"
        if not (self.var >= -1e-6).all():
             print(f"Warning: Negative variance values detected (min: {self.var.min()}). Clamping to 0.")
             self.var.data.clamp_(min=0.0)

        # 设备一致性检查
        model_device = next(self.parameters()).device
        assert self.mu.device == model_device, \
            f"mu device {self.mu.device} != model device {model_device}"
        assert self.var.device == model_device, \
            f"var device {self.var.device} != model device {model_device}"
        if hasattr(self, 'n_updates'):
             assert self.n_updates.device == model_device, \
                 f"n_updates device {self.n_updates.device} != model device {model_device}"

    def plot_mean_var(self, save_dir: str = "./plots", figsize=(12, 6)):
        """
        可视化位置相关的奖励统计量，生成并保存图像文件

        参数：
            save_dir : 图片保存路径（自动创建目录）
            figsize : (宽度, 高度) 单位英寸

        数学原理：
            对每个相对位置 i ∈ [0, 1]:
                μ(i) ≈ E[r|relative_position=i] 的条件期望
                σ²(i) ≈ Var(r|relative_position=i) 的条件方差
            可视化曲线展示统计量随相对位置的变化趋势
        """
        if not hasattr(self, 'mu') or not hasattr(self, 'var') or not hasattr(self, 'n_updates'):
             print("Warning: Stats (mu/var/n_updates) not initialized. Cannot plot.")
             return {}

        # 设备转移和类型转换
        mu = self.mu.detach().cpu().numpy()
        var = self.var.detach().cpu().numpy().clip(min=0) # Ensure variance is non-negative for plotting std dev
        n_updates_val = self.n_updates.item() if self.n_updates is not None else 'N/A'

        os.makedirs(save_dir, exist_ok=True)

        # 相对位置坐标 (bin centers)
        positions = np.linspace(0.5 / self.max_N, 1 - 0.5 / self.max_N, self.max_N)

        plt.figure(figsize=figsize)

        # 均值子图
        plt.subplot(1, 2, 1)
        plt.plot(positions, mu, 'b-', linewidth=2, label='Mean Reward')
        plt.xlabel("Relative Position (0=start, 1=end)", fontsize=10)
        plt.ylabel("Mean Reward (μ)", fontsize=10)
        plt.title(f"Position-wise Mean (n={n_updates_val})", fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.legend()

        # 标准差子图 (sqrt of variance)
        plt.subplot(1, 2, 2)
        std_dev = np.sqrt(var)
        plt.plot(positions, std_dev, 'r--', linewidth=2, label='Std Dev Reward')
        # Optionally plot mean +/- std dev bounds
        # plt.fill_between(positions, mu - std_dev, mu + std_dev, color='red', alpha=0.1, label='μ ± σ')
        plt.xlabel("Relative Position (0=start, 1=end)", fontsize=10)
        plt.ylabel("Std Dev Reward (σ)", fontsize=10)
        plt.title(f"Position-wise Std Dev (n={n_updates_val})", fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.suptitle(f"Reward Statistics vs Relative Position (max_N={self.max_N})", fontsize=14)

        save_path = os.path.join(save_dir, f"reward_stats_n{n_updates_val}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")

        return {"mu_plot": mu, "var_plot": var} # Return raw variance