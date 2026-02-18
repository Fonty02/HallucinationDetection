"""
Generalized SLiMedNet - Model-agnostic version of the SLiM architecture.

Reference: "State-wise Linear Modulation (SLiM): A Novel Approach for Steering
Large Language Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.truthx.truthx_model import LLMArchitectureDetector


class StateBlock(nn.Module):
    """
    Projects an arbitrary-dimensional state vector into the model's hidden space.

    Architecture: state_dim → target_dim//4 → target_dim//2 → target_dim + LayerNorm
    """
    def __init__(self, state_dim: int, target_dim: int, activation=nn.ReLU):
        super(StateBlock, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(state_dim, target_dim // 4),
            activation(),
            nn.Linear(target_dim // 4, target_dim // 2),
            activation(),
            nn.Linear(target_dim // 2, target_dim),
            nn.LayerNorm(target_dim),
        )

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        return self.projection(state_vector)


class LowRankLinear(nn.Module):
    """
    Low-rank factorization: W ≈ B·A with A∈ℝ^{H×r}, B∈ℝ^{r×H}.

    Replaces nn.Linear(H, H) (H² params) with two small matrices:
        2·H·r params instead of H².

    Example with H=3584, r=64:
        Full:     3584² = 12.9M params per layer
        LowRank:  2·3584·64 = 459K params per layer  (28× reduction)

    """
    def __init__(self, in_features: int, out_features: int, rank: int = 64):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=True)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


class GeneralSLiMedNet(nn.Module):
    """
    Model-agnostic State-wise Linear Modulation (SLiM) wrapper.

    Wraps any HuggingFace CausalLM and applies FiLM-like modulation
    (scale/shift/gate) to each transformer layer's output, conditioned
    on an external state vector.

    The gate mechanism λ_i = σ(g_i(s)) learns per-layer modulation strength,
    enabling Top-K layer selection at inference time based on learned gate values.

    Args:
        model: A HuggingFace CausalLM (frozen)
        state_embed_dim: Dimension of the input state vector (1 for scalar)
        apply_SLiM_at_layers: List of layer indices to apply SLiM at (default: all)
        record_SLiM: Whether to record γ/scale/shift values for analysis
        record_hidden_states: Whether to record pre/post modulation hidden states
    """

    def __init__(
        self,
        model: nn.Module,
        state_embed_dim: int,
        apply_SLiM_at_layers: Optional[List[int]] = None,
        record_SLiM: bool = False,
        record_hidden_states: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        slim_rank: int = 64,
    ):
        super(GeneralSLiMedNet, self).__init__()

        # Store the base model (frozen)
        self.base_model = model
        self.slim_dtype = dtype
        self.slim_rank = slim_rank

        # Detect architecture
        self.arch_name, self.arch_config = LLMArchitectureDetector.detect_architecture(model)
        self.hidden_size = LLMArchitectureDetector.get_hidden_size(model)
        self.num_layers = LLMArchitectureDetector.get_num_layers(model, self.arch_config)

        print(f"[SLiM] Detected architecture: {self.arch_name}")
        print(f"[SLiM] Hidden size: {self.hidden_size}, Num layers: {self.num_layers}")
        print(f"[SLiM] SLiM modules dtype: {dtype}, rank: {slim_rank}")

        # Determine which layers to apply SLiM
        self.apply_SLiM_at_layers = (
            apply_SLiM_at_layers
            if apply_SLiM_at_layers
            else list(range(self.num_layers))
        )
        self.n_slim_layers = len(self.apply_SLiM_at_layers)

        # State projector: state_dim → hidden_size  (in slim_dtype to match activations)
        self.state_proj = StateBlock(state_embed_dim, self.hidden_size).to(dtype)

        # Per-layer modulation parameters (in slim_dtype)
        # gate: tiny — 1 scalare per layer
        self.gate = nn.ModuleList([
            nn.Linear(state_embed_dim, 1).to(dtype)
            for _ in range(self.n_slim_layers)
        ])
        # scale/shift: low-rank  H→r→H  invece di H→H
        # params: 2·H·r per layer  vs  H² per layer
        self.SLiM_scale = nn.ModuleList([
            LowRankLinear(self.hidden_size, self.hidden_size, rank=slim_rank).to(dtype)
            for _ in range(self.n_slim_layers)
        ])
        self.SLiM_shift = nn.ModuleList([
            LowRankLinear(self.hidden_size, self.hidden_size, rank=slim_rank).to(dtype)
            for _ in range(self.n_slim_layers)
        ])

        # Current state (set during forward/generate)
        self.current_state_embed = None

        # Active layers mask (for Top-K inference)
        self._active_layer_indices: Optional[set] = None  # None = all active

        # Register hooks
        self.hooks = self._register_hooks()

        # Recording flags
        self.record_SLiM = record_SLiM
        self.record_hidden_states = record_hidden_states

        if self.record_SLiM:
            self.SLiM_values = {
                "gamma": [[] for _ in range(self.n_slim_layers)],
                "scale": [[] for _ in range(self.n_slim_layers)],
                "shift": [[] for _ in range(self.n_slim_layers)],
            }

        if self.record_hidden_states:
            self.hidden_states = {
                "unmodulated": [[] for _ in range(self.n_slim_layers)],
                "modulated": [[] for _ in range(self.n_slim_layers)],
            }

    def _register_hooks(self) -> list:
        """Register forward hooks on the selected transformer layers."""
        hooks = []
        layers = LLMArchitectureDetector.get_layers(self.base_model, self.arch_config)

        for i, idx in enumerate(self.apply_SLiM_at_layers):
            layer = layers[idx]
            hooks.append(layer.register_forward_hook(self._create_hook(i)))

        return hooks

    def _create_hook(self, layer_idx: int):
        """
        Create a SLiM modulation hook for a specific layer.

        The hook applies:
            projected = StateBlock(state)
            γ = σ(W_gate · state)     -- gate (learned importance)
            scale = tanh(W_scale · projected)
            shift = tanh(W_shift · projected)
            output' = LayerNorm((output * scale + shift) * γ) + output  -- residual
        """
        def hook(module, input, output):
            if self.current_state_embed is None:
                return output

            # Check if this layer is active (for Top-K inference)
            if self._active_layer_indices is not None:
                if layer_idx not in self._active_layer_indices:
                    return output

            # Cast state to match hidden dtype
            state_input = self.current_state_embed.to(self.slim_dtype)

            # Project state to hidden space
            projected_state = self.state_proj(state_input)

            # Gate: learned per-layer importance
            gate_value = torch.sigmoid(self.gate[layer_idx](state_input))

            # Scale and shift (FiLM)
            scale = torch.tanh(self.SLiM_scale[layer_idx](projected_state))
            shift = torch.tanh(self.SLiM_shift[layer_idx](projected_state))

            # Record values for analysis
            if self.record_SLiM:
                self.SLiM_values["gamma"][layer_idx].append(
                    gate_value.detach().cpu().numpy()
                )
                self.SLiM_values["scale"][layer_idx].append(
                    scale.detach().cpu().numpy()
                )
                self.SLiM_values["shift"][layer_idx].append(
                    shift.detach().cpu().numpy()
                )

            # Apply modulation: (output * scale + shift) * gate
            hidden = output[0] if isinstance(output, tuple) else output
            steered_output = (hidden * scale + shift) * gate_value

            # Layer normalization for stability (Appendix D of paper)
            steered_output = F.layer_norm(steered_output, steered_output.shape[-1:])

            # Record hidden states
            if self.record_hidden_states:
                self.hidden_states["modulated"][layer_idx].append(
                    steered_output.detach().cpu().numpy()
                )
                self.hidden_states["unmodulated"][layer_idx].append(
                    hidden.detach().cpu().numpy()
                )

            # Residual connection (Appendix F of paper)
            steered_output = steered_output + hidden

            # Reconstruct output tuple if needed
            if isinstance(output, tuple):
                output = (steered_output,) + output[1:]
            else:
                output = steered_output

            return output

        return hook

    def forward(
        self,
        input_ids: torch.Tensor,
        state_tensor: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        active_layers: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional state-conditioned modulation.

        Args:
            input_ids: Token IDs [batch, seq_len]
            state_tensor: State vector [batch, state_dim] (None = no modulation)
            attention_mask: Attention mask [batch, seq_len]
            active_layers: List of SLiM layer indices to activate (None = all)

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Set state for hooks
        if state_tensor is None:
            self.current_state_embed = None
        else:
            self.current_state_embed = state_tensor.unsqueeze(1)

        # Set active layers for Top-K
        if active_layers is not None:
            self._active_layer_indices = set(active_layers)
        else:
            self._active_layer_indices = None

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(
        self,
        input_ids: torch.Tensor,
        state_tensor: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        active_layers: Optional[List[int]] = None,
        **generate_kwargs,
    ):
        """
        Generate text with optional state-conditioned modulation.

        Args:
            input_ids: Token IDs [batch, seq_len]
            state_tensor: State vector [batch, state_dim] (None = no modulation)
            attention_mask: Attention mask [batch, seq_len]
            active_layers: List of SLiM layer indices to activate (None = all)
            **generate_kwargs: Additional args for model.generate()

        Returns:
            Generated token IDs
        """
        if state_tensor is None:
            self.current_state_embed = None
        else:
            self.current_state_embed = state_tensor.unsqueeze(1)

        if active_layers is not None:
            self._active_layer_indices = set(active_layers)
        else:
            self._active_layer_indices = None

        return self.base_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
        )

    # =========================================================================
    # Gate-based Top-K Layer Selection
    # =========================================================================

    def get_gate_values(self, state_tensor: torch.Tensor) -> Dict[int, float]:
        """
        Compute gate values λ_i = σ(g_i(s)) for all SLiM layers.

        This reveals how much each layer contributes to the modulation
        for a given state. Higher gate = more important layer.

        Args:
            state_tensor: State vector [state_dim] (e.g., [1.0] for truthful)

        Returns:
            Dict mapping layer_idx → gate_value (float in [0,1])
        """
        gate_values = {}
        state = state_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]

        with torch.no_grad():
            for i, layer_idx in enumerate(self.apply_SLiM_at_layers):
                gate_input = state.to(device=self.gate[i].weight.device, dtype=self.slim_dtype)
                gate_val = torch.sigmoid(self.gate[i](gate_input)).item()
                gate_values[layer_idx] = gate_val

        return gate_values

    def get_top_k_layers(self, state_tensor: torch.Tensor, k: int) -> List[int]:
        """
        Select Top-K layers with highest gate values for a given state.

        After training on all layers, the gate mechanism learns which layers
        are most important for the steering task. This method enables
        efficient inference by activating only the most impactful layers.

        Args:
            state_tensor: State vector [state_dim]
            k: Number of top layers to select (-1 for all)

        Returns:
            List of SLiM layer indices (into self.apply_SLiM_at_layers),
            sorted by gate value descending
        """
        if k == -1 or k >= self.n_slim_layers:
            return list(range(self.n_slim_layers))

        gate_values = self.get_gate_values(state_tensor)

        # Sort by gate value descending
        sorted_layers = sorted(gate_values.items(), key=lambda x: x[1], reverse=True)

        # Return top-K layer indices (mapped to SLiM internal indices)
        top_k_physical = [layer_idx for layer_idx, _ in sorted_layers[:k]]
        # Convert physical layer indices to SLiM hook indices
        layer_to_slim_idx = {
            phys_idx: slim_idx
            for slim_idx, phys_idx in enumerate(self.apply_SLiM_at_layers)
        }
        return [layer_to_slim_idx[phys_idx] for phys_idx in top_k_physical]

    def get_layer_ranking(self, state_tensor: torch.Tensor) -> List[Tuple[int, float]]:
        """
        Get all layers ranked by gate value for a given state.

        Returns:
            List of (physical_layer_idx, gate_value) sorted descending
        """
        gate_values = self.get_gate_values(state_tensor)
        return sorted(gate_values.items(), key=lambda x: x[1], reverse=True)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_trainable_params_count(self) -> int:
        """Count the number of trainable parameters (SLiM only, not base model)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset_SLiM_values(self):
        """Clear recorded SLiM values."""
        if self.record_SLiM:
            self.SLiM_values = {
                "gamma": [[] for _ in range(self.n_slim_layers)],
                "scale": [[] for _ in range(self.n_slim_layers)],
                "shift": [[] for _ in range(self.n_slim_layers)],
            }

    def reset_record_hidden_states(self):
        """Clear recorded hidden states."""
        if self.record_hidden_states:
            self.hidden_states = {
                "unmodulated": [[] for _ in range(self.n_slim_layers)],
                "modulated": [[] for _ in range(self.n_slim_layers)],
            }
