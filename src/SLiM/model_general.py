"""
Generalized SLiMedNet - Model-agnostic version of the SLiM architecture.

Reference: "State-wise Linear Modulation (SLiM): A Novel Approach for Steering
Large Language Models"

Adapted for Hallucination Reduction: applies a single SLiM module on a
user-specified target layer (no gating, no Top-K) with optional Low-Rank
factorization for scale and shift (default rank=32).
"""

import torch
import torch.nn as nn
from typing import Optional
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
            nn.Linear(state_dim, target_dim ),
            activation(),
            nn.LayerNorm(target_dim),
        )

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        return self.projection(state_vector)


class LowRankLinear(nn.Module):
    """
    Low-rank factorized linear layer: y = up(down(x)).

    Approximates nn.Linear(in_features, out_features) as W ≈ A @ B where
    A has shape (out_features, rank) and B has shape (rank, in_features),
    reducing the parameter count from in*out to rank*(in+out).

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the factorization (default: 32)
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 32):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class GeneralSLiMedNet(nn.Module):
    """
    Model-agnostic State-wise Linear Modulation (SLiM) wrapper.

    Wraps any HuggingFace CausalLM and applies FiLM-like modulation
    (scale/shift) to a single target transformer layer's output,
    conditioned on an external state vector.

    Scale and shift use a low-rank factorization (LowRankLinear) to reduce
    the parameter count from hidden_size² to rank*(2*hidden_size).

    Args:
        model: A HuggingFace CausalLM (frozen)
        state_embed_dim: Dimension of the input state vector (1 for scalar)
        target_layer: Index of the single transformer layer to apply SLiM at
        slim_rank: Rank for the low-rank factorization of scale and shift (default: 32)
        record_SLiM: Whether to record scale/shift values for analysis
        record_hidden_states: Whether to record pre/post modulation hidden states
        dtype: Data type for SLiM modules (default: bfloat16)
    """

    def __init__(
        self,
        model: nn.Module,
        state_embed_dim: int,
        target_layer: int,
        slim_rank: int = 32,
        record_SLiM: bool = False,
        record_hidden_states: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super(GeneralSLiMedNet, self).__init__()

        # Store the base model (frozen)
        self.base_model = model
        self.slim_dtype = dtype

        # Detect architecture
        self.arch_name, self.arch_config = LLMArchitectureDetector.detect_architecture(model)
        self.hidden_size = LLMArchitectureDetector.get_hidden_size(model)
        self.num_layers = LLMArchitectureDetector.get_num_layers(model, self.arch_config)

        # Validate target layer
        if target_layer < 0 or target_layer >= self.num_layers:
            raise ValueError(
                f"target_layer={target_layer} fuori range [0, {self.num_layers - 1}]"
            )

        self.target_layer = target_layer
        # Keep as list for checkpoint compatibility
        self.apply_SLiM_at_layers = [target_layer]
        self.n_slim_layers = 1

        print(f"[SLiM] Detected architecture: {self.arch_name}")
        print(f"[SLiM] Hidden size: {self.hidden_size}, Num layers: {self.num_layers}")
        print(f"[SLiM] Target layer: {target_layer}, rank: {slim_rank}, dtype: {dtype}")

        self.slim_rank = slim_rank

        # State projector: state_dim → hidden_size
        self.state_proj = StateBlock(state_embed_dim, self.hidden_size).to(dtype)

        # SLiM modulation parameters — low-rank factorized (H → rank → H)
        # scale: LowRankLinear(H, H, rank) — element-wise multiplicative modulation
        # shift: LowRankLinear(H, H, rank) — element-wise additive modulation
        self.SLiM_scale = LowRankLinear(self.hidden_size, self.hidden_size, rank=slim_rank).to(dtype)
        self.SLiM_shift = LowRankLinear(self.hidden_size, self.hidden_size, rank=slim_rank).to(dtype)
        self._init_modulation_layers()

        # Current state (set during forward/generate)
        self.current_state_embed = None
        self._last_scale = None
        self._last_shift = None

        # Steering strength: 0.0 = no steering, 1.0 = full steering
        self.alpha = 1.0

        # Register hook on the target layer
        self.hooks = self._register_hooks()

        # Recording flags
        self.record_SLiM = record_SLiM
        self.record_hidden_states = record_hidden_states

        if self.record_SLiM:
            self.SLiM_values = {
                "scale": [],
                "shift": [],
            }

        if self.record_hidden_states:
            self.hidden_states = {
                "unmodulated": [],
                "modulated": [],
            }

        # Contrastive training: capture mode
        self._capture_mode = False
        self.captured_representations = {}

    def _init_modulation_layers(self):
        """
        Identity-preserving initialization for stable training.

        `up` is zero-initialized so scale/shift start from 0, therefore
        the initial modulation is exactly the identity mapping.
        """
        for module in (self.SLiM_scale, self.SLiM_shift):
            nn.init.normal_(module.down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.up.weight)
            if module.up.bias is not None:
                nn.init.zeros_(module.up.bias)

    def _register_hooks(self) -> list:
        """Register a forward hook on the target transformer layer."""
        hooks = []
        layers = LLMArchitectureDetector.get_layers(self.base_model, self.arch_config)
        layer = layers[self.target_layer]
        hooks.append(layer.register_forward_hook(self._create_hook()))
        return hooks

    def _create_hook(self):
        """
        Create a SLiM modulation hook for the target layer.

        Modulation formula (following the original SLiM paper):
            projected = StateBlock(state)
            scale = tanh(W_scale · projected)
            shift = tanh(W_shift · projected)
            delta = hidden * scale + shift
            output' = hidden + alpha * delta

        Where alpha controls the steering strength (0.0 = no modulation,
        1.0 = full modulation). With this formulation, scale=0 and shift=0
        produce identity (output' = hidden), preventing initialization collapse.

        In capture mode (contrastive training):
        - Saves the modulated hidden state WITH gradients for loss computation.
          torch.enable_grad() is required because bitsandbytes 4-bit quantization
          may run the base model's forward inside a torch.no_grad() context,
          which would strip grad_fn from steered_output even though SLiM params
          have requires_grad=True.
        - Detaches the output sent to subsequent layers to save memory
          (subsequent layers are frozen and their output is not used)

        In inference mode:
        - The steered output flows naturally through the model without detach.
        - No LayerNorm is applied post-modulation, preserving the hidden state
          distribution expected by subsequent frozen layers.
        """
        def hook(module, input, output):
            if self.current_state_embed is None:
                self._last_scale = None
                self._last_shift = None
                self.captured_representations = {}
                return output

            hidden = output[0] if isinstance(output, tuple) else output

            # ── SLiM FiLM computation ────────────────────────────────────────
            # torch.enable_grad() ensures gradient tracking is active even if
            # the 4-bit base model runs in a no_grad context internally.
            with torch.enable_grad():
                state_input = self.current_state_embed.to(self.slim_dtype)
                projected_state = self.state_proj(state_input)
                scale = torch.tanh(self.SLiM_scale(projected_state)).to(hidden.dtype)
                shift = torch.tanh(self.SLiM_shift(projected_state)).to(hidden.dtype)
                self._last_scale = scale
                self._last_shift = shift

                # Always detach hidden: we never need gradients through
                # the base model, only through SLiM params (scale/shift).
                # This saves VRAM by cutting the graph before the target layer.
                h = hidden.detach()

                # FiLM modulation with residual connection and alpha scaling:
                # steered = hidden + alpha * (hidden * scale + shift)
                #         = hidden * (1 + alpha*scale) + alpha*shift
                delta = h * scale + shift
                steered_output = h + self.alpha * delta

            # Save modulation for downstream losses (e.g. InfoNCE in hybrid training).
            # This is kept even when _capture_mode is False.
            self.captured_representations["modulated"] = steered_output

            # --- Capture mode for contrastive-only memory optimization ---
            if self._capture_mode:
                # Detach what propagates to frozen subsequent layers
                steered_for_model = steered_output.detach()
            else:
                # Inference: let the steered tensor flow naturally
                steered_for_model = steered_output

            # Record values for analysis
            if self.record_SLiM:
                self.SLiM_values["scale"].append(
                    scale.detach().cpu().numpy()
                )
                self.SLiM_values["shift"].append(
                    shift.detach().cpu().numpy()
                )

            # Record hidden states
            if self.record_hidden_states:
                self.hidden_states["modulated"].append(
                    steered_for_model.detach().cpu().numpy()
                )
                self.hidden_states["unmodulated"].append(
                    hidden.detach().cpu().numpy()
                )

            # Reconstruct output tuple if needed
            if isinstance(output, tuple):
                output = (steered_for_model,) + output[1:]
            else:
                output = steered_for_model

            return output

        return hook

    def forward(
        self,
        input_ids: torch.Tensor,
        state_tensor: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional state-conditioned modulation.

        Args:
            input_ids: Token IDs [batch, seq_len]
            state_tensor: State vector [batch, state_dim] (None = no modulation)
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Set state for hooks
        self.captured_representations = {}
        if state_tensor is None:
            self.current_state_embed = None
            self._last_scale = None
            self._last_shift = None
        else:
            self.current_state_embed = state_tensor.unsqueeze(1)

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(
        self,
        input_ids: torch.Tensor,
        state_tensor: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Generate text with optional state-conditioned modulation.

        Args:
            input_ids: Token IDs [batch, seq_len]
            state_tensor: State vector [batch, state_dim] (None = no modulation)
            attention_mask: Attention mask [batch, seq_len]
            **generate_kwargs: Additional args for model.generate()

        Returns:
            Generated token IDs
        """
        if state_tensor is None:
            self.captured_representations = {}
            self.current_state_embed = None
            self._last_scale = None
            self._last_shift = None
        else:
            self.captured_representations = {}
            self.current_state_embed = state_tensor.unsqueeze(1)

        return self.base_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    # =========================================================================
    # Contrastive Training Support
    # =========================================================================

    def set_capture_mode(self, enabled: bool):
        """
        Enable/disable representation capture for contrastive training.

        When enabled, the hook saves the modulated hidden state WITH gradients
        and detaches what flows to subsequent layers (memory optimization).
        """
        self._capture_mode = enabled
        if not enabled:
            self.captured_representations = {}

    def get_captured_representation(self) -> Optional[torch.Tensor]:
        """
        Get the captured modulated representation from the last forward pass.

        Returns:
            Tensor [batch, seq_len, hidden_size] with gradient connection
            to SLiM parameters, or None if no modulation was applied.
        """
        return self.captured_representations.get("modulated", None)

    def get_modulation_penalty(self) -> torch.Tensor:
        """
        L2 penalty on current scale/shift activations.

        Useful as a regularizer to keep steering close to identity and avoid
        degenerate generations.
        """
        if self._last_scale is None or self._last_shift is None:
            device = next(self.state_proj.parameters()).device
            return torch.zeros((), device=device, dtype=torch.float32)
        return self._last_scale.float().pow(2).mean() + self._last_shift.float().pow(2).mean()

    @staticmethod
    def extract_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract the hidden state of the last non-padding token for each sample.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] with 1 for real tokens, 0 for padding

        Returns:
            [batch, hidden_size] — last-token representation (preserves gradients)
        """
        # Index of last real token per sample
        lengths = attention_mask.sum(dim=1).long() - 1  # [batch]
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, lengths]  # [batch, H]

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
                "scale": [],
                "shift": [],
            }

    def reset_record_hidden_states(self):
        """Clear recorded hidden states."""
        if self.record_hidden_states:
            self.hidden_states = {
                "unmodulated": [],
                "modulated": [],
            }
