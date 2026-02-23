"""
Generalized SLiMedNet - Model-agnostic version of the SLiM architecture.

Reference: "State-wise Linear Modulation (SLiM): A Novel Approach for Steering
Large Language Models"

Adapted for Hallucination Reduction: applies a single SLiM module on a
user-specified target layer (no gating, no Top-K, no LowRank).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Linear(state_dim, target_dim // 4),
            activation(),
            nn.Linear(target_dim // 4, target_dim // 2),
            activation(),
            nn.Linear(target_dim // 2, target_dim),
            nn.LayerNorm(target_dim),
        )

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        return self.projection(state_vector)


class GeneralSLiMedNet(nn.Module):
    """
    Model-agnostic State-wise Linear Modulation (SLiM) wrapper.

    Wraps any HuggingFace CausalLM and applies FiLM-like modulation
    (scale/shift) to a single target transformer layer's output,
    conditioned on an external state vector.

    Following the original SLiM paper, scale and shift are standard
    nn.Linear(hidden_size, hidden_size) — no low-rank factorization.

    Args:
        model: A HuggingFace CausalLM (frozen)
        state_embed_dim: Dimension of the input state vector (1 for scalar)
        target_layer: Index of the single transformer layer to apply SLiM at
        record_SLiM: Whether to record scale/shift values for analysis
        record_hidden_states: Whether to record pre/post modulation hidden states
        dtype: Data type for SLiM modules (default: bfloat16)
    """

    def __init__(
        self,
        model: nn.Module,
        state_embed_dim: int,
        target_layer: int,
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
        print(f"[SLiM] Target layer: {target_layer}, dtype: {dtype}")

        # State projector: state_dim → hidden_size
        self.state_proj = StateBlock(state_embed_dim, self.hidden_size).to(dtype)

        # SLiM modulation parameters — single modules (as in original paper)
        # scale: nn.Linear(H, H) — element-wise multiplicative modulation
        # shift: nn.Linear(H, H) — element-wise additive modulation
        self.SLiM_scale = nn.Linear(self.hidden_size, self.hidden_size).to(dtype)
        self.SLiM_shift = nn.Linear(self.hidden_size, self.hidden_size).to(dtype)

        # Current state (set during forward/generate)
        self.current_state_embed = None

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
            output' = LayerNorm((output * scale + shift) + output)

        In capture mode (contrastive training):
        - Saves the modulated hidden state WITH gradients for loss computation.
          torch.enable_grad() is required because bitsandbytes 4-bit quantization
          may run the base model's forward inside a torch.no_grad() context,
          which would strip grad_fn from steered_output even though SLiM params
          have requires_grad=True.
        - Detaches the output sent to subsequent layers to save memory
          (subsequent layers are frozen and their output is not used)
        """
        def hook(module, input, output):
            if self.current_state_embed is None:
                return output

            hidden = output[0] if isinstance(output, tuple) else output

            # ── SLiM FiLM computation ────────────────────────────────────────
            # torch.enable_grad() ensures gradient tracking is active even if
            # the 4-bit base model runs in a no_grad context internally.
            with torch.enable_grad():
                state_input = self.current_state_embed.to(self.slim_dtype)
                projected_state = self.state_proj(state_input)
                scale = torch.tanh(self.SLiM_scale(projected_state))
                shift = torch.tanh(self.SLiM_shift(projected_state))
                steered_output = (hidden.detach() * scale + shift)
                # Residual connection (Appendix F of paper)
                steered_output = steered_output + hidden.detach()
                # Layer normalization for stability (Appendix D of paper)
                steered_output = F.layer_norm(steered_output, steered_output.shape[-1:])

            # --- Capture mode for contrastive training ---
            if self._capture_mode:
                # Save WITH grad_fn (through scale/shift ← SLiM params)
                self.captured_representations["modulated"] = steered_output
                # Detach what propagates to frozen subsequent layers
                steered_for_model = steered_output.detach()
            else:
                steered_for_model = steered_output.detach()

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
        if state_tensor is None:
            self.current_state_embed = None
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
            self.current_state_embed = None
        else:
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
            to SLiM parameters, or None if not in capture mode.
        """
        return self.captured_representations.get("modulated", None)

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
