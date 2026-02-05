import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
from torch import Tensor


class MLPAE(nn.Module):
    """
    MLP-based Autoencoder per TruthX.
    Separa le rappresentazioni in:
    - Semantic latent space (contenuto/significato)
    - Truthful latent space (veridicità)
    
    Architettura (come da paper):
    - Semantic Encoder:  input_dim -> 2048 -> 1024 (latent)
    - Truthful Encoder:  input_dim -> 2048 -> 1024 (latent)
    - Decoder:           1024 (latent) -> 2048 -> input_dim
    """

    def __init__(
        self,
        in_channels: int,  # hidden_size del LLM (es. 4096 per Llama-3-8B)
        semantic_latent_dim: int = 1024,
        truthful_latent_dim: int = 1024,
        semantic_hidden_dims: Optional[List[int]] = None,
        truthful_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        # Architettura TruthX: 2-layer MLPs [input_dim→2048→1024] come da paper
        if semantic_hidden_dims is None:
            # Default: input_dim -> 2048 -> 1024 (latent)
            # Specifica [2048] per avere 1 hidden layer intermedio
            semantic_hidden_dims = [2048]
        if truthful_hidden_dims is None:
            # Default: input_dim -> 2048 -> 1024 (latent)
            # Specifica [2048] per avere 1 hidden layer intermedio
            truthful_hidden_dims = [2048]
        if decoder_hidden_dims is None:
            # Default: 1024 (latent) -> 2048 -> input_dim
            # Specifica [2048] per avere 1 hidden layer intermedio
            decoder_hidden_dims = [2048]

        # Semantic Encoder
        self.semantic_encoder = self._build_encoder(
            in_channels, semantic_hidden_dims, semantic_latent_dim
        )

        # Truthful Encoder
        self.truthful_encoder = self._build_encoder(
            in_channels, truthful_hidden_dims, truthful_latent_dim
        )

        # Cross-Attention Module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=semantic_latent_dim, num_heads=1
        )

        # Projection layer se le dimensioni sono diverse
        self.proj = None
        if semantic_latent_dim != truthful_latent_dim:
            self.proj = nn.Linear(truthful_latent_dim, semantic_latent_dim, bias=False)

        # Decoder (hidden layers) + final layer
        self.decoder = self._build_decoder(semantic_latent_dim, decoder_hidden_dims)
        # Final layer: dall'ultimo hidden dim a in_channels
        decoder_out_dim = (
            decoder_hidden_dims[-1] if decoder_hidden_dims else semantic_latent_dim
        )
        self.final_layer = nn.Linear(decoder_out_dim, in_channels)

        self.semantic_latent_dim = semantic_latent_dim
        self.truthful_latent_dim = truthful_latent_dim

        # Centri per rappresentazioni positive (truthful) e negative (hallucinated)
        self.register_buffer("pos_center", None)
        self.register_buffer("neg_center", None)

    def _build_encoder(self, in_dim, hidden_dims, latent_dim):
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(),
                ]
            )
            current_dim = h_dim
        layers.extend(
            [
                nn.Linear(current_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.LeakyReLU(),
            ]
        )
        return nn.Sequential(*layers)

    def _build_decoder(self, latent_dim, hidden_dims):
        """Build decoder hidden layers (senza final layer)"""
        if not hidden_dims:
            return None
        layers = []
        current_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(),
                )
            )
            current_dim = h_dim
        return nn.Sequential(*layers)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to input space"""
        result = z
        if self.decoder is not None:
            result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def encode_semantic(self, x: Tensor) -> Tensor:
        return self.semantic_encoder(x)

    def encode_truthful(self, x: Tensor) -> Tensor:
        z = self.truthful_encoder(x)
        return F.normalize(z, p=2, dim=-1)

    def attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.proj is not None and query.size(-1) != key.size(-1):
            key = self.proj(key)
            value = self.proj(value)

        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        output, _ = self.cross_attention(query, key, value)
        return output.squeeze(0)

    def forward(self, x: Tensor, truthful_latent_rep=None):
        semantic_rep = self.encode_semantic(x)

        if truthful_latent_rep is None:
            truthful_rep = self.encode_truthful(x)
        else:
            truthful_rep = truthful_latent_rep
            if truthful_rep.dim() == 3:
                truthful_rep = truthful_rep.reshape(-1, truthful_rep.size(-1))

        # Combina semantic + truthful via attention
        z = semantic_rep + self.attention(semantic_rep, truthful_rep, truthful_rep)
        output = self.decode(z)

        return [output, x, semantic_rep, truthful_rep]

    def get_semantic_latent_rep(self, x: Tensor) -> Tensor:
        """Get semantic latent representation"""
        return self.encode_semantic(x)

    def get_truthful_latent_rep(self, x: Tensor) -> Tensor:
        """Get truthful latent representation (L2 normalized)"""
        return self.encode_truthful(x)

    def loss_function(self, recons, input):
        """Reconstruction loss"""
        return F.mse_loss(recons, input)


class LLMArchitectureDetector:
    """
    Rileva automaticamente l'architettura del modello LLM e fornisce
    accesso ai layer in modo agnostico.
    Supporta: Llama, Mistral, GPT-2, GPT-Neo, GPT-J, OPT, Qwen, ecc.
    """

    SUPPORTED_ARCHITECTURES = {
        # Llama-based models (Llama-1, Llama-2, Llama-3, Mistral, Qwen, ecc.)
        "llama": {
            "layers_path": "model.layers",
            "attn_path": "self_attn",
            "mlp_path": "mlp",
        },
        # GPT-2
        "gpt2": {
            "layers_path": "transformer.h",
            "attn_path": "attn",
            "mlp_path": "mlp",
        },
        # GPT-Neo, GPT-J
        "gpt_neo": {
            "layers_path": "transformer.h",
            "attn_path": "attn",
            "mlp_path": "mlp",
        },
        # OPT
        "opt": {
            "layers_path": "model.decoder.layers",
            "attn_path": "self_attn",
            "mlp_path": "fc1",  # OPT usa fc1/fc2 per MLP
            "mlp_path2": "fc2",
        },
        # Qwen (usando architettura Llama-like)
        "qwen2": {
            "layers_path": "model.layers",
            "attn_path": "self_attn",
            "mlp_path": "mlp",
        },
        # Falcon
        "falcon": {
            "layers_path": "transformer.h",
            "attn_path": "self_attention",
            "mlp_path": "mlp",
        },
        # Mixtral (Mistral con MoE)
        "mixtral": {
            "layers_path": "model.layers",
            "attn_path": "self_attn",
            "mlp_path": "block_sparse_moe",  # MoE layer
        },
        # Default fallback (assumes Llama-like structure)
        "default": {
            "layers_path": "model.layers",
            "attn_path": "self_attn",
            "mlp_path": "mlp",
        },
    }

    @staticmethod
    def detect_architecture(model) -> Tuple[str, Any]:
        """
        Rileva l'architettura del modello basandosi sulla classe o sulla struttura.

        Returns:
            (arch_name, arch_config): Nome architettura e configurazione
        """
        model_class = model.__class__.__name__.lower()
        model_name = getattr(model, "name_or_path", "").lower()

        # Detect based on model class name
        if "llama" in model_class or "mistral" in model_class:
            arch_config = LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["llama"]
            return "llama", arch_config
        elif "gpt2" in model_class or "gpt2" in model_name:
            arch_config = LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["gpt2"]
            return "gpt2", arch_config
        elif "gptneo" in model_class or "gptj" in model_class:
            arch_config = LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["gpt_neo"]
            return "gpt_neo", arch_config
        elif "opt" in model_class:
            arch_config = LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["opt"]
            return "opt", arch_config
        elif "qwen" in model_class or "qwen" in model_name:
            arch_config = LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["qwen2"]
            return "qwen2", arch_config
        elif "falcon" in model_class or "falcon" in model_name:
            arch_config = LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["falcon"]
            return "falcon", arch_config
        elif "mixtral" in model_class or "mixtral" in model_name:
            arch_config = LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["mixtral"]
            return "mixtral", arch_config
        else:
            # Try to infer from structure
            return LLMArchitectureDetector._infer_from_structure(model)

    @staticmethod
    def _infer_from_structure(model) -> Tuple[str, Dict[str, str]]:
        """
        Prova a inferire l'architettura dalla struttura del modello.
        """
        # Check for Llama-like structure (model.layers)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return "llama", LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["llama"]
        # Check for GPT-2 structure (transformer.h)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return "gpt2", LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["gpt2"]
        # Check for OPT structure
        elif (
            hasattr(model, "model")
            and hasattr(model.model, "decoder")
            and hasattr(model.model.decoder, "layers")
        ):
            return "opt", LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["opt"]
        else:
            # Default to Llama structure
            print(
                f"Warning: Unknown architecture {model.__class__.__name__}. Using default Llama-like structure."
            )
            return "default", LLMArchitectureDetector.SUPPORTED_ARCHITECTURES["default"]

    @staticmethod
    def get_layers(model, arch_config: Dict[str, str]):
        """
        Ottiene la lista dei layer del modello.
        """
        layers_path = arch_config["layers_path"]
        parts = layers_path.split(".")

        current = model
        for part in parts:
            current = getattr(current, part)

        return current

    @staticmethod
    def get_num_layers(model, arch_config: Dict[str, str]) -> int:
        """
        Ottiene il numero di layer del modello.
        """
        layers = LLMArchitectureDetector.get_layers(model, arch_config)
        return len(layers)

    @staticmethod
    def get_layer_module(model, layer_idx: int, arch_config: Dict[str, str]):
        """
        Ottiene un layer specifico del modello.
        """
        layers = LLMArchitectureDetector.get_layers(model, arch_config)
        return layers[layer_idx]

    @staticmethod
    def get_attention_module(layer, arch_config: Dict[str, str]):
        """
        Ottiene il modulo attention da un layer.
        """
        attn_path = arch_config["attn_path"]
        return getattr(layer, attn_path)

    @staticmethod
    def get_mlp_module(layer, arch_config: Dict[str, str]):
        """
        Ottiene il modulo MLP da un layer.
        """
        mlp_path = arch_config["mlp_path"]
        return getattr(layer, mlp_path)

    @staticmethod
    def get_hidden_size(model) -> int:
        """
        Ottiene la dimensione hidden del modello in modo agnostico.
        """
        # Try different config attributes
        config = model.config if hasattr(model, "config") else model

        if hasattr(config, "hidden_size"):
            return config.hidden_size
        elif hasattr(config, "n_embd"):
            return config.n_embd
        elif hasattr(config, "d_model"):
            return config.d_model
        else:
            # Try to infer from first layer
            arch_name, arch_config = LLMArchitectureDetector.detect_architecture(model)
            layers = LLMArchitectureDetector.get_layers(model, arch_config)
            first_layer = layers[0]

            # Try to get from attention
            attn = LLMArchitectureDetector.get_attention_module(
                first_layer, arch_config
            )
            if hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
                return attn.q_proj.weight.shape[1]
            elif hasattr(attn, "qkv_proj") and hasattr(attn.qkv_proj, "weight"):
                return attn.qkv_proj.weight.shape[1]
            else:
                raise ValueError("Cannot determine hidden size from model structure")


class TruthX:
    """
    Wrapper per l'editing delle rappresentazioni durante l'inferenza.
    Supporta diversi LLM in modo agnostico.

    TruthX richiede due componenti (come da paper Sezione 3.2-3.3):
    1. Autoencoder (TruthEnc, SemEnc, Dec) per codificare/decodificare
    2. Steering vectors (pos_center, neg_center, rank) per calcolare δ ed editare
    """

    def __init__(
        self,
        autoencoder_path: str,
        steering_vectors_path: str,
        hidden_size: int,
        edit_strength: float = 1.0,
        top_layers: int = 10,
        use_mask: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Carica Autoencoder
        ae_checkpoint = torch.load(autoencoder_path, map_location=self.device)
        args = ae_checkpoint["args"]

        # Se args è un dict, convertilo in namespace per compatibilità
        if isinstance(args, dict):
            from argparse import Namespace

            args = Namespace(**args)

        # Ricostruisci il modello
        self.ae_model = MLPAE(
            in_channels=hidden_size,
            semantic_latent_dim=args.semantic_latent_dim,
            truthful_latent_dim=args.truthful_latent_dim,
            semantic_hidden_dims=[int(x) for x in args.semantic_hidden_dims.split(",")]
            if args.semantic_hidden_dims
            else None,
            truthful_hidden_dims=[int(x) for x in args.truthful_hidden_dims.split(",")]
            if args.truthful_hidden_dims
            else None,
            decoder_hidden_dims=[int(x) for x in args.decoder_hidden_dims.split(",")]
            if args.decoder_hidden_dims
            else None,
        ).to(self.device)

        self.ae_model.load_state_dict(ae_checkpoint["state_dict"])
        self.ae_model.eval()

        # 2. Carica Steering Vectors (pos_center, neg_center, rank)
        # Come da Eq. 12: δ = H̄_truth^pos - H̄_truth^neg
        sv_checkpoint = torch.load(steering_vectors_path, map_location=self.device)

        self.ae_model.pos_center = sv_checkpoint["pos_center"].to(self.device)
        self.ae_model.neg_center = sv_checkpoint["neg_center"].to(self.device)

        # rank contiene gli indici dei virtual layers ordinati per probing accuracy
        # rank[0] è il miglior layer, rank[1] il secondo, ecc.
        self.rank = sv_checkpoint["rank"]

        # Crea una mappa inversa: virtual_layer_idx -> posizione nel rank
        # Se un layer non è nei top-k, non sarà nella mappa
        self.rank_position = {layer_idx: pos for pos, layer_idx in enumerate(self.rank)}

        # virtual_layer_info: lista di (physical_layer, module_type)
        # Es: [(0, 'attn'), (0, 'mlp'), (1, 'attn'), (1, 'mlp'), ...]
        self.virtual_layer_info = sv_checkpoint.get("virtual_layer_info", [])

        self.top_layers = top_layers
        self.edit_strength = edit_strength
        self.cur_layer_id = "0"
        self.prompt_length = None
        self.mc = False
        self.use_mask = use_mask
        self.mask_type = None  # Può essere 'last_token', 'after_prompt', 'probing'

    @torch.inference_mode()
    def edit(self, X: Tensor, physical_layer: int, module_type: str) -> Tensor:
        """
        Versione con maschere opzionali.
        """
        # Trova il virtual layer index
        virtual_layer_idx = None
        for idx, (pl, mt) in enumerate(self.virtual_layer_info):
            if pl == physical_layer and mt == module_type:
                virtual_layer_idx = idx
                break

        if virtual_layer_idx is None:
            return X

        if virtual_layer_idx not in self.rank_position:
            return X

        rank_pos = self.rank_position[virtual_layer_idx]
        if rank_pos >= self.top_layers:
            return X

        bsz, s_len, d = X.size()
        x = X.contiguous().view(-1, d).type_as(self.ae_model.semantic_encoder[0].weight)

        x_truthful = self.ae_model.get_truthful_latent_rep(x)

        pos_center = self.ae_model.pos_center[virtual_layer_idx].unsqueeze(0)
        neg_center = self.ae_model.neg_center[virtual_layer_idx].unsqueeze(0)
        delta = (pos_center - neg_center).unsqueeze(0)
        recon_x_pos = (
            self.ae_model(
                x,
                truthful_latent_rep=F.normalize(
                    x_truthful + delta, p=2, dim=-1
                ).type_as(x),
            )[0]
            .contiguous()
            .view(bsz, s_len, d)
        )
        
        recon_x_neg = (
            self.ae_model(
                x,
                truthful_latent_rep=F.normalize(
                    x_truthful - delta, p=2, dim=-1
                ).type_as(x),
            )[0]
            .contiguous()
            .view(bsz, s_len, d)
        )

        Delta = (recon_x_pos - recon_x_neg).contiguous().to(X.dtype)

        if not self.use_mask:
            # Versione base: Eq. 14 senza maschere
            new_X = X + self.edit_strength * Delta.type_as(X)
        else:
            # Crea maschera
            mask = torch.ones((bsz, s_len), device=Delta.device)
            
            if self.mask_type == "last_token":
                # Edita solo l'ultimo token
                mask[:, :-1] = 0
            elif self.mask_type == "after_prompt" and self.prompt_length is not None:
                # Edita solo dopo il prompt
                mask[:, : self.prompt_length + 1] = 0
            elif self.mask_type == "probing":
                # Usa probing scores per selezionare token
                probing = (
                    torch.nn.functional.cosine_similarity(
                        x_truthful, neg_center.unsqueeze(1), dim=-1
                    )
                    - torch.nn.functional.cosine_similarity(
                        x_truthful, pos_center.unsqueeze(1), dim=-1
                    )
                ).clamp(0, 999)
                mask = mask * probing
            
            new_X = X + self.edit_strength * Delta.type_as(X) * mask.unsqueeze(2).type_as(X)
        
        return new_X
