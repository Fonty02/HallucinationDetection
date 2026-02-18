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


class ResidualMLPAE(nn.Module):
    """
    MLP-based Autoencoder per TruthX con Residual Connections, Dropout e LayerNorm.

    Migliora la convergenza rispetto a MLPAE aggiungendo:
    - Skip/Residual connections tra ogni layer dell'encoder e il corrispondente del decoder
    - Dropout per regolarizzazione
    - LayerNorm per stabilità del training

    Architettura encoder (esempio con hidden_dims=[2048,1024,512], latent_dim=256):
        input_dim -> 2048 -> 1024 -> 512 -> 256 (latent)
    Architettura decoder (simmetrica):
        256 (latent) -> 512 (+skip da encoder layer 2) -> 1024 (+skip da encoder layer 1) -> 2048 (+skip da encoder layer 0) -> input_dim
    """

    def __init__(
        self,
        in_channels: int,
        semantic_latent_dim: int = 1024,
        truthful_latent_dim: int = 1024,
        semantic_hidden_dims: Optional[List[int]] = None,
        truthful_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if semantic_hidden_dims is None:
            semantic_hidden_dims = [2048]
        if truthful_hidden_dims is None:
            truthful_hidden_dims = [2048]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [2048]

        self.dropout_rate = dropout

        # Semantic Encoder (layer-by-layer per catturare attivazioni intermedie)
        self.semantic_encoder_layers = self._build_encoder_layers(
            in_channels, semantic_hidden_dims, semantic_latent_dim, dropout
        )
        self.semantic_encoder_dims = [in_channels] + semantic_hidden_dims + [semantic_latent_dim]

        # Truthful Encoder
        self.truthful_encoder_layers = self._build_encoder_layers(
            in_channels, truthful_hidden_dims, truthful_latent_dim, dropout
        )
        self.truthful_encoder_dims = [in_channels] + truthful_hidden_dims + [truthful_latent_dim]

        # Cross-Attention Module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=semantic_latent_dim, num_heads=1
        )

        self.proj = None
        if semantic_latent_dim != truthful_latent_dim:
            self.proj = nn.Linear(truthful_latent_dim, semantic_latent_dim, bias=False)

        # Decoder (layer-by-layer per ricevere skip connections)
        # decoder_hidden_dims va dal latent verso l'output
        # Le skip connections vengono aggiunte dal corrispondente encoder layer
        self.decoder_layers, self.decoder_skip_projs = self._build_decoder_layers(
            semantic_latent_dim, decoder_hidden_dims, semantic_hidden_dims, dropout
        )
        decoder_out_dim = decoder_hidden_dims[-1] if decoder_hidden_dims else semantic_latent_dim
        self.final_layer = nn.Linear(decoder_out_dim, in_channels)

        self.semantic_latent_dim = semantic_latent_dim
        self.truthful_latent_dim = truthful_latent_dim

        self.register_buffer("pos_center", None)
        self.register_buffer("neg_center", None)

    def _build_encoder_layers(self, in_dim, hidden_dims, latent_dim, dropout):
        """Costruisce una lista di layer dell'encoder (ModuleList)."""
        layers = nn.ModuleList()
        all_dims = hidden_dims + [latent_dim]
        current_dim = in_dim
        for h_dim in all_dims:
            layers.append(nn.Sequential(
                nn.Linear(current_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ))
            current_dim = h_dim
        return layers

    def _build_decoder_layers(self, latent_dim, decoder_hidden_dims, encoder_hidden_dims, dropout):
        """
        Costruisce i layer del decoder con proiezioni per le skip connections.

        La skip connection al decoder layer i (che produce dim decoder_hidden_dims[i])
        viene dall'encoder layer corrispondente (in ordine inverso).

        Es. encoder: [in -> 2048 -> 1024 -> 512 -> 256(latent)]
            decoder: [256 -> 512(+skip da 512) -> 1024(+skip da 1024) -> 2048(+skip da 2048) -> in]
        """
        layers = nn.ModuleList()
        skip_projs = nn.ModuleList()

        # Dimensioni delle attivazioni intermedie dell'encoder (escluso latent e input):
        # Es. encoder_hidden_dims = [2048, 1024, 512] -> reversed = [512, 1024, 2048]
        encoder_skip_dims = list(reversed(encoder_hidden_dims))

        current_dim = latent_dim
        for i, h_dim in enumerate(decoder_hidden_dims):
            layers.append(nn.Sequential(
                nn.Linear(current_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ))
            # Proiezione per skip connection (se c'è un matching encoder layer)
            if i < len(encoder_skip_dims):
                skip_dim = encoder_skip_dims[i]
                if skip_dim != h_dim:
                    skip_projs.append(nn.Linear(skip_dim, h_dim))
                else:
                    skip_projs.append(nn.Identity())
            else:
                skip_projs.append(None)
            current_dim = h_dim

        return layers, skip_projs

    def _encode_with_intermediates(self, x: Tensor, encoder_layers: nn.ModuleList):
        """Esegue l'encoder salvando le attivazioni intermedie per le skip connections."""
        intermediates = []
        h = x
        for layer in encoder_layers:
            h = layer(h)
            intermediates.append(h)
        # intermediates[-1] è il latent
        return h, intermediates[:-1]  # latent, [hidden_0, hidden_1, ...]

    def decode(self, z: Tensor, skip_intermediates: Optional[List[Tensor]] = None) -> Tensor:
        """Decode con residual/skip connections dall'encoder."""
        # skip_intermediates: attivazioni encoder in ordine [hidden_0, hidden_1, ...]
        # Per il decoder, servono in ordine inverso: [hidden_n-1, hidden_n-2, ...]
        if skip_intermediates is not None:
            reversed_skips = list(reversed(skip_intermediates))
        else:
            reversed_skips = []

        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
            # Aggiungi skip connection se disponibile
            if i < len(reversed_skips) and i < len(self.decoder_skip_projs):
                proj = self.decoder_skip_projs[i]
                if proj is not None:
                    skip = proj(reversed_skips[i])
                    h = h + skip

        h = self.final_layer(h)
        return h

    def encode_semantic(self, x: Tensor):
        """Encoder semantico, ritorna (latent, intermediates)."""
        return self._encode_with_intermediates(x, self.semantic_encoder_layers)

    def encode_truthful(self, x: Tensor) -> Tensor:
        """Encoder truthful, ritorna solo il latent (L2-normalized)."""
        z, _ = self._encode_with_intermediates(x, self.truthful_encoder_layers)
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
        semantic_rep, sem_intermediates = self.encode_semantic(x)

        if truthful_latent_rep is None:
            truthful_rep = self.encode_truthful(x)
        else:
            truthful_rep = truthful_latent_rep
            if truthful_rep.dim() == 3:
                truthful_rep = truthful_rep.reshape(-1, truthful_rep.size(-1))

        z = semantic_rep + self.attention(semantic_rep, truthful_rep, truthful_rep)
        output = self.decode(z, skip_intermediates=sem_intermediates)

        return [output, x, semantic_rep, truthful_rep]

    def get_semantic_latent_rep(self, x: Tensor) -> Tensor:
        z, _ = self.encode_semantic(x)
        return z

    def get_truthful_latent_rep(self, x: Tensor) -> Tensor:
        return self.encode_truthful(x)

    def loss_function(self, recons, input):
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
        pre_aligner=None,
        post_aligner=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Carica Autoencoder
        ae_checkpoint = torch.load(autoencoder_path, map_location=self.device)
        args = ae_checkpoint["args"]
        
        # Per esperimenti cross-model, leggi hidden_size dal checkpoint invece del modello corrente
        if pre_aligner is not None or post_aligner is not None:
            # L'autoencoder è stato trainato con la dimensione del trainer model
            # Leggi la dimensione corretta dai pesi salvati
            state_dict = ae_checkpoint["state_dict"]
            # semantic_encoder.0.weight ha shape [semantic_latent_dim, hidden_size]
            checkpoint_hidden_size = state_dict["semantic_encoder.0.weight"].shape[1]
            print(f"  CrossModel: usando hidden_size={checkpoint_hidden_size} dal checkpoint (invece di {hidden_size} dal modello corrente)")
            hidden_size = checkpoint_hidden_size

        # Se args è un dict, convertilo in namespace per compatibilità
        if isinstance(args, dict):
            from argparse import Namespace

            args = Namespace(**args)

        # Ricostruisci il modello - scegli architettura in base agli args salvati
        use_residual = getattr(args, "residual", False)
        dropout_rate = getattr(args, "dropout", 0.1)

        semantic_hidden_dims = (
            [int(x) for x in args.semantic_hidden_dims.split(",")]
            if args.semantic_hidden_dims
            else None
        )
        truthful_hidden_dims = (
            [int(x) for x in args.truthful_hidden_dims.split(",")]
            if args.truthful_hidden_dims
            else None
        )
        decoder_hidden_dims = (
            [int(x) for x in args.decoder_hidden_dims.split(",")]
            if args.decoder_hidden_dims
            else None
        )

        if use_residual:
            self.ae_model = ResidualMLPAE(
                in_channels=hidden_size,
                semantic_latent_dim=args.semantic_latent_dim,
                truthful_latent_dim=args.truthful_latent_dim,
                semantic_hidden_dims=semantic_hidden_dims,
                truthful_hidden_dims=truthful_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
                dropout=dropout_rate,
            ).to(self.device)
        else:
            self.ae_model = MLPAE(
                in_channels=hidden_size,
                semantic_latent_dim=args.semantic_latent_dim,
                truthful_latent_dim=args.truthful_latent_dim,
                semantic_hidden_dims=semantic_hidden_dims,
                truthful_hidden_dims=truthful_hidden_dims,
                decoder_hidden_dims=decoder_hidden_dims,
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
        
        # Aligner per cross-model editing
        self.pre_aligner = pre_aligner
        self.post_aligner = post_aligner

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
        # Handle both MLPAE and ResidualMLPAE architectures
        if hasattr(self.ae_model, 'semantic_encoder'):
            weight = self.ae_model.semantic_encoder[0].weight
        else:
            weight = self.ae_model.semantic_encoder_layers[0][0].weight
        x = X.contiguous().view(-1, d).type_as(weight)

        # Apply pre-aligner (Target → Trainer space) for cross-model editing
        if self.pre_aligner is not None:
            x_np = x.detach().cpu().numpy()
            x_aligned = self.pre_aligner.predict(x_np)
            x = torch.from_numpy(x_aligned).to(X.device).type_as(weight)

        # d_ae: dimensione nello spazio dell'autoencoder (può differire da d in cross-model)
        d_ae = x.shape[-1]

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
            .view(bsz, s_len, d_ae)
        )
        
        recon_x_neg = (
            self.ae_model(
                x,
                truthful_latent_rep=F.normalize(
                    x_truthful - delta, p=2, dim=-1
                ).type_as(x),
            )[0]
            .contiguous()
            .view(bsz, s_len, d_ae)
        )

        Delta = (recon_x_pos - recon_x_neg).contiguous()

        # Per cross-model: l'addizione avviene nello spazio del trainer (d_ae),
        # poi il post-aligner rimappa nello spazio del target (d).
        # Per same-model: d_ae == d, l'addizione avviene direttamente su X.
        if self.pre_aligner is not None:
            # Cross-model: somma nello spazio del trainer
            x_3d = x.view(bsz, s_len, d_ae)
            if not self.use_mask:
                new_X_aligned = x_3d + self.edit_strength * Delta.type_as(x_3d)
            else:
                mask = torch.ones((bsz, s_len), device=Delta.device)
                if self.mask_type == "last_token":
                    mask[:, :-1] = 0
                elif self.mask_type == "after_prompt" and self.prompt_length is not None:
                    mask[:, : self.prompt_length + 1] = 0
                elif self.mask_type == "probing":
                    probing = (
                        torch.nn.functional.cosine_similarity(
                            x_truthful, neg_center.unsqueeze(1), dim=-1
                        )
                        - torch.nn.functional.cosine_similarity(
                            x_truthful, pos_center.unsqueeze(1), dim=-1
                        )
                    ).clamp(0, 999)
                    mask = mask * probing
                new_X_aligned = x_3d + self.edit_strength * Delta.type_as(x_3d) * mask.unsqueeze(2).type_as(x_3d)
            
            # Post-aligner: Trainer → Target space
            if self.post_aligner is not None:
                new_X_flat = new_X_aligned.contiguous().view(-1, d_ae)
                new_X_np = new_X_flat.detach().cpu().numpy()
                new_X_back = self.post_aligner.predict(new_X_np)
                new_X = torch.from_numpy(new_X_back).to(X.device).type_as(X).view(bsz, s_len, d)
            else:
                new_X = new_X_aligned
        else:
            # Same-model: somma direttamente su X nello spazio originale
            Delta = Delta.to(X.dtype)
            if not self.use_mask:
                new_X = X + self.edit_strength * Delta.type_as(X)
            else:
                mask = torch.ones((bsz, s_len), device=Delta.device)
                if self.mask_type == "last_token":
                    mask[:, :-1] = 0
                elif self.mask_type == "after_prompt" and self.prompt_length is not None:
                    mask[:, : self.prompt_length + 1] = 0
                elif self.mask_type == "probing":
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
