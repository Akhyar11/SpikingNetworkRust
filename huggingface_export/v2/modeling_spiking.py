import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

class SpikingConfig(PretrainedConfig):
    model_type = "spiking_snn"

    def __init__(
        self,
        d_model=384,
        max_position_embeddings=128,
        vocab_size=32000, # default BPE vocab size
        neuron_type="homogeneous_lif",
        beta=0.90,
        margin=0.05,
        **kwargs
    ):
        self.d_model = d_model
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.neuron_type = neuron_type
        self.beta = beta
        self.margin = margin
        super().__init__(**kwargs)

def lif_step_pytorch(potentials, dot_product, beta, threshold):
    """
    Fungsi LIF dasar yang dijalankan secara paralel melalui PyTorch.
    Menerjemahkan min(1.0, (U * beta) + dot) dan mekanisme soft-reset.
    """
    current_p = torch.clamp((potentials * beta) + dot_product, max=1.0)
    spikes = (current_p >= threshold).float()
    potentials = torch.where(spikes > 0, current_p - threshold, current_p)
    return potentials, spikes, current_p

class SpikingSentenceEmbedder(PreTrainedModel):
    config_class = SpikingConfig

    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.d_model
        self.max_seq_len = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        
        # Inisialisasi bobot (nantinya di-load dari model_weights.json)
        self.embedding_weights = nn.Parameter(torch.randn(self.vocab_size, self.d_model))
        self.emb_beta = nn.Parameter(torch.ones(self.d_model) * config.beta)
        self.emb_threshold = nn.Parameter(torch.ones(self.d_model) * 0.5)

        # Hapus Spatial Attention secara keseluruhan (Sesuai Paper V2)
        
        self.pooler_weights = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.pooler_bias = nn.Parameter(torch.zeros(self.d_model))
        self.pooler_beta = nn.Parameter(torch.ones(self.d_model) * config.beta)
        self.pooler_threshold = nn.Parameter(torch.ones(self.d_model) * 0.8)
        
        # Load JSON weights if available
        self._load_from_json()

    def _load_from_json(self):
        # Look for model_weights.json in the same directory
        base_dir = self.config.name_or_path if hasattr(self.config, 'name_or_path') else "."
        weight_path = os.path.join(base_dir, "model_weights.json")
        if not os.path.exists(weight_path):
            weight_path = "model_weights.json"
        
        if os.path.exists(weight_path):
            print(f"Loading Spiking Network weights from {weight_path}...")
            with open(weight_path, "r") as f:
                data = json.load(f)
                
            emb = data.get("embedding", {})
            if "weights" in emb:
                w = torch.tensor(emb["weights"]).reshape(-1, self.d_model)
                v_size = min(self.vocab_size, w.shape[0])
                self.embedding_weights.data[:v_size] = w[:v_size]
            if "beta" in emb: self.emb_beta.data = torch.tensor(emb["beta"])
            if "threshold" in emb: self.emb_threshold.data = torch.tensor(emb["threshold"])
                
            # Skip attention weights entirely as they are not used in V2 Recurrent Pooler
            
            pool = data.get("pooler", {})
            if "kernel" in pool: self.pooler_weights.data = torch.tensor(pool["kernel"]).reshape(self.d_model, self.d_model)
            if "bias" in pool: self.pooler_bias.data = torch.tensor(pool["bias"])
            if "beta" in pool: self.pooler_beta.data = torch.tensor(pool["beta"])
            if "threshold" in pool: self.pooler_threshold.data = torch.tensor(pool["threshold"])
            print("Successfully loaded SNN native weights into PyTorch (V2 - Attention Free)!")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # =====================================================================
        # 1. LAYER EMBEDDING
        # =====================================================================
        emb_dot = F.embedding(input_ids, self.embedding_weights) # [batch, seq, d_model]
        
        emb_potentials = torch.zeros_like(emb_dot)
        emb_potentials, emb_spikes, _ = lif_step_pytorch(emb_potentials, emb_dot, self.emb_beta, self.emb_threshold)
        
        # =====================================================================
        # 2. LAYER TEMPORAL POOLER (BPTT Dense)
        # Bypassing spatial attention entirely.
        # =====================================================================
        # Menerapkan attention_mask agar PAD tokens menjadi 0.0 sebelum masuk pooler
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            emb_spikes = emb_spikes * mask
            
        pooler_potentials = torch.zeros(batch_size, self.d_model, device=device)
        final_embeddings = torch.zeros(batch_size, self.d_model, device=device)
        
        # Iterasi BPTT
        for t in range(self.max_seq_len):
            if t < seq_len:
                step_input = emb_spikes[:, t, :]
            else:
                step_input = torch.zeros(batch_size, self.d_model, device=device)
                
            dot_pool = torch.matmul(step_input, self.pooler_weights) + self.pooler_bias
            pooler_potentials, _, current_p = lif_step_pytorch(pooler_potentials, dot_pool, self.pooler_beta, self.pooler_threshold)
            
            # Sum up fractional potentials (as described in the paper)
            if attention_mask is not None and t < seq_len:
                mask_t = attention_mask[:, t].unsqueeze(-1).float()
                final_embeddings += current_p * mask_t
                
        # =====================================================================
        # 3. L2 NORMALIZATION (METRIC SPACE)
        # =====================================================================
        final_embeddings = F.normalize(final_embeddings, p=2, dim=1)
        
        return final_embeddings
