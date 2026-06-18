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
        vocab_size=32000, # default BPE vocab size dari Rust
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
    Menerjemahkan min(1.0, (U * beta) + dot) dan mekanisme reset.
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
        
        # Inisialisasi bobot (nantinya bisa diload dari model_weights.json)
        self.embedding_weights = nn.Parameter(torch.randn(self.vocab_size, self.d_model))
        self.emb_beta = nn.Parameter(torch.ones(self.d_model) * config.beta)
        self.emb_threshold = nn.Parameter(torch.ones(self.d_model) * 0.5)

        self.kernel_q = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.kernel_k = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.kernel_v = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.att_beta = nn.Parameter(torch.ones(self.d_model) * config.beta)
        self.att_threshold = nn.Parameter(torch.ones(self.d_model) * 0.2)
        
        self.att_beta_scores = nn.Parameter(torch.ones(self.max_seq_len) * 0.9)
        self.att_threshold_scores = nn.Parameter(torch.ones(self.max_seq_len) * 1.0)
        
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
                
            att = data.get("attention", {})
            if "kernel_q" in att: self.kernel_q.data = torch.tensor(att["kernel_q"]).reshape(self.d_model, self.d_model)
            if "kernel_k" in att: self.kernel_k.data = torch.tensor(att["kernel_k"]).reshape(self.d_model, self.d_model)
            if "kernel_v" in att: self.kernel_v.data = torch.tensor(att["kernel_v"]).reshape(self.d_model, self.d_model)
            if "beta_qkv" in att: self.att_beta.data = torch.tensor(att["beta_qkv"])
            if "threshold_qkv" in att: self.att_threshold.data = torch.tensor(att["threshold_qkv"])
            if "beta_scores" in att: self.att_beta_scores.data = torch.tensor(att["beta_scores"])
            if "threshold_scores" in att: self.att_threshold_scores.data = torch.tensor(att["threshold_scores"])
            
            pool = data.get("pooler", {})
            if "kernel" in pool: self.pooler_weights.data = torch.tensor(pool["kernel"]).reshape(self.d_model, self.d_model)
            if "bias" in pool: self.pooler_bias.data = torch.tensor(pool["bias"])
            if "beta" in pool: self.pooler_beta.data = torch.tensor(pool["beta"])
            if "threshold" in pool: self.pooler_threshold.data = torch.tensor(pool["threshold"])
            print("Successfully loaded SNN native weights into PyTorch!")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # =====================================================================
        # 1. LAYER EMBEDDING
        # =====================================================================
        emb_dot = F.embedding(input_ids, self.embedding_weights) # [batch, seq, d_model]
        
        # Karena Embedding di Rust beroperasi per token, kita asumsikan 1 step
        emb_potentials = torch.zeros_like(emb_dot)
        emb_potentials, emb_spikes, _ = lif_step_pytorch(emb_potentials, emb_dot, self.emb_beta, self.emb_threshold)
        
        # =====================================================================
        # 2. LAYER COINCIDENCE SELF-ATTENTION
        # =====================================================================
        # Proyeksi Dense -> Dot Product
        dot_q = torch.matmul(emb_spikes, self.kernel_q) # [batch, seq, d_model]
        dot_k = torch.matmul(emb_spikes, self.kernel_k)
        dot_v = torch.matmul(emb_spikes, self.kernel_v)
        
        att_potentials_q = torch.zeros_like(dot_q)
        att_potentials_k = torch.zeros_like(dot_k)
        att_potentials_v = torch.zeros_like(dot_v)
        
        # LIF Step untuk Q, K, V
        _, sq, _ = lif_step_pytorch(att_potentials_q, dot_q, self.att_beta, self.att_threshold)
        _, sk, _ = lif_step_pytorch(att_potentials_k, dot_k, self.att_beta, self.att_threshold)
        _, sv, _ = lif_step_pytorch(att_potentials_v, dot_v, self.att_beta, self.att_threshold)
        
        # Menghitung Coincidence Match Scores secara paralel (pengganti loop C++)
        # sq: [batch, seq, d_model], sk: [batch, seq, d_model]
        # BMM (Batch Matrix Multiply) untuk menghitung spike yang bertepatan
        match_counts = torch.bmm(sq, sk.transpose(1, 2)) # [batch, seq, seq]
        
        # Normalisasi terhadap jumlah maksimum match per baris
        max_matches = match_counts.max(dim=-1, keepdim=True)[0]
        max_matches = torch.where(max_matches > 0, max_matches, torch.ones_like(max_matches))
        match_scores = match_counts / max_matches
        
        # Skor akhir (Graded Score)
        graded_scores_potentials = torch.zeros_like(match_scores)
        b_scores = self.att_beta_scores[:seq_len].unsqueeze(0).unsqueeze(0)
        t_scores = self.att_threshold_scores[:seq_len].unsqueeze(0).unsqueeze(0)
        _, graded_spikes, _ = lif_step_pytorch(graded_scores_potentials, match_scores, b_scores, t_scores)
        
        # Modulasi V dengan Graded Score
        att_out = torch.bmm(graded_spikes, sv) # [batch, seq, d_model]
        
        if getattr(self, '_debug_print', False):
            print(f"PYTORCH att_out SUM: {att_out[0].sum().item():.4f}")
            self._debug_print = False
        
        # Residu: Output Embedding + Attention
        aggregated_features = emb_spikes + att_out
        
        # Menerapkan attention_mask agar PAD tokens menjadi 0.0 (meniru logika 't < actual_lengths[b]' di Rust)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            aggregated_features = aggregated_features * mask
        
        # =====================================================================
        # 3. LAYER TEMPORAL POOLER (BPTT Dense)
        # =====================================================================
        # Agar identik 100% dengan Rust, iterasi pooler HARUS berjalan sampai max_seq_len.
        # Walaupun input batch lebih pendek, neuron tetap akan meluruh (decay) ke max_seq_len.
        pooler_potentials = torch.zeros(batch_size, self.d_model, device=device)
        final_embeddings = torch.zeros(batch_size, self.d_model, device=device)
        
        for t in range(self.max_seq_len):
            if t < seq_len:
                step_input = aggregated_features[:, t, :]
            else:
                step_input = torch.zeros(batch_size, self.d_model, device=device)
                
            dot_pool = torch.matmul(step_input, self.pooler_weights) + self.pooler_bias
            pooler_potentials, _, current_p = lif_step_pytorch(pooler_potentials, dot_pool, self.pooler_beta, self.pooler_threshold)
            
            # Sesuai dengan iterasi 't < actual_lengths[b]' di Rust, penjumlahan ke final_embeddings harus berhenti pada saat padding
            if attention_mask is not None and t < seq_len:
                mask_t = attention_mask[:, t].unsqueeze(-1).float()
                final_embeddings += current_p * mask_t
        # =====================================================================
        # 4. L2 NORMALIZATION (METRIC SPACE)
        # =====================================================================
        final_embeddings = F.normalize(final_embeddings, p=2, dim=1)
        
        return final_embeddings
