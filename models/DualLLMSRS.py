# here put the import lib
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GRU4Rec import GRU4Rec
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.utils import Multi_CrossAttention


class LLMSemanticRerankerMixin:
    """Mean-centered recency-aware semantic residual reranking (MCRR).

    MCRR is deliberately kept outside the training loss.  It builds a user
    semantic intent directly from the frozen raw LLM item table and adds the
    resulting cosine score as a residual to any sequential backbone.  The
    three internal mechanisms can be switched independently for ablations:

    C: catalog-mean centering of raw LLM item vectors;
    R: recency-aware rather than mean history pooling;
    Z: candidate-wise score standardization before residual fusion.
    """

    def _init_semantic_reranker(self, args, llm_item_emb):
        self.use_semantic_rerank = not getattr(
            args, "disable_semantic_rerank", False
        )
        self.use_semantic_centering = not getattr(
            args, "disable_semantic_centering", False
        )
        self.semantic_pooling = getattr(args, "semantic_pooling", "recency")
        if self.semantic_pooling not in {"recency", "mean"}:
            raise ValueError(
                "semantic_pooling must be either 'recency' or 'mean'"
            )
        self.use_candidate_standardization = not getattr(
            args, "disable_candidate_standardization", False
        )
        self.semantic_recency_decay = max(
            float(getattr(args, "semantic_recency_decay", 0.2)), 0.0
        )

        # The semantic residual must use the same pretrained geometry in every
        # ablation.  Keep an immutable copy instead of the trainable embedding
        # module used by the original LLM-ESR backbone.
        semantic_item_table = torch.as_tensor(
            llm_item_emb, dtype=torch.float32
        ).clone()
        self.register_buffer(
            "semantic_item_table", semantic_item_table, persistent=False
        )
        self.register_buffer(
            "semantic_item_center",
            semantic_item_table[1:self.item_num + 1].mean(dim=0),
            persistent=False,
        )
        self.register_buffer(
            "semantic_rerank_weight",
            torch.tensor(
                float(getattr(args, "semantic_rerank_weight", 1.0)),
                dtype=torch.float32,
            ),
        )

    def set_semantic_rerank_weight(self, value):
        self.semantic_rerank_weight.fill_(float(value))

    @staticmethod
    def _standardize_candidate_scores(scores):
        centered = scores - scores.mean(dim=-1, keepdim=True)
        scale = centered.square().mean(dim=-1, keepdim=True).sqrt()
        return centered / scale.clamp_min(1e-6)

    def _semantic_vectors(self, item_ids):
        vectors = F.embedding(item_ids.long(), self.semantic_item_table)
        if self.use_semantic_centering:
            vectors = vectors - self.semantic_item_center
        return F.normalize(vectors, dim=-1, eps=1e-8)

    def semantic_rerank_scores(self, seq, item_indices):
        """Return history-to-candidate cosine scores.

        Ordinal distances are computed only over valid interactions, so left
        padding and internal gaps cannot change an event's recency weight.
        """
        valid_history = seq.gt(0) & seq.le(self.item_num)
        history = self._semantic_vectors(seq)
        candidates = self._semantic_vectors(item_indices)

        if self.semantic_pooling == "mean":
            weights = valid_history.to(dtype=history.dtype)
        else:
            valid_count = valid_history.sum(dim=1, keepdim=True)
            ordinal = valid_history.cumsum(dim=1)
            distance = (valid_count - ordinal).clamp_min(0).to(history.dtype)
            weights = torch.exp(-self.semantic_recency_decay * distance)
            weights = weights * valid_history

        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        profile = (history * weights.unsqueeze(-1)).sum(dim=1)
        profile = F.normalize(profile, dim=-1, eps=1e-8)

        if candidates.dim() == 2:
            # A shared full-catalog candidate matrix: [items, dimension].
            semantic_scores = profile.matmul(candidates.t())
        else:
            # Per-user sampled candidates: [batch, candidates, dimension].
            semantic_scores = torch.einsum("bcd,bd->bc", candidates, profile)

        return semantic_scores.masked_fill(
            ~valid_history.any(dim=1, keepdim=True), 0.0
        )

    def prepare_fusion_scores(self, base_scores, semantic_scores):
        if self.use_candidate_standardization:
            base_scores = self._standardize_candidate_scores(base_scores)
            semantic_scores = self._standardize_candidate_scores(
                semantic_scores
            )
        return base_scores, semantic_scores

    def fuse_semantic_rerank_scores(
        self, base_scores, semantic_scores, weight=None
    ):
        if weight is None:
            weight = self.semantic_rerank_weight
        base_scores, semantic_scores = self.prepare_fusion_scores(
            base_scores, semantic_scores
        )
        return base_scores + weight * semantic_scores

    def predict(
        self,
        seq,
        item_indices,
        positions,
        return_semantic_components=False,
        **kwargs,
    ):
        base_scores = super().predict(
            seq=seq,
            item_indices=item_indices,
            positions=positions,
            **kwargs,
        )
        if not self.use_semantic_rerank and not return_semantic_components:
            return base_scores

        semantic_scores = self.semantic_rerank_scores(seq, item_indices)
        if return_semantic_components:
            return base_scores, semantic_scores
        return self.fuse_semantic_rerank_scores(base_scores, semantic_scores)



class DualLLMGRU4Rec(LLMSemanticRerankerMixin, GRU4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        self._init_semantic_reranker(args, llm_item_emb)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs):

        id_seqs = self.id_item_emb(log_seqs)
        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
    


class DualLLMSASRec(LLMSemanticRerankerMixin, SASRec_seq):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        self._init_semantic_reranker(args, llm_item_emb)
        # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):

        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
            cross_id_seqs = 1 * cross_id_seqs + 0 * id_seqs
            cross_llm_seqs = 1 * cross_llm_seqs + 0 * llm_seqs
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
    


class DualLLMBert4Rec(LLMSemanticRerankerMixin, Bert4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        self._init_semantic_reranker(args, llm_item_emb)
        # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):

        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
