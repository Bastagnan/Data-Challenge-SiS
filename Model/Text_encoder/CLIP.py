import clip
import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    """
    A drop-in replacement for your text encoder that uses CLIP.
    The pipeline calls this with signature CLIPTextEncoder(vocab_size, embed_dim).
    We ignore vocab_size, but use embed_dim to define a projection layer if desired.
    """
    def __init__(self, vocab_size=128, embed_dim=32, model_name="ViT-B/32"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP; we only need the text model part
        self.clip_model, _ = clip.load(model_name, device=self.device)
        # By default, CLIP's text encoder outputs a 512-dim vector (for ViT-B/32).
        self.clip_dim = self.clip_model.ln_final.weight.shape[0]  # 512 for ViT-B/32
        self.embed_dim = embed_dim

        # Optional: A learnable linear projection from CLIP dimension -> embed_dim
        # If embed_dim == clip_dim, you can skip this projection
        self.proj = nn.Linear(self.clip_dim, embed_dim)

    def forward(self, text_list):
        """
        Args:
            text_list: A list of strings (batch of text prompts).
        Returns:
            text_emb: A tensor of shape (batch_size, embed_dim).
        """
        print("aaaaaaa")
        # 1) Tokenize the text (list of strings)
        tokens = clip.tokenize(text_list, truncate=True).to(self.device)
        
        # 2) Extract text features from CLIP
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        # text_features shape: (batch_size, clip_dim)

        # 3) (Optional) Normalize the CLIP embedding
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        print(type(text_features))

        # 4) Project to the desired embed_dim
        text_emb = self.proj(text_features)  # (batch_size, embed_dim)
        
        return text_emb
