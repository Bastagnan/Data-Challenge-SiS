import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, text_embed_dim=32, motion_dim=6600):
        super().__init__()
        self.fc1 = nn.Linear(text_embed_dim, 512)
        self.fc2 = nn.Linear(512, 1028)
        self.fc3 = nn.Linear(1028, motion_dim)
        self.relu = nn.ReLU()

    def forward(self, text_emb):
        """
        text_emb: (batch_size, text_embed_dim)
        returns motion: (batch_size, motion_dim)
        """

        x = self.relu(self.fc1(text_emb))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x