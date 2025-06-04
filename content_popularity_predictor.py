# content_popularity_predictor.py
import torch
import torch.nn as nn

class ContentPopularityPredictor(nn.Module):
    def __init__(self, num_contents, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_size=num_contents, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_contents)
        self.hidden_state = None

    def predict_popularity(self, prev_content_vector):
        inp = torch.from_numpy(prev_content_vector).float().unsqueeze(0).unsqueeze(0)
        out, self.hidden_state = self.gru(inp, self.hidden_state)
        logits = self.fc(out[:, -1, :])
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().numpy().flatten()
