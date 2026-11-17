from transformers import AutoTokenizer, pipeline, AutoModel
import torch

BATCH_SIZE = 128
MODEL_NAME = "BAAI/bge-small-en-v1.5"

if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Macbook M
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()


# We are only doing inference, so we don't need gradients
@torch.no_grad()
def encode_in_batches(texts: list[str]) -> torch.Tensor:
    all_vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**encoded)

        # Normalize the vectors
        cls_vectors = torch.nn.functional.normalize(cls_vectors, p=2, dim=1)
        cls_vectors = outputs.last_hidden_state[:, 0, :].cpu()

        all_vectors.append(cls_vectors)

    return torch.cat(all_vectors, dim=0)
