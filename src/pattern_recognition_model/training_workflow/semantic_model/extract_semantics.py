import os

from transformers import AutoTokenizer, pipeline, AutoModel
import torch
from pattern_recognition_model.utils import BASE_DIR, load_dataset
import json

BATCH_SIZE = 128
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DATASET_VERSION = "v1"

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

        # Normalize the vectors. Shortcut for computing cosine similarity between vectors.
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = torch.nn.functional.normalize(cls_vectors, p=2, dim=1).cpu()

        all_vectors.append(cls_vectors)

    return torch.cat(all_vectors, dim=0)


def main():
    """
    Main function to load the dataset and extract user prompts.
    """
    ds = load_dataset(DATASET_VERSION, include_spliced=False)

    user_texts = [ds[i]["user_prompt"] for i in range(len(ds))]  # type: ignore

    tool_names = [ds[i]["tool"] for i in range(len(ds))]  # type: ignore
    ds_args = [ds[i]["args"] for i in range(len(ds))]  # type: ignore

    # Pair tool names with their corresponding arguments
    tool_args_paired = [f"{a} {b}" for a, b in zip(tool_names, ds_args)]  # type: ignore

    prev_tools = [
        (
            ", ".join(map(str, ds[i]["prev_tools"]))  # type: ignore
            if isinstance(ds[i].get("prev_tools"), list)  # type: ignore
            else str(ds[i].get("prev_tools") or "")
        )  # type: ignore
        for i in range(len(ds))
    ]
    read_digest = [
        (
            json.dumps(ds[i]["read_digest"])  # type: ignore
            if isinstance(ds[i].get("read_digest"), (dict, list))  # type: ignore
            else str(ds[i].get("read_digest") or "")
        )  # type: ignore
        for i in range(len(ds))
    ]

    emb_user_texts = encode_in_batches(user_texts)
    emb_tool_args = encode_in_batches(tool_args_paired)
    emb_prev_tools = encode_in_batches(prev_tools)
    emb_read_digest = encode_in_batches(read_digest)

    torch.save(
        emb_user_texts,
        os.path.join(BASE_DIR, "data", DATASET_VERSION, "emb_user_texts.pt"),
    )
    torch.save(
        emb_tool_args,
        os.path.join(BASE_DIR, "data", DATASET_VERSION, "emb_tool_args.pt"),
    )
    torch.save(
        emb_prev_tools,
        os.path.join(BASE_DIR, "data", DATASET_VERSION, "emb_prev_tools.pt"),
    )
    torch.save(
        emb_read_digest,
        os.path.join(BASE_DIR, "data", DATASET_VERSION, "emb_read_digest.pt"),
    )


if __name__ == "__main__":
    main()
