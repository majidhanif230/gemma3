# Gemma 3 Fine-Tuning

Fine-tuning experiments with Google's Gemma 3 models, using Unsloth for efficient training, plus small scripts that load the resulting fine-tuned models from Hugging Face and chat with them on CPU.

## What it does

- **`Gemma3_(270M).ipynb`** — Supervised fine-tuning (SFT) of `unsloth/gemma-3-270m-it` with LoRA adapters (rank 128) on the first 10,000 examples of the [Thytu/ChessInstruct](https://huggingface.co/datasets/Thytu/ChessInstruct) dataset, using the Gemma 3 chat template. Training runs for 100 steps at batch size 8. The merged model is saved to Hugging Face as `majid230/gemma-3-finetune`.
- **`Gemma3_(1B)_GRPO.ipynb`** — GRPO (reinforcement learning) fine-tuning of `unsloth/gemma-3-1b-it` on OpenAI's [GSM8K](https://huggingface.co/datasets/openai/gsm8k) grade-school math dataset, with a cosine learning-rate schedule and 4 generations per prompt. The merged model is saved to Hugging Face as `majid230/gemma-3-1b-finetune`.
- **`gemma3 270m.py`** — Terminal chatbot that loads `majid230/gemma-3-finetune` with `transformers` and chats on CPU (float32, up to 128 new tokens per reply).
- **`gemma3_1b.py`** — Terminal chatbot that loads `majid230/gemma-3-1b-finetune` with `transformers` and chats on CPU, keeping the conversation history in the prompt.

The notebooks are built on [Unsloth](https://github.com/unslothai/unsloth) templates and are set up to run on a free Tesla T4 Google Colab instance. `nb/` holds an extra copy of the 270M notebook.

## Tech stack

- Python, PyTorch, Hugging Face `transformers`, `datasets`, `trl`
- Unsloth (`FastModel`, LoRA / PEFT, `SFTTrainer`, `GRPOTrainer`)
- Google Colab (Tesla T4) for training; plain CPU for the chatbot scripts

## How to run

### Training notebooks

Open either notebook in Google Colab (a free T4 GPU instance) and choose **Runtime → Run all**. The notebooks install Unsloth and their dependencies themselves.

### Chatbot scripts

Install the dependencies, then run either script and type `quit` / `exit` to stop:

```bash
git clone https://github.com/majidhanif230/gemma3.git
cd gemma3
pip install torch transformers
python "gemma3 270m.py"
python gemma3_1b.py
```

## Project structure

```
gemma3/
├── Gemma3_(270M).ipynb      # SFT fine-tune of Gemma 3 270M on ChessInstruct
├── Gemma3_(1B)_GRPO.ipynb  # GRPO fine-tune of Gemma 3 1B on GSM8K
├── gemma3 270m.py           # CPU chatbot using majid230/gemma-3-finetune
├── gemma3_1b.py             # CPU chatbot using majid230/gemma-3-1b-finetune
└── nb/                      # Extra copy of the 270M notebook
```

## Results

No training metrics (loss curves, evaluation scores) are stored in this repository, so none are claimed here.

## Author

**Majid Hanif** — AI Engineer, BS Artificial Intelligence, The Islamia University of Bahawalpur (CGPA 3.81/4.0)

- Portfolio: https://snap-portfolio-me.lovable.app/portfolio
- LinkedIn: https://www.linkedin.com/in/majid-hanif-23720b2a2/
- GitHub: https://github.com/majidhanif230
