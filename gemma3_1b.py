import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "majid230/gemma-3-1b-finetune"  # Your Hugging Face repo
USE_AUTH_TOKEN = None  # Add your token if the repo is private

# Force CPU
device = torch.device("cpu")

# ---------------------------
# LOAD MODEL & TOKENIZER
# ---------------------------
print("[INFO] Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=USE_AUTH_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # Ensure CPU-compatible dtype
    device_map=None
)

model.to(device)
model.eval()
print("[INFO] Model loaded successfully!")

# ---------------------------
# CHAT LOOP
# ---------------------------
def chat():
    print("\n=== Gemma 3 1B Finetuned Chatbot (CPU) ===")
    print("Type 'exit' to quit.\n")

    conversation = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting chat...")
            break

        conversation += f"User: {user_input}\nAssistant:"

        inputs = tokenizer(conversation, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=1.0,
                top_p=0.9,
                top_k=50,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the latest assistant reply
        bot_reply = response[len(conversation):].strip()
        print(f"Bot: {bot_reply}\n")

        conversation += f" {bot_reply}\n"

if __name__ == "__main__":
    chat()
