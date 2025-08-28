from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer using standard transformers
model = AutoModelForCausalLM.from_pretrained(
    "majid230/gemma-3-1b-finetune",
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu",           # Force CPU usage
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("majid230/gemma-3-finetune")

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def simple_chatbot():
    print("Gemma Chatbot started! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        # Format input for Gemma3
        messages = [
            {"role": "user", "content": user_input}
        ]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # ==== KEY CHANGE: decode only newly generated tokens ====
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        assistant_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # ========================================================

        print(f"\nAssistant: {assistant_response}\n")

# Start the chatbot
if __name__ == "__main__":
    simple_chatbot()
