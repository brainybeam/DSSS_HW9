from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import asyncio
import nest_asyncio

nest_asyncio.apply()
TOKEN = "7207301757:AAFfELMAJ9yoyDvFsxlENf6kR1udqPigHnI"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing model...")
tknzr = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(DEVICE_TYPE)
gen = pipeline("text-generation", model=mdl, tokenizer=tknzr, device=0 if torch.cuda.is_available() else -1)
print("Model ready!")

async def greet_user(upd: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await upd.message.reply_text("Hi! How can I assist you?")

async def handle_text(upd: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = upd.message.text
    chat_input = [
        {"role": "system", "content": "You are an intelligent and helpful assistant."},
        {"role": "user", "content": msg},
    ]
    try:
        query = tknzr.apply_chat_template(chat_input, tokenize=False, add_generation_prompt=True)
        result = gen(query, max_new_tokens=250, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        reply = result[0]["generated_text"].split("<|assistant|>")[-1].strip()
        await upd.message.reply_text(reply)
    except Exception as err:
        await upd.message.reply_text("An error occurred. Try again later.")
        print(f"Error: {err}")

def run_bot():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", greet_user))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print("Bot operational...")
    app.run_polling()

if __name__ == "__main__":
    run_bot()
