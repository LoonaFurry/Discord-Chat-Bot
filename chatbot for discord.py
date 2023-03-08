import discord
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Connect to database
conn = sqlite3.connect("conversations.db")
c = conn.cursor()

# Create conversations table if it doesn't exist
c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY,
        context TEXT,
        user_input TEXT,
        bot_response TEXT
    )
""")
conn.commit()

# Define function to generate response
async def generate_response(input_text):
    # Generate response with DialoGPT-large model
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt").to(device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=1024,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    # Convert response from IDs to text
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], clean_up_tokenization_spaces=True)

    return response.strip()

# Define function to send response message
async def send_response_message(channel, response, user_mention):
    await channel.send(f"{user_mention} {response}")

# Define event handlers
client = discord.Client(intents=discord.Intents.default())

@client.event
async def on_ready():
    print(f"Logged in as {client.user}.")

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if message mentions the bot
    if client.user in message.mentions:
        # Generate response and send it back
        prompt = message.content.replace(client.user.mention, "").strip()
        response = await generate_response(prompt)
        response = response.replace("", "").strip()

        # Mention the user and send the response message in a new async function
        user_mention = message.author.mention
        await send_response_message(message.channel, response, user_mention)

        # Store conversation in database
        c.execute("INSERT INTO conversations (context, user_input, bot_response) VALUES (?, ?, ?)", ("", prompt, response))
        conn.commit()
        conversation_id = c.lastrowid

        # Add conversation ID to the response message
        response += f" Conversation ID: {conversation_id}"
        await send_response_message(message.channel, response, client.user.mention)

# Start the Discord bot client
client.run("your bot token here")
