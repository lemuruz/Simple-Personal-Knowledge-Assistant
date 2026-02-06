import ollama

client = ollama.Client()

message = [
    {
        'role':'system',
        'content':'You are a professional chef.',
    },
    {
        'role': 'user',
        'content': 'What should I cook today?',
    },
]

for responsePart in client.chat('llama3',messages=message,
    stream=True):
    print(responsePart.message.content,end='',flush=True)