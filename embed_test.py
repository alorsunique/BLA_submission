import ollama

embed = ollama.embeddings(model='nomic-embed-text:v1.5', prompt='The house is called')

print(type(embed))

print(embed)
