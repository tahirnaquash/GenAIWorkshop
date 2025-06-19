from transformers import pipeline

# Load Llama model for sentiment analysis (make sure you have the model downloaded or access to it)
sentiment_pipeline = pipeline(
    "text-classification",
    model="meta-llama/Llama-2-7b-chat-hf",  # Replace with a Llama model fine-tuned for classification if available
    tokenizer="meta-llama/Llama-2-7b-chat-hf"
)

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]

sentences = [
    "I love the new AI advancements, they are truly revolutionary!",
    "The new software update is frustrating and full of bugs.",
    "Customer service was amazing, very helpful and responsive.",
    "I'm disappointed with the product quality, not what I expected.",
    "This experience has been wonderful, highly recommend!"
]

for sentence in sentences:
    sentiment_result = analyze_sentiment(sentence)
    print(f"Sentence: {sentence}\nSentiment: {sentiment_result}\n")