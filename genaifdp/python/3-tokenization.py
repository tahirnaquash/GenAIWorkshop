import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib_venn import venn2   
# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample sentences
sentence1 = "She went to the bank to deposit her check."
sentence2 = "He sat by the bank of the river."

# Tokenize and clean
stop_words = set(stopwords.words('english'))
words1 = set(word_tokenize(sentence1.lower())) - stop_words
words2 = set(word_tokenize(sentence2.lower())) - stop_words

# Remove punctuation
words1 = {word for word in words1 if word.isalpha()}
words2 = {word for word in words2 if word.isalpha()}

# Difference
unique_to_1 = words1 - words2
unique_to_2 = words2 - words1

print("Words unique to Sentence 1:", unique_to_1)
print("Words unique to Sentence 2:", unique_to_2)

print("Words common to both sentences:", words1 & words2)
print("Words in Sentence 1:", words1)       
print("Words in Sentence 2:", words2)
print("Total unique words across both sentences:", words1 | words2)
print("Total words in Sentence 1:", len(words1))
print("Total words in Sentence 2:", len(words2))    
print("Total unique words across both sentences:", len(words1 | words2))
print("Total common words in both sentences:", len(words1 & words2))
print("Total unique words in Sentence 1:", len(unique_to_1))
print("Total unique words in Sentence 2:", len(unique_to_2))
print("Total words in both sentences:", len(words1) + len(words2))

import spacy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import en_core_web_sm
nlp = en_core_web_sm.load()
# Load spaCy's English model
#nlp = spacy.load("en_core_web_sm")  # Use 'en_core_web_sm' if 'md' isn't available


# Process and extract unique words
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)
words = list(set([token.text.lower() for token in doc1 if token.is_alpha] +
                 [token.text.lower() for token in doc2 if token.is_alpha]))

# Get word vectors
vectors = [nlp(word).vector for word in words]

# Reduce dimensions to 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word, fontsize=12)
plt.title("2D Visualization of Word Embeddings")
plt.grid(True)
plt.show()

# convert dimensions to 3D
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import PCA
import numpy as np  
# Reduce dimensions to 3D
pca = PCA(n_components=3)


reduced = pca.fit_transform(vectors)
# Plot in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, word in enumerate(words):
    x, y, z = reduced[i]
    ax.scatter(x, y, z)
    ax.text(x + 0.01, y + 0.01, z + 0.01, word, fontsize=12)
ax.set_title("3D Visualization of Word Embeddings")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()  
# Venn diagram of unique words
plt.figure(figsize=(8, 6))      
venn2([words1, words2], ('Sentence 1', 'Sentence 2'))
plt.title("Venn Diagram of Unique Words")
plt.show()
