import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

messages = [
    "Win money now",
    "Limited time offer",
    "Call me tomorrow",
    "Let's have lunch",
    "Congratulations you won a prize",
    "Are you free today?",
    "Claim your reward now",
    "See you soon"
]

labels = [
    "spam",
    "spam",
    "ham",
    "ham",
    "spam",
    "ham",
    "spam",
    "ham"
]

df = pd.DataFrame({"text": messages, "label": labels})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=1
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, predictions))

sample = ["You have won a free gift"]
sample_vec = vectorizer.transform(sample)
print("Sample Prediction:", model.predict(sample_vec)[0])
