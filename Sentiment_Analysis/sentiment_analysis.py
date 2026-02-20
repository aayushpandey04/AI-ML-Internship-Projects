from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

texts = [
    "I love this product",
    "This is fantastic",
    "Very happy with the service",
    "I hate this",
    "This is terrible",
    "Very disappointing experience"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative"
]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=1
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, predictions))

sample = ["I really like this service"]
sample_vec = vectorizer.transform(sample)
print("Sentiment:", model.predict(sample_vec)[0])
