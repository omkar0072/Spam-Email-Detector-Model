print("SPAM EMAIL DETECTOR - TRAINING ")


# required packages
!pip install pandas scikit-learn -q
print("✅ Packages installed")

# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# training data (spam/normal)
print("\n📊 Loading training data...")

# Spam emails (label = 1)
spam_emails = [
    "WIN FREE IPHONE CLICK NOW",
    "YOU WON $1000000 CASH PRIZE",
    "URGENT CLAIM YOUR PRIZE",
    "FREE MONEY TRANSFER NOW",
    "GET RICH QUICK EASY",
    "LOTTERY WINNER CONGRATULATIONS",
    "BUY NOW LIMITED OFFER",
    "SPECIAL DISCOUNT JUST FOR YOU",
    "EARN MONEY FROM HOME",
    "EXCLUSIVE DEAL FREE GIFT"
]

# Normal emails (label = 0)
normal_emails = [
    "MEETING AT OFFICE TOMORROW",
    "PROJECT REPORT SUBMISSION",
    "TEAM LUNCH ON FRIDAY",
    "YOUR INVOICE IS READY",
    "WEEKLY STATUS UPDATE",
    "REMINDER FOR APPOINTMENT",
    "FEEDBACK ON PRESENTATION",
    "SCHEDULE A MEETING",
    "RESUME ATTACHED FOR REVIEW",
    "THANKS FOR YOUR HELP"
]

# Combine all emails and labels
all_emails = spam_emails + normal_emails
labels = [1]*10 + [0]*10  # 1=spam, 0=normal

print(f"✅ Total emails: {len(all_emails)}")
print(f"   - Spam emails: {len(spam_emails)}")
print(f"   - Normal emails: {len(normal_emails)}")

#Split data for training and testing
print("\n🎯 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(all_emails, labels, test_size=0.2, random_state=42)

print(f"✅ Training emails: {len(X_train)}")
print(f"✅ Testing emails: {len(X_test)}")

# binary
print("\n🔢 Converting text to numbers...")
vectorizer = CountVectorizer()
X_train_numbers = vectorizer.fit_transform(X_train)
X_test_numbers = vectorizer.transform(X_test)

print(f"✅ Found {len(vectorizer.vocabulary_)} unique words")
print("   Some words learned:", list(vectorizer.vocabulary_.keys())[:5])

# Train the model
print("\n🤖 Training the AI model...")
model = MultinomialNB()
model.fit(X_train_numbers, y_train)
print("✅ Model trained successfully!")

# Test the model
print("\n🧪 Testing the model...")
predictions = model.predict(X_test_numbers)
accuracy = accuracy_score(y_test, predictions)

print(f" Accuracy: {accuracy*100:.1f}%")
print("\nTest Results:")
print("-" * 30)
for i in range(len(X_test)):
    actual = "SPAM" if y_test[i] == 1 else "NORMAL"
    predicted = "SPAM" if predictions[i] == 1 else "NORMAL"
    print(f"Email: {X_test[i][:20]}...")
    print(f"  Actual: {actual} | Predicted: {predicted}")
    print()

#  Try it emails!
print("🚀 TEST WITH YOUR OWN EMAILS!")

# Some example emails to test
test_examples = [
    "WIN FREE PRIZE MONEY NOW",  # Should be SPAM
    "MEETING SCHEDULE FOR TOMORROW",  # Should be normal
    "GET RICH QUICK EASY MONEY",  # Should be SPAM
    "PROJECT UPDATE FROM TEAM",  # Should be Normal
]

for email in test_examples:
    # Convert email to numbers
    email_numbers = vectorizer.transform([email])
    
    # Make prediction
    prediction = model.predict(email_numbers)[0]
    confidence = model.predict_proba(email_numbers).max() * 100
    
    result = "SPAM 🚨" if prediction == 1 else "NORMAL ✅"
    
    print(f"\n📩 Email: '{email}'")
    print(f"   Result: {result}")
    print(f"   Confidence: {confidence:.1f}%")

# 9. See what words make an email spam
print("🔍 SPAM WORDS LEARNED BY THE MODEL")

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()
# Get importance of each word for spam classification
spam_importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]

# Show top spam words
word_scores = list(zip(feature_names, spam_importance))
word_scores.sort(key=lambda x: x[1], reverse=True)

print("Top 5 spam words (higher score = more spammy):")
print("-" * 40)
for word, score in word_scores[:5]:
    print(f"  '{word}' → score: {score:.2f}")

print("🎉 TRAINING COMPLETE!")