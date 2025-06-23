import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report

data = pd.read_csv('./yektanet_train.csv')
data.head()
data.shape
data.drop(['id','domain','url'], axis=1, inplace=True)

"""
for i in range(3,6):
    for column_name in data.columns:
        field = data.loc[i, column_name]
        print(f"{column_name}: {len(field)} : {field}")
    print('|||||||||||||||||||||||||||||||||')
"""
X = data['text_content']
Y = data['category']

#تقسيم البينات الي تدريب واختبار
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= 42)
#تحويل النصوس الي فيتشرات ويعطيب كم كل كلمة تكررت بكل نص 
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
#موديول التصنيف
model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)

Y_pred = model.predict(X_test_tfidf)


print(classification_report(Y_test, Y_pred))

#new text need to classification

new_text = ['']
new_text_tfidf = vectorizer.transform(new_text)
predication = model.predict(new_text_tfidf)

print(predication)