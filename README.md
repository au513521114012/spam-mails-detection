{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import datetime\n",
    "import nltk\n",
    "import seaborn as sns\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#read file\n",
    "data = pd.read_csv(\"spam.csv\", encoding='latin1')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#clean and update df\n",
    "data=data.drop([\"Unnamed: 2\",\"Unnamed: 3\",\"Unnamed: 4\"],axis=1)\n",
    "data=data.rename(columns={\"v1\":\"label\",\"v2\":\"text\"})\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#check total rows\n",
    "data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#count unique values\n",
    "data.label.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#summary\n",
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create new column and measure length\n",
    "data['length']=data['text'].apply(len)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#histogram for length\n",
    "data['length'].plot(bins=25, kind='hist', figsize=(5,4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#histogram group by label\n",
    "data.hist(column='length', by='label', bins=25, figsize=(8,4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# map \"ham\" to 0, \"spam\" to 1\n",
    "data.loc[:,'label']=data.label.map({'ham':0, 'spam':1})\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Uses CountVectorizer on 'data' text, splits, transforms new text for classification\n",
    "count=CountVectorizer()\n",
    "input=[\"Thank you for your recent purchase. Here's a special discount code for your next order.\"]\n",
    "text=count.fit_transform(data['text'],input)\n",
    "\n",
    "x_train, x_test, y_train, y_test= train_test_split(text,data['label'], test_size=0.20, random_state=1)\n",
    "text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Shapes training set, testing set\n",
    "print(x_train.shape)\n",
    "print(x_test.shape)\n",
    "input=text[5571]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#logistic regression model\n",
    "model=LogisticRegression()\n",
    "model.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Predicts labels on test set\n",
    "prediction=model.predict(x_test)\n",
    "print(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#accuracy, precision, recall, and F1 scores for model evaluation\n",
    "print(\"Accuracy score: {}\".format(accuracy_score(y_test, prediction)))\n",
    "print(\"precision score: {}\".format(precision_score(y_test, prediction)))\n",
    "print(\"recall score: {}\".format(recall_score(y_test, prediction)))\n",
    "print(\"f1 score: {}\".format(f1_score(y_test, prediction)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Uses trained model to predict label for the input text\n",
    "input\n",
    "result = data['text']\n",
    "input =\"Great deals on electronics this weekend. Visit our store for exclusive discounts.\"\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# append new text to df\n",
    "result.loc[len(result.index)]=\"Congratulations! You've been selected for a special promotion. Claim your prize now.\"\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
 /  "metadata": {},
   "outputs": [],
   "source": [
    "#transform\n",
    "text=count.fit_transform(result)\n",
    "text\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict label using trained model\n",
    "input=text[5572]\n",
    "model.predict(input)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict label using trained model (not spam)\n",
    "input_text = [\"Reminder: Your appointment with Dr. Smith is scheduled for tomorrow at 3 PM.\"]\n",
    "input_transformed = count.transform(input_text)\n",
    "prediction = model.predict(input_transformed)\n",
    "print(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict label using trained model  (spam)\n",
    "input_text = [\"Congratulations! You've won a free cruise. Call now to claim your prize.\"]\n",
    "input_transformed = count.transform(input_text)\n",
    "prediction = model.predict(input_transformed)\n",
    "print(prediction)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
