#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries
# 

# In[2]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import re


# ## Reading the url into a pandas object

# In[3]:


df = pd.read_excel('Input.xlsx')


# ## Web Scraping using BeautifulSoup

# In[6]:


for index,row in df.iterrows():
    url = row['URL']
    url_id = row['URL_ID']
    
    # Making a request to url
    header = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    try:
        response = requests.get(url,headers=header)
    except:
        print("Can't get response of {}".format(url_id))
        
    # Creating a BeautifulSoup Object
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
    except:
        print("Can't get page of {}".format(url_id))
        
    # Finding title
    try:
        title = soup.find('h1').get_text()
    except:
        print("Can't get title of {}".format(url_id))
        continue
        
    # Finding text
    article = ""
    try:
        for p in soup.find_all('p'):
            article += p.get_text()
    except:
        print("Can't get text of {}".format(url_id))
        
    # Writing title and text to the file
    file_name = 'C:/Users/KIIT/Downloads/Blackcoffer Assignment/TitleText/' + str(url_id) + '.txt'
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(title + '\n' + article)


# ## Making directories and calculating positive score, negative score, checking for the positive words and the negative words

# In[12]:


text_dir = "C:/Users/KIIT/Downloads/Blackcoffer Assignment/TitleText"
stopwords_dir = "C:/Users/KIIT/Downloads/Blackcoffer Assignment/StopWords-20240331T182805Z-001/StopWords"
sentiment_dir = "C:/Users/KIIT/Downloads/Blackcoffer Assignment/MasterDictionary-20240331T182805Z-001/MasterDictionary" 

# Loading all stopwords from the stopwords directory and store in the set variable
stop_words = set()
for files in os.listdir(stopwords_dir):
    with open(os.path.join(stopwords_dir,files),'r',encoding = 'ISO-8859-1') as fi:
        stop_words.update(set(fi.read().splitlines()))

# Load all text files from the directory and store in a list
docs = []
for text_file in os.listdir(text_dir):
    with open(os.path.join(text_dir,text_file),'r', encoding = 'utf-8') as fi:
        text = fi.read()
        # Tokenizing the text file
        words = word_tokenize(text)
        # Remove the stop words from the tokens
        filtered_text = [word for word in words if word.lower() not in stop_words]
        # Adding each filtered token of each file into a list
        docs.append(filtered_text)
        
        

# Storing words from the dir into positive, negative
pos = set()
neg = set()

for files in os.listdir(sentiment_dir):
    if files == 'positive-words.txt':
        with open (os.path.join(sentiment_dir,files),'r',encoding = 'ISO-8859-1') as fi:
            pos.update(fi.read().splitlines())
    else:
        with open(os.path.join(sentiment_dir,files),'r', encoding = 'ISO-8859-1') as fi:
            neg.update(fi.read().splitlines())
            
            
            
positive_words = []
negative_words = []
positive_score = []
negative_score = []
polarity_score = []
subjectivity_score = []

# Iterate through the list of docs
for i in range(len(docs)):
    positive_words.append([word for word in docs[i] if word.lower() in pos])
    negative_words.append([word for word in docs[i] if word.lower() in neg])
    positive_score.append(len(positive_words[i]))
    negative_score.append(len(negative_words[i]))
    polarity_score.append((positive_score[i] - negative_score[i]) / ((positive_score[i] + negative_score[i]) + 0.000001))
    subjectivity_score.append((positive_score[i] + negative_score[i]) / ((len(docs[i])) + 0.000001))


# ## Calculating Average Sentence Length, Percentage of Complex words, Fog Index, Average Number of Words Per Sentence, Complex word count, Average syllable count

# In[75]:


import os
import re
import nltk
from nltk.corpus import stopwords

# Load the NLTK stopwords corpus
nltk.download('stopwords')

# Define variables to store the calculated values
avg_sentence_length = []
percentage_of_complex_words = []
fog_index = []
avg_words_per_sent = []
complex_word_count = []
avg_syll_count = []

# Rename the set of stopwords to avoid conflicts
stopwords_set = set(stopwords.words('english'))

def measure(file):
    with open(os.path.join(text_dir, file), 'r', encoding = 'utf-8') as f:
        text = f.read()
        # Removing punctuation marks
        text = re.sub(r'[^\w\s]', '', text)
        # Splitting the given text file into sentences
        sentences = text.split('.')
        # Counting the total number of sentences in a file
        num_sentences = len(sentences)
        # Total no. of words in the file
        words = [word for word in text.split() if word.lower() not in stopwords_set]
        num_words = len(words)
        
        # Complex words = words with more than two syllables
        comp_wrd = []
        for word in words:
            vowel = 'aeiou'
            syll_cnt_wrd = sum(1 for letter in word if letter.lower() in vowel)
            if syll_cnt_wrd > 2:
                comp_wrd.append(word)
                
        syll_cnt = 0
        syll_wrds = []
        for word in words:
            if word.endswith('es'):
                word = word[:-2]
            elif word.endswith('ed'):
                word = word[:-2]
            vowels = 'aeiou'
            syll_cnt_wrd = sum(1 for letter in word if letter.lower() in vowels)
            if syll_cnt_wrd >= 1:
                syll_wrds.append(word)
                syll_cnt += syll_cnt_wrd
        
        avg_sentence_len = num_words / num_sentences
        avg_syllable_word_count = syll_cnt / len(syll_wrds)
        percent_complex_words = len(comp_wrd) / num_words
        fog_index_val = 0.4 * (avg_sentence_len + percent_complex_words)
        avg_num_wrds_sent = len(words)/len(sentences)
        
        return avg_sentence_len, percent_complex_words, fog_index_val, avg_num_wrds_sent, len(comp_wrd), avg_syllable_word_count

for file in os.listdir(text_dir):
    x, y, z, a, b, c = measure(file)
    avg_sentence_length.append(x)
    percentage_of_complex_words.append(y)
    fog_index.append(z)
    avg_words_per_sent.append(a)
    complex_word_count.append(b)
    avg_syll_count.append(c)


# ## Calculating word count, average word length, personal pronoun count

# In[26]:


def cleaned_words(file):
    with open(os.path.join(text_dir,file),'r',encoding = 'utf-8') as f:
        text = f.read()
        text = re.sub(r'[^\w\s]','',text)
        words = [word for word in text.split() if word.lower() not in stopwords]
        length = sum(len(word) for word in words)
        average_word_length = length / len(words)
    return len(words),average_word_length

word_count = []
average_word_length = []
for file in os.listdir(text_dir):
    x,y = cleaned_words(file)
    word_count.append(x)
    average_word_length.append(y)
    
    
def count_personal_pronouns(file):
    with open(os.path.join(text_dir,file),'r',encoding = 'utf-8') as f:
        text = f.read()
        personal_pronouns = ["I","we","my","ours","us"]
        count = 0
        for pronoun in personal_pronouns:
            count += len(re.findall(r"\b" + pronoun + r"\b", text))
    return count

pp_count = []
for file in os.listdir(text_dir):
    x = count_personal_pronouns(file)
    pp_count.append(x)


# ## Exporting the results into csv format

# In[73]:


import pandas as pd

output_df = pd.read_excel('Output Data Structure.xlsx')

output_df.drop([37,50], axis = 0 , inplace = True)

variables = [positive_score,
            negative_score,
            polarity_score,
            subjectivity_score,
            avg_sentence_length,
            percentage_of_complex_words,
            fog_index,
            avg_words_per_sent,
            complex_word_count,
            word_count,
            avg_syll_count,
            pp_count,
            average_word_length]

for i, var in enumerate(variables):
    output_df.iloc[:, i + 2] = var
    
output_df.to_csv('Output_Data.csv')

