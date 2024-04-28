#!/usr/bin/env python
# coding: utf-8

# In[88]:


import re

import matplotlib as plt
import sklearn
import pandas as pd
import nltk
import warnings
warnings.filterwarnings('ignore')


# In[89]:


df = pd.read_csv('twitter_sentiment_data.csv')
df.head()


# In[90]:


len(df)


# # Remove duplicate tweets

# In[91]:


df.drop_duplicates(subset=['message'], inplace=True)
#df['original_message'] = df['message']


# In[92]:


df.head(10)


# In[93]:


len(df)


# # Preprocessing

# # Lowercase

# In[94]:


befLowCase = df['message'][1]
df['message'] = df['message'].apply(str.lower)
aftLowCase = df['message'][1]

print('Before was:\n',befLowCase,'\nNow it is:\n',aftLowCase)


# # Punctuation Removal

# In[95]:


import string

punctuation = string.punctuation
print(punctuation)


# In[96]:


punctuation = punctuation.replace('@','')
punctuation = punctuation.replace('_','')
punctuation = punctuation.replace('#','')
print(punctuation)


# In[97]:


before = df['message'][1]
df['message'] = df['message'].apply(lambda x: x.translate(str.maketrans('','',punctuation)))
after = df['message'][1]

print('Before was:\n',before,'\nNow it is:\n',after)


# # Remove RT 

# In[98]:


before = df['message'][1]
df['message'] = df['message'].apply(lambda x: re.sub(r'\d+|rt', '', x))
after = df['message'][1]

print('The text before the transformation was:\n',before,'\nNow it is:\n',after)


# # Remove # and @ from the message and create another columns for them

# In[100]:


def extract_hashtags_mentions(text):
    hashtags = re.findall(r"#(\w+)", text)
    mentions = re.findall(r"@(\w+)", text)
    cleaned_text = re.sub(r"#(\w+)|@(\w+)", "", text)
    return hashtags, mentions, cleaned_text


# In[101]:


df['hashtags'], df['mentions'], df['cleaned_message'] = zip(*df['message'].apply(extract_hashtags_mentions))


# In[102]:


df.head()


# # Remove links

# In[103]:


df['cleaned_message'] = df['cleaned_message'].apply(lambda x: re.sub(r'htt\S+|www\S+', '', x))


# In[104]:


df['cleaned_message']


# # stopwords removal

# In[105]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[106]:


stop_words = set(stopwords.words('english'))


# In[107]:


def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# In[108]:


df['cleaned_message'] = df['cleaned_message'].apply(remove_stopwords)


# In[109]:


df['cleaned_message']


# # Removal of emoticon and emojis

# In[110]:


def remove_emoji_emoticon(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
  
    cleaned_text = emoji_pattern.sub(r'', text)
    return cleaned_text


# In[111]:


df['cleaned_message'] = df['cleaned_message'].apply(remove_emoji_emoticon)


# # Spellchecking

# In[112]:


get_ipython().system(' pip install spello')


# In[113]:


# import model to fix spelling after installing spello
from spello.model import SpellCorrectionModel

sp = SpellCorrectionModel(language='en')
sp.load('C:\\Users\\wutyi\\anaconda3\\Lib\\site-packages\\spello\\en.pkl\\en.pkl')


# In[114]:


df['cleaned_message'] = df['cleaned_message'].apply(lambda x: sp.spell_correct(x)['spell_corrected_text'])


# # Handling of contractions

# In[115]:


contractions_dict = {
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'll": "he shall / he will",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}


# In[116]:


def expand_contractions(text):
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text


# In[117]:


df['cleaned_message'] = df['cleaned_message'].apply(expand_contractions)


# In[69]:


# Defining the dictionary of negations to handle
#negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                 "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                 "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                 "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                 "mustn't":"must not"}


#neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

#def remove_contractions(text):
    #return neg_pattern.sub(lambda x: negations_dic[x.group()], text)

#df['message'] = df['message'].apply(lambda x: remove_contractions(x))


# # Tokenization of words
# 
# 
# TweetTokenizer from the package nltk.tokenize 

# In[118]:


from nltk.tokenize import TweetTokenizer
nltk.download('punkt')

tok = TweetTokenizer()
df['cleaned_message'] = df['cleaned_message'].apply(lambda x: tok.tokenize(x))
df.head()


# # Lemmatization of words
# 

# In[119]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')


# In[120]:


nltk.download('wordnet')
nltk.download('omw-1.4')


# In[124]:


lemmatizer = WordNetLemmatizer()

def lemmat(w_list):
    lemm_sentence = []
    for w in w_list:
        pos_tag = nltk.pos_tag([w])[0]
        # Default to noun
        wtag = wordnet.NOUN
        
        if pos_tag[1].startswith('J'):
            wtag = wordnet.ADJ
        elif pos_tag[1].startswith('N'):
            wtag = wordnet.NOUN
        elif pos_tag[1].startswith('R'):
            wtag = wordnet.ADV
        elif pos_tag[1].startswith('V'):
            continue

        lemmetized_word = lemmatizer.lemmatize(w, pos=wtag)
        lemm_sentence.append(lemmetized_word)
    return lemm_sentence


df['cleaned_message'] = df['cleaned_message'].apply(lambda x: ' '.join(lemmat(x)))


print(df['cleaned_message'])


# In[125]:


df['preprocessed_text'] = df['cleaned_message'].apply(lambda x: " ".join(x))


# In[126]:


df.to_csv('preprocessed_twitterdata.csv', index=False)

