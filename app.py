import pandas as pd
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import spacy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

nlp = spacy.load('en_core_web_sm')


def create_lang_detector(nlp, name):
    return LanguageDetector()


Language.factory("language_detector", func=create_lang_detector)
nlp.add_pipe('language_detector', last=True)


def vader_sent(tweet):
    '''This creates a score for a tweet for social media sentiment'''
    vader_sent_raw = analyzer.polarity_scores(tweet)
    vader_sent = vader_sent_raw.get('compound', 0.0)
    if vader_sent >= 0.05:
        vader_sent_score = 'Positive'
    elif vader_sent < -0.05:
        vader_sent_score = 'Negative'
    elif (vader_sent > -0.05 and vader_sent < 0.05):
        vader_sent_score = 'Neutral'
    else:
        vader_sent_score = 'Neutral'
    return vader_sent_score


def language_detection(tweet):
    '''Takes a tweet and returns a language'''
    doc = nlp(tweet)
    return doc._.language.get('language')


def cleaned_tweet(tweet):
    '''Removes whitespace and generally cleans tweet'''
    removed_link_tweet = re.sub(r"http\S+", "", tweet)
    removed_mention_tweet = re.sub("@[A-Za-z0-9]+", "", removed_link_tweet)
    removed_non_alpha = re.sub(r'[\W_]+', ' ', removed_mention_tweet)
    removed_rt = removed_non_alpha.replace('RT ', '')
    single_spaced_tweet = " ".join(removed_rt.split())
    lowercased_tweet = single_spaced_tweet.lower()
    return lowercased_tweet


def get_hashtag_count(tweet):
    '''Returns the number of hashtags present'''
    hashtags = tweet.count('#')
    return hashtags


def get_mentions_count(tweet):
    '''Returns the number of mentions in tweet'''
    mentions = tweet.count('@')
    return mentions


def get_retweets_count(tweet):
    '''Gets the number of retweets'''
    retweets = tweet.count('RT ')
    return retweets


def get_links_count(tweet):
    '''Get the number of links'''
    links = tweet.count('http')
    return links


def get_num_words_dirty(tweet):
    '''Returns the num of words before cleaning'''
    word_list = tweet.split()
    number_of_words = len(word_list)
    return number_of_words


def get_num_words_cleaned(tweet):
    '''Returns the number of words in the cleaned tweet'''
    removed_link_tweet = re.sub(r"http\S+", "", tweet)
    removed_mention_tweet = re.sub("@[A-Za-z0-9]+", "", removed_link_tweet)
    removed_non_alpha = re.sub(r'[\W_]+', ' ', removed_mention_tweet)
    removed_rt = removed_non_alpha.replace('RT ', '')
    single_spaced_tweet = " ".join(removed_rt.split())
    lowercased_tweet = single_spaced_tweet.lower()
    word_list = lowercased_tweet.split()
    number_of_words = len(word_list)
    return number_of_words


def get_num_chars_cleaned(tweet):
    '''Get chars cleaned'''
    removed_link_tweet = re.sub(r"http\S+", "", tweet)
    removed_mention_tweet = re.sub("@[A-Za-z0-9]+", "", removed_link_tweet)
    removed_non_alpha = re.sub(r'[\W_]+', ' ', removed_mention_tweet)
    removed_rt = removed_non_alpha.replace('RT ', '')
    single_spaced_tweet = " ".join(removed_rt.split())
    return len(single_spaced_tweet)


def get_num_chars_dirty(tweet):
    '''Get num chars'''
    return len(tweet)

# df = (pd.read_parquet('all_black_twitter_tweets'))

# df['vader_sent'] = df['tweet'].apply(vader_sent)
# df['language_detection'] = df['tweet'].apply(language_detection)
# df['cleaned_tweet'] = df['tweet'].apply(cleaned_tweet)
# df['hashtag_count'] = df['tweet'].apply(get_hashtag_count)
# df['mentions_count'] = df['tweet'].apply(get_mentions_count)
# df['retweets_count'] = df['tweet'].apply(get_retweets_count)
# df['links_count'] = df['tweet'].apply(get_links_count)
# df['num_words_cleaned'] = df['tweet'].apply(get_num_words_cleaned)
# df['num_words_dirty'] = df['tweet'].apply(get_num_words_dirty)
# df['ratio_words_used_over_total'] = df['num_words_cleaned']/df['num_words_dirty']
# df['num_chars_cleaned'] = df['tweet'].apply(get_num_chars_cleaned)
# df['num_chars_dirty'] = df['tweet'].apply(get_num_chars_dirty)
# df['cleanliness_ratio'] = df['num_chars_cleaned']/df['num_chars_dirty']

# df.to_parquet('all_black_twitter_tweets_for_ml.parquet', engine='pyarrow',partition_cols=['twitterhandle'])


print(pd.read_parquet('all_black_twitter_tweets_for_ml.parquet'))
