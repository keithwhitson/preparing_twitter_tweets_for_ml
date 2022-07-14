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

for tweet in ['Shot 60 times. Armed with nothing but his Blac...']:
    print('language: ', language_detection(tweet))
    print('cleaned_tweet: ', cleaned_tweet(tweet))
    print('vader_sent: ', vader_sent(tweet))
    print('hashtag_count: ', get_hashtag_count(tweet))
    print('retweets_count:', get_retweets_count(tweet))
    print('mentions_count: ', get_mentions_count(tweet))
    print('links_count: ', get_links_count(tweet))
