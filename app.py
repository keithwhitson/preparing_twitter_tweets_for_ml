import pandas as pd
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import spacy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

nlp = spacy.load('en_core_web_sm')

def create_lang_detector(nlp, name):
    '''Used for lang detection'''
    return LanguageDetector()


Language.factory("language_detector", func=create_lang_detector)
nlp.add_pipe('language_detector', last=True)


class PrepareTweetML:
    def __init__(self, tweet:str):
        self.tweet = tweet
        self.vader_sent_ = self.vader_sent()
        self.language = self.language_detection()
        self.cleaned_tweet = self.cleaned_tweet()
        self.hashtag_count = self.get_hashtag_count()
        self.mentions_count = self.get_mentions_count()
        self.retweets_count = self.get_retweets_count()
        self.links_count = self.get_links_count()
        self.num_words_dirty = self.get_num_words_dirty()
        self.num_words_cleaned = self.get_num_words_cleaned()
        self.num_chars_cleaned = self.get_num_chars_cleaned()
        self.num_chars_dirty = self.get_num_chars_dirty()
        self.ratio_words_used_over_total = self.num_words_cleaned/self.num_words_dirty
        self.cleanliness_ratio = self.num_chars_cleaned/self.num_chars_dirty


    def vader_sent(self):
        '''This creates a score for a tweet for social media sentiment'''
        vader_sent_raw = analyzer.polarity_scores(self.tweet)
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


    def language_detection(self):
        '''Takes a tweet and returns a language'''
        doc = nlp(self.tweet)
        return doc._.language.get('language')


    def cleaned_tweet(self):
        '''Removes whitespace and generally cleans tweet'''
        removed_link_tweet = re.sub(r"http\S+", "", self.tweet)
        removed_mention_tweet = re.sub("@[A-Za-z0-9]+", "", removed_link_tweet)
        removed_non_alpha = re.sub(r'[\W_]+', ' ', removed_mention_tweet)
        removed_rt = removed_non_alpha.replace('RT ', '')
        single_spaced_tweet = " ".join(removed_rt.split())
        lowercased_tweet = single_spaced_tweet.lower()
        return lowercased_tweet


    def get_hashtag_count(self):
        '''Returns the number of hashtags present'''
        hashtags = self.tweet.count('#')
        return hashtags


    def get_mentions_count(self):
        '''Returns the number of mentions in tweet'''
        mentions = self.tweet.count('@')
        return mentions


    def get_retweets_count(self):
        '''Gets the number of retweets'''
        retweets = self.tweet.count('RT ')
        return retweets


    def get_links_count(self):
        '''Get the number of links'''
        links = self.tweet.count('http')
        return links


    def get_num_words_dirty(self):
        '''Returns the num of words before cleaning'''
        word_list = self.tweet.split()
        number_of_words = len(word_list)
        return number_of_words


    def get_num_words_cleaned(self):
        '''Returns the number of words in the cleaned tweet'''
        removed_link_tweet = re.sub(r"http\S+", "", self.tweet)
        removed_mention_tweet = re.sub("@[A-Za-z0-9]+", "", removed_link_tweet)
        removed_non_alpha = re.sub(r'[\W_]+', ' ', removed_mention_tweet)
        removed_rt = removed_non_alpha.replace('RT ', '')
        single_spaced_tweet = " ".join(removed_rt.split())
        lowercased_tweet = single_spaced_tweet.lower()
        word_list = lowercased_tweet.split()
        number_of_words = len(word_list)
        return number_of_words


    def get_num_chars_cleaned(self):
        '''Get chars cleaned'''
        removed_link_tweet = re.sub(r"http\S+", "", self.tweet)
        removed_mention_tweet = re.sub("@[A-Za-z0-9]+", "", removed_link_tweet)
        removed_non_alpha = re.sub(r'[\W_]+', ' ', removed_mention_tweet)
        removed_rt = removed_non_alpha.replace('RT ', '')
        single_spaced_tweet = " ".join(removed_rt.split())
        return len(single_spaced_tweet)


    def get_num_chars_dirty(self):
        '''Get num chars'''
        return len(self.tweet)



# p1 = PrepareTweetML("John")

# print(p1.cleaned_tweet())

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


#print(pd.read_parquet('all_black_twitter_tweets_for_ml.parquet'))

#print(pd.read_parquet('all_black_twitter_tweets'))


p1 = PrepareTweetML('the amount I have left to do this month &gt; t..')
print(p1.__dict__)