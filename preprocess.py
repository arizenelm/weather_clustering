# %%
import pandas as pd
import datetime
import pymystem3
import pymorphy3
import textdistance
import numpy as np
import re
from collections import Counter
from nltk.stem import SnowballStemmer
from matplotlib import pyplot as plt
from wordcloud import WordCloud

# %% [markdown]
# ## Извлечение дат

# %%
df = pd.DataFrame(columns=['date', 'weather'])

# %%
months = {
    "Января" : "January",
    "Февраля" : "February",
    "Марта" : "March",
    "Апреля" : "April",
    "Мая" : "May",
    "Июня" : "June",
    "Июля" : "July",
    "Августа" : "August",
    "Сентября" : "September",
    "Октября" : "October",
    "Ноября" : "November",
    "Декабря" : "December"
}

def parse_time(s : str) -> list[str]:
    for ru_month in months.keys():
        s = s.replace(ru_month, months[ru_month])
    l = list(s)
    i1 = l.index('[')
    i2 = l.index(' ', i1)
    l[i2] = ':'
    l.remove('[')
    l.remove(']')
    return "".join(l)

print(parse_time('10 Января [11 45]      '))

# %%
def extract_date_weather(line : str) -> tuple[datetime.datetime, str]:
    l = line.split('"')
    date_string = parse_time(l[1])
    date = datetime.datetime.strptime(date_string, "%d %B %Y %H:%M")
    return (date, l[3])

# %%
weather = {"date" : [], "weather" : []}

# %%
with open("data/dataMeteo.txt") as data_file, open("data/dataMeteo_new.txt") as data_file_new:
    for line1, line2 in zip(data_file, data_file_new):
        for l in (line1, line2):
            t = extract_date_weather(l)
            weather['date'].append(t[0])
            weather['weather'].append(t[1])


# %%
df = pd.DataFrame(weather)
df.head()

# %% [markdown]
# ## Извлечение названий регионов

# %%
mystem = pymystem3.Mystem()
morphy = pymorphy3.MorphAnalyzer()

# %%
def lemmatizeSentences(texts):
    lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    txtpart = lol(texts, 1000)  # Куски по 1000 предложений
    res = []
    # txtp = 1000 предложений
    for txtp in txtpart:
        # Объединяем 1000 предложений
        alltexts = ' '.join([txt + ' br ' for txt in txtp])

        words = mystem.lemmatize(alltexts)
        doc = []
        for txt in words:
            if txt != '\n' and txt.strip() != '':
                if txt == 'br':
                    res.append(" ".join(doc))
                    doc = []
                else:
                    doc.append(txt)
    return res

# %%
df['weather_l'] = lemmatizeSentences(df['weather'])

# %%
df.head()

# %%
df_towns = pd.read_csv('data/towns.csv')

# %%
df_towns.head()

# %%
df_regions = pd.read_csv('data/regions.csv')
df_regions.head()

# %%
class MyStemmer :
    snowball_stemmer = SnowballStemmer("russian")
    def stem(self, s : str) -> str :
        s = self.snowball_stemmer.stem(s)
        if s[-2:] == "ск" :
            s = s[:-2]
        return s

# %%
my_stemmer = MyStemmer()
print(my_stemmer.stem("чувашский"), my_stemmer.stem("чувашия"))

# %%
stop_words = ["автономная", "автономный", "область", "округ", "край", "республика"] 

# %%
def region_transform(s : str, stemmer) :
    s = s.lower()
    for w in stop_words :
        s = s.replace(w, "")
    return stemmer.stem(s.split()[0])

# %%
snowball_stemmer = SnowballStemmer("russian")

# %%
df_regions['region_stemmed'] = df_regions['region'].map(lambda x : region_transform(x, snowball_stemmer))
df_regions['region_stemmed_full'] = df_regions['region'].map(lambda x : region_transform(x, my_stemmer))
df_regions.head()

# %%
regions_dict = df_regions['region_stemmed'].to_dict()
regions_dict_full = df_regions['region_stemmed_full'].to_dict()
regions_dict = {v : k for k, v in regions_dict.items()}
regions_dict_full = {v : k for k, v in regions_dict_full.items()}

# %%
stem = my_stemmer.stem

print(stem("чувашская"), ":", stem("чувашия"), textdistance.levenshtein(stem("чувашская"), stem("чувашия")), "\n",
    stem("чувашия"), ":", stem("чукотский"), textdistance.levenshtein(stem("чувашия"), stem("чукотский")), "\n",
    stem("чувашский"), ":", stem("чукотcкий"), textdistance.levenshtein(stem("чувашский"), stem("чукотский")), "\n",
    stem("чувашский"), ":", stem("чукотка"),textdistance.levenshtein(stem("чувашский"), stem("чукотка")), "\n",
    stem("карачаево-черкесия"), ":", stem("карачаево-черкесская"), textdistance.levenshtein(stem("карачаево-черкесия"), stem("карачаево-черкесская")))

# print(textdistance.levenshtein(stem("чувашская"), stem("чувашия")), 
#       textdistance.levenshtein(stem("чувашия"), stem("чукотский")), 
#       textdistance.levenshtein(stem("чувашский"), stem("чукотский")),
#       textdistance.levenshtein(stem("чувашский"), stem("чукотка")),
#       textdistance.levenshtein(stem("карачево-черкесия"), stem("карачаево-черкесская")))

# %%
towns_dict = df_towns[['city', 'region_name']]
towns_dict['city'] = towns_dict['city'].transform(lambda x : region_transform(x, snowball_stemmer))
towns_dict['region_name'] = towns_dict['region_name'].map(lambda x : region_transform(x, snowball_stemmer))
towns_dict.set_index('city', inplace=True)
towns_dict = towns_dict.to_dict()['region_name']

# %%
def region_match(s : str) -> list :
    matched = set()
    for w in s.split() :
        stem = snowball_stemmer.stem(w)
        stem_full = my_stemmer.stem(w)
        if stem in regions_dict :
            matched.add(regions_dict[stem])
        elif stem_full in regions_dict_full :
            matched.add(regions_dict_full[stem_full])
        elif stem in towns_dict :
            matched.add(regions_dict[towns_dict[stem]])
    return matched
        

# %%
forecast1 = df['weather_l'][771]
forecast2 = df['weather_l'][773]
forecast3 = "".join(mystem.lemmatize("Завтра в Новосибирске солнечно, а в Екатеринбурге идут дожди"))
forecast3 = lemmatizeSentences(["Завтра в Новосибирске солнечно, а в Екатеринбурге идут дожди"])[0]
print(forecast1)
matched = region_match(forecast1)
print(matched, [df_regions['region'][i] for i in matched])
print(forecast2)
matched = region_match(forecast2)
print(matched, [df_regions['region'][i] for i in matched])
print(forecast3)
matched = region_match(forecast3)
print(matched, [df_regions['region'][i] for i in matched])

# %%
df['region_ids'] = [np.nan] * len(df)
df['region_names'] = [np.nan] * len(df)
df['region_ids'] = df['region_ids'].astype('object')
df['region_names'] = df['region_names'].astype('object')
for i in range(len(df)) :
    matched = region_match(df['weather_l'][i])
    df['region_ids'][i] = list(matched)
    df['region_names'][i] = [df_regions['region'][j] for j in matched]

# %%
df.head()

# %% [markdown]
# ## Извлечение погодных явлений

# %%
print(morphy.parse("гора")[0].tag)
print(morphy.parse('и')[0].tag)
print(morphy.parse('покров')[0].tag)

# %%
nonPunct = re.compile(r'\b[а-яА-Я]+\b')

# %%
all_words = dict()
filtered = []
for i in range(len(df)) :
    for w in df['weather_l'][i].split() :
        tag = morphy.parse(w)[0].tag
        stemmed = snowball_stemmer.stem(w)
        if nonPunct.match(w) and 'PREP' not in tag and 'CONJ' not in tag and stemmed not in regions_dict and stemmed not in towns_dict:
            filtered.append(w)
 

# %%
word_counts = pd.Series(Counter(filtered))
word_counts.sort_values(inplace=True, ascending=False)

# %%
word_counts.head()

# %%
word_list = [w for w in word_counts.index]
wordcloud = WordCloud(width = 2000, height = 2000,
                background_color ='white',
                min_font_size = 10).generate(" ".join(word_list))

# %%
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)

# %%
word_counts.to_csv('data/word_counts.csv')

# %%
weather_ids = pd.read_csv('data/weather_ids.csv', header = None, index_col = 0, squeeze = True)
weather2id = pd.read_csv('data/weather2id.csv', header = None, index_col = 0, squeeze = True)

# %%
df['conditions_ids'] = [np.nan] * len(df)
df['conditions'] = [np.nan] * len(df)
df['condition_ids'] = df['conditions_ids'].astype('object')
df['conditions'] = df['conditions_ids'].astype('object')

# %%
for i in range(len(df)) :
    condition_ids = set()
    conditions = set()
    for w in df['weather_l'][i].split() :
        if w in weather2id.index :
            condition_ids.add(weather2id[w])
            conditions.add(weather_ids[weather2id[w]])
    df['condition_ids'][i] = list(condition_ids)
    df['conditions'][i] = list(conditions)

# %%
df[['weather', 'conditions']].to_csv('data/validate.csv')

# %%
df.to_csv('data/weather.csv')


