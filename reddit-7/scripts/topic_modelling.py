# %%
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# %%
ROOT = Path().absolute().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"

# %%
df = pd.read_excel(PROC / "df.xlsx")
df.loc[:, "check"] = df.loc[:, "body"].apply(lambda x: isinstance(x, int))
df = df.query("check == False")
df.loc[:, "length"] = df.body.apply(lambda x: len(x.split()))
df = df.query("length > 10")
texts = [item for item in df.body]
# %%
sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
embeddings = sentence_model.encode(texts, show_progress_bar=True)


# %%
representation_model = MaximalMarginalRelevance(diversity=1)
vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
## Uncomment the below to not remove stop words
## vectorizer_model = None
## topic_model = BERTopic(calculate_probabilities=True, representation_model=representation_model, vectorizer_model=vectorizer_model, language = 'multilingual')
topic_model = BERTopic(
    calculate_probabilities=True, ctfidf_model=ctfidf_model, language="multilingual"
)
topics, probs = topic_model.fit_transform(texts, embeddings)

# %%
topic_model.get_topic_info()
