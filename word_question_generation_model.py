import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer


# 데이터 준비
current_directory = os.getcwd()
relative_path = os.path.join("dataset", "words_question.csv")

# 데이터 불러오기
csv = pd.read_csv(os.path.join(current_directory, relative_path))

# 데이터 처리
words = csv["QUESTION"]
levels = csv["CURRICULUM_STEP_NO"]
meaning = csv["MEANING"]
answer = csv["ANSWER"]
distractor_1 = csv["DISTRACTOR_1"]
distractor_2 = csv["DISTRACTOR_2"]
distractor_3 = csv["DISTRACTOR_3"]
distractor_4 = csv["DISTRACTOR_4"]