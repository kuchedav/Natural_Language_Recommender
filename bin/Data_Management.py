import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from stop_words import get_stop_words
import sys
import subprocess
import re
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.neural_network import MLPClassifier

from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# read in all pdf texts
from tika import parser
from cleantext.clean import clean
from langdetect import detect


### general functions ###

def create_hit_data_frame(data, all_topics = None):
        from itertools import product

        # create data frame with all possible combinations of ID-Thema
        combination_list = list(product(data["ID"].unique(),data["Thema"].unique()))
        combination_df = pd.DataFrame(combination_list,columns=["ID","Thema"])

        # join pca information to combination_df table
        df_ID = data.copy()
        df_ID.pop("Thema")
        hit_df = combination_df.merge(df_ID.drop_duplicates("ID"), how="left", on="ID")

        # create hit df. We use this df to see which combinations of ID-Thema actually occured and which ones are wrong
        hit_Thema = data[["Thema","ID"]]
        hit_Thema["Hit"] = 1
        hit_Thema = hit_Thema.drop_duplicates(["Thema","ID"])

        # join hit df to
        hit_df_full = hit_df.merge(hit_Thema, how="left", on=["ID","Thema"])
        hit_df_full.fillna(0, inplace=True)
        return hit_df_full.reset_index(drop=True)

def reduce_dimensions(df, n_dim, ignore_columns = []):
  """
  This function reduces dimensions of a data frame while excluding the Thema attribute
  """
  from sklearn.decomposition import PCA
  pca_model = PCA(n_dim)
  try:
    # remove columns to ignore
    pip_df = dict()
    for i in ignore_columns:
      try:
        pip_df[i] = df[i]
        df = df.drop(columns=i)
      except:
        continue
    
    reduced_embedding = pca_model.fit_transform(df)
    embedding_df = pd.DataFrame(reduced_embedding)

    # add ignored columns again
    for i,j in pip_df.items(): embedding_df[i] = j

    return embedding_df

  except Exception as e:
      print("ERROR: " + e)
      return

def create_classifier_df(embedding,identifier_str):
    
    # remove all topics smaller than 10 counts
    topics_larger_10 = [j for j,i in dict(Counter(list(embedding["Thema"]))).items() if i > 9]

    embeddings_larger_10 = embedding[embedding["Thema"].isin(topics_larger_10)]

    embeddings_reduced = fun_save(fun_input = reduce_dimensions,
                                    attr_input = {"df":embeddings_larger_10, "n_dim":500, "ignore_columns":["Thema","index","ID"]},
                                    identifier = identifier_str, only_save_one = True)

    data_hit = fun_save(fun_input = create_hit_data_frame, attr_input = {"data":embeddings_reduced},
                        identifier = identifier_str, only_save_one = True)

    classifier_df = pd.DataFrame()
    for i in data_hit["Thema"].unique():
        # make sure only string topics are being processed
        if not isinstance(i,str): continue

        classifier_dict = dict()
        classifier_dict.update({"Thema":i})
        data_thema = data_hit[data_hit["Thema"] == i]

        # extract X and y
        y_train = data_thema["Hit"]
        X = data_thema
        X_train = X.drop(columns=["ID","Hit","Thema"])

        # train and append classifier
        clf = MLPClassifier(max_iter=1000).fit(X_train, y_train)
        classifier_dict.update({"classifier":clf})

        # Collect probabilities, used to generate top 5 score later on
        X_proba = clf.predict_proba(X_train)
        X_proba_1 = [i[1] for i in X_proba]
        ID_to_prob = dict(zip(X["ID"],X_proba_1))
        classifier_dict.update(ID_to_prob)
        classifier_dict_df = pd.DataFrame([classifier_dict])

        classifier_df = classifier_df.append(classifier_dict_df,ignore_index=True)
    
    return classifier_df

def predict_df_fun(embedding,embeddings_chosen,classifier_df):
    
    embeddings_chosen_dropped = embeddings_chosen.drop(columns=["Thema","ID"]).append(embedding)
    embeddings_reduced = reduce_dimensions(df=embeddings_chosen_dropped, n_dim=500)
    reduced_embedding = embeddings_reduced.tail(embedding.shape[0])

    predict_df = pd.DataFrame(columns=["Thema","Probability"])
    for i in classifier_df["Thema"].unique():

        # make sure only string topics are being processed
        if not isinstance(i,str): continue

        # Collect probabilities, used to generate top 5 score later on
        X_proba = classifier_df["classifier"][classifier_df["Thema"] == i].values[0].predict_proba(reduced_embedding)
        X_proba_1 = [i[1] for i in X_proba]

        predict_df = predict_df.append({"Thema":i,"Probability":X_proba_1[0]},ignore_index=True)
    
    return predict_df

def predict_top_n(embedding,embeddings_chosen,classifier_df, top_n = 5, cut_under_n = 0):
    # create prediction
    predict_df = predict_df_fun(embedding,embeddings_chosen,classifier_df)

    # filter out top n which have more than 50% probability
    top_n_predictions = predict_df.sort_values("Probability").tail(top_n)
    top_n_predictions_05 = top_n_predictions[top_n_predictions["Probability"]>cut_under_n]
    return top_n_predictions_05

def get_scores(classifier_df, embeddings_chosen, top_n = 5, cut_under_n = 0, plot_boolean = False):
    
    classifier_df_prob = classifier_df.drop(columns=["Thema","classifier"])

    test_IDs = classifier_df_prob.columns
    scores = list()
    for i in test_IDs:
        if i == 0: continue

        top_5 = classifier_df[[i,"Thema"]].sort_values(i)[-top_n:]
        top_5 = top_5[top_5.iloc[:,0] > cut_under_n]
        top_5_topics = list(top_5["Thema"])
        true_topics = list(embeddings_chosen[embeddings_chosen["ID"] == int(i)]["Thema"])

        correct_pred_topics = [i for i in true_topics if i in top_5_topics]

        if top_n == 1:
          score = len(set(correct_pred_topics))
        else:
          score = np.round(len(set(correct_pred_topics)) / len(set(true_topics)),2)
        scores.append(score)
    
    if plot_boolean:
        fig = sns.countplot(scores)
        fig.set(xlabel='Score', ylabel='Count of test documents')
        plt.title("Prediction Scores")
        plt.show()

    return scores

### Read, clean and convert pdf files ###

def extract_pdf(path):
    try:
        raw = parser.from_file(path)
        pdf_text = raw['content']
    except Exception as e:
        print("Could not read PDF file")
        print(e)
        return

    if pdf_text == None:
        print("There is no content in the PDF file")
        return
    else:
        # clean the text
        pdf_text_clean = clean(pdf_text,
            fix_unicode=True,               # fix various unicode errors
            to_ascii=True,                  # transliterate to closest ASCII representation
            lower=False,                    # lowercase text
            no_line_breaks=True,            # fully strip line breaks as opposed to only normalizing them
            no_urls=True,                   # replace all URLs with a special token
            no_emails=True,                 # replace all email addresses with a special token
            no_phone_numbers=True,          # replace all phone numbers with a special token
            no_numbers=True,                # replace all numbers with a special token
            no_digits=True,                 # replace all digits with a special token
            no_currency_symbols=True,       # replace all currency symbols with a special token
            no_punct=False,                 # fully remove punctuation
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            lang="en"                       # set to 'de' for German special handling
        )

        # check language
        if detect(pdf_text_clean) != "en":
          return None
          print("This pdf was detected not to be in english.")
          print("Please make sure you only use english texts.")
          input_bool = input("Do you want to continue anyways?")
          if not "y" in input_bool.lower():
              print("Process has been stopped")
              return
        
        return pdf_text_clean

def create_text_data_file(path_to_pdfs):
  pdf_names = os.listdir(path_to_pdfs)
  # keep only pdf files
  pdf_names = [i for i in pdf_names if i[-4:] == ".pdf"]

  print("Number of pdf-names which contain FIT_ID: " + str(np.sum(["FIT_ID" in i for i in pdf_names])))
  print("Number of pdf-names which contain 'et al': " + str(np.sum(["et al" in i for i in pdf_names])))

  df_pdf = fun_save(read_and_clean_texts, {"paths_to_pdfs":path_to_pdfs,"pdf_names":pdf_names}, only_save_one=True)
  # df_pdf["Name"] = df_pdf.Name.apply(lambda x:x.replace(".pdf",""))

  # Fit files with Author name
  def proper_format(text):
    return text.replace("et al._","et al., ")[:-4]

  df_pdf["Bewertung Qualität Originalautor"] = df_pdf["Name"].apply(proper_format)
  df_pdf_string = df_pdf[["et al" in i for i in df_pdf["Bewertung Qualität Originalautor"]]][["Bewertung Qualität Originalautor","Text"]]

  # Fit files with ID name
  def extract_ID(text):
    try:
      return re.search("FIT.ID_(\d*).pdf",text).group(1)
    except:
      return None

  df_pdf["ID"] = df_pdf["Name"].apply(extract_ID)
  df_pdf["ID"] = [str(i) for i in df_pdf["ID"]]
  df_pdf_ID = df_pdf[df_pdf["ID"]!="None"][["ID","Text"]]

  return df_pdf_ID

def read_and_clean_texts(paths_to_pdfs, pdf_names):
  # read in all pdf texts
  pip_install(["tika","clean-text"])
  from tika import parser
  from cleantext.clean import clean
    
  pdf_list = list()

  for j,i in enumerate(pdf_names):
    try:
      # read in documents
      raw = parser.from_file(paths_to_pdfs + i)
      pdf_text = raw['content']
    except:
      raise Exception("PDF-files could not be read.")

    if pdf_text == None:
      pdf_list.append("None")
    else:
      # clean the text
      pdf_text_clean = clean(pdf_text,
          fix_unicode=True,               # fix various unicode errors
          to_ascii=True,                  # transliterate to closest ASCII representation
          lower=False,                    # lowercase text
          no_line_breaks=True,            # fully strip line breaks as opposed to only normalizing them
          no_urls=True,                   # replace all URLs with a special token
          no_emails=True,                 # replace all email addresses with a special token
          no_phone_numbers=True,          # replace all phone numbers with a special token
          no_numbers=True,                # replace all numbers with a special token
          no_digits=True,                 # replace all digits with a special token
          no_currency_symbols=True,       # replace all currency symbols with a special token
          no_punct=False,                 # fully remove punctuation
          replace_with_url="<URL>",
          replace_with_email="<EMAIL>",
          replace_with_phone_number="<PHONE>",
          replace_with_number="<NUMBER>",
          replace_with_digit="0",
          replace_with_currency_symbol="<CUR>",
          lang="en"                       # set to 'de' for German special handling
      )
      pdf_list.append(pdf_text_clean)

    print(str(np.round(j / len(pdf_names) *100,4)) + "% Progress")

  return pd.DataFrame(zip(pdf_names,pdf_list),columns=["Name","Text"])

### Helper function from daves_utilities ###

def pip_install(package):
  """
  This code snipet installs packages to the current python istance
  """
  for i in package:
    subprocess.run(sys.executable + " -m pip install " + i, shell=True)

def fun_save(fun_input, attr_input, path = "./", identifier : str = "", only_save_one = False):

    # create hidden folder under path to save function outputs
    fun_save_path = path + ".fun_save"
    if not os.path.exists(fun_save_path):
        os.makedirs(fun_save_path)

    # add underline to identifier if it exists
    identifier = "_" + identifier if identifier else identifier

    # structure to check pas calculations
    save_structure = f"{fun_input.__name__}{identifier}"
    files_found = [f"{fun_save_path}/{i}" for i in os.listdir(fun_save_path) if save_structure in i]

    # Warning if multiple files found while only_save_one = True
    if len(files_found) > 1 and only_save_one:
        print("only_save_one is True but multiple files were found. Only the last file will be overwritten if not output/input combination is found.")

    file_id = 0
    # open all files and compare attributes, return if same input/output is found
    for iter_path in files_found:
        # read files
        with open(iter_path,"rb") as f:
            output = pickle.load(f)
        
        # check if input is the same
        if is_equal(output["attr_input"],attr_input):
            print("Function file has been found and reused: " + save_structure)
            return output["fun_output"]
        
        # increament id if larger
        file_id_iter = int(iter_path.split("_")[-1].replace(".pkl",""))
        file_id = file_id_iter if file_id_iter > file_id else file_id

    # increment file_id
    if not only_save_one: file_id += 1
    print("Input/Output combination not found => recalculating function now: "  + save_structure)

    # create full file name with identifier for multiple input variations
    save_name = f"{fun_save_path}/{save_structure}_{file_id}.pkl"

    # run function
    fun_output = fun_input(**attr_input)
    output = {"attr_input":attr_input,"fun_output":fun_output}

    # save function output
    with open(save_name,"wb") as f:
        pickle.dump(output,f,protocol=4)

    return output["fun_output"]

def is_equal_recursive(elem1,elem2):
    if(type(elem1) != type(elem2)):
        return False
    if(isinstance(elem1,list)):
        if len(elem1) != len(elem2):
            return False
        return [is_equal_recursive(i,j) for i,j in zip(elem1,elem2)]
    elif(isinstance(elem1,dict)):
        if len(elem1) != len(elem2):
            return False
        elem1_in = list(elem1.values())
        elem2_in = list(elem2.values())
        return [is_equal_recursive(i,j) for i,j in zip(elem1_in,elem2_in)]
    elif(isinstance(elem1,int) or isinstance(elem1,float) or isinstance(elem1,str)):
        return elem1 == elem2
    elif "pandas" in str(type(elem1)):
        if elem1.shape != elem2.shape:
            return False
        return elem1.equals(elem2)
    elif "sklearn" in str(type(elem1)):
        return str(elem1) == str(elem2)

def list_flatten(l, a=None):
    if a is None: a = []
    l = [l] if isinstance(l,bool) else l
    for i in l:
        if isinstance(i, list):
            list_flatten(i, a)
        else:
            a.append(i)
    return a

def is_equal(elem1,elem2,flatten = True):
    boolen_list = is_equal_recursive(elem1,elem2)
    
    if flatten: return all(list_flatten(boolen_list))
    else: return boolen_list

### Embeddings ###

def tf_idf_data_cleaning(text):
  """
  Tokenization & Lemmtization & Stop Word removal

  [Manual for german stop-words](https://pypi.org/project/stop-words/)  
  [Tokenization and Lemmatization](https://data-science-blog.com/blog/2018/10/18/einstieg-in-natural-language-processing-teil-2-preprocessing-von-rohtext-mit-python/)
  """

  import nltk

  if not isinstance(text,list):
    if isinstance(text,str):
      text = [text]
    else:
      print("text need to be a list of strings or a single string")
      return

  # create or load cleaned data
  # Tokenization
  nltk.download('punkt')
  # Lemmatization
  snowball = nltk.SnowballStemmer(language="english")
  # Remove stopwords
  stop_words = get_stop_words("english")
  
  tokenized_list = list()
  removed_stopword_list = list()
  lemmatized_list = list()

  def normalization(x):
    import nltk
    # Tokenization
    token = nltk.word_tokenize(x)
    tokenized_list.append(token)

    # Remove stopwords
    removed_stopwords = [w for w in token if not w in stop_words]
    removed_stopword_list.append(removed_stopwords)

    # Lemmatization
    lemmatized = [snowball.stem(token_i) for token_i in removed_stopwords]
    lemmatized_list.append(lemmatized)

    return ' '.join(lemmatized)

  text_cleaned = [normalization(i) for i in text]

  return text_cleaned

def tf_idf_fit(text):
  """
  Create a tf-idf embedding from a text

  [tf-idf clustering](https://www.kaggle.com/jbencina/clustering-documents-with-tfidf-and-kmeans)
  """
  from sklearn.feature_extraction.text import TfidfVectorizer

  # clean text
  text_cleaned = tf_idf_data_cleaning(text)

  # Set stopwords
  stop_words = get_stop_words("english")

  # Define tf-idf model
  tfidf_model_raw = TfidfVectorizer(stop_words="english")

  tfidf_model = tfidf_model_raw.fit(text_cleaned)

  return tfidf_model

def tf_idf_pred(text, tfidf_model):

  # clean text
  text_cleaned = tf_idf_data_cleaning(text)

  # Fit the tf-idf model
  response = tfidf_model.transform(text_cleaned)

  # Create embedding DataFrame
  tf_idf_embedding = pd.DataFrame(response.todense())

  # rename columns of embedding
  tf_idf_embedding.columns = ["tf-idf_" + str(i) for i in list(tf_idf_embedding.columns)]
  
  return tf_idf_embedding


def word_2_vec_fit(text):
  """
  Create a word2vec embedding from a text

  [link to documentation](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
  """

  from gensim.test.utils import common_texts, get_tmpfile
  from gensim.models import Word2Vec
  import gensim.models

  if not isinstance(text,list):
    if isinstance(text,str):
      text = [text]
    else:
      print("text needs to be a list of strings or a single string")
      return

  # train your own word2vec model
  df_split = [i.split(" ") for i in text]
  word_2_vec_model = gensim.models.Word2Vec(sentences=df_split)

  return word_2_vec_model

def word_2_vec_pred(text, word_2_vec_model):

  if not isinstance(text,list):
    if isinstance(text,str):
      text = [text]
    else:
      print("text needs to be a list of strings or a single string")
      return
  
  df_split = [i.split(" ") for i in text]

  # transform the texts into word2vec embeddings
  embedding_texts = list()
  percentage_out_of_vocabulary_list = list()

  for num_t, df_text in enumerate(df_split):
    embedding_words = list()
    word_not_in_vocabulary_text = list()

    for num_w, df_word in enumerate(df_text):
      try:
        # add embedding of single word
        if df_word in word_2_vec_model.wv.vocab.keys():
          embedding_words.append(word_2_vec_model[str(df_word)])
        else:
          word_not_in_vocabulary_text.append(df_word)
      except Exception as e:
        # print("Error:" + str(e))
        word_not_in_vocabulary_text.append(df_word)
        pass

    # average over the returned word embeddings to get the text embedding
    embedding_texts.append(np.average(np.array(embedding_words),axis=0))

  # average out-of-vocabulary words
  print(str(np.round(np.average(percentage_out_of_vocabulary_list)*100,3)) + "% average percentage of out-of-vocabulary words")
  print("\n")

  # convert embedding to data frame and rename columns
  word2vec_embedding = pd.DataFrame(embedding_texts)
  word2vec_embedding.columns = ["word2vec_" + str(i) for i in list(word2vec_embedding.columns)]

  # add embedding to config file
  return word2vec_embedding


def bert_pred(text,parallel_boolean = False):
  """
  ## BERT

  [Documentation](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)

  The BERT embedding was created over different channels since it takes a very long time to calculate.
  Here we will just read in the pickle data which resulted as output of the BERT embedding.
  """

  if not isinstance(text,list):
    if isinstance(text,str):
      text = [text]
    else:
      print("text needs to be a list of strings or a single string")
      return

  # paralell processing
  from joblib import Parallel, delayed
  import multiprocessing
  num_cores = multiprocessing.cpu_count()

  # Bert model import
  import torch
  from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

  def unpack_if_list(df):
    if isinstance(df,list):
      if len(df) == 1:
        return str(df[0])
      else:
        print("ERROR: This list ist being unpacked even tough it has more than one element")
    if isinstance(df,str):
      return df

  def BERT_preprocessing_text(text):
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensors

  def BERT_postprocessing_encoded_layers(encoded_layers):
    
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    return token_embeddings

  def BERT_sentence_embedding(sentence):

    # create the token tensor and the segments tensor from the text
    tokens_tensor, segments_tensors = BERT_preprocessing_text(sentence)

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
      encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # post process encoded layers
    token_embeddings = BERT_postprocessing_encoded_layers(encoded_layers)

    # VERY important step
    # I average over all extracted layers AND over all tokens (words of the sentence)
    sentence_embedding = torch.mean(token_embeddings, dim=[0,1])

    return sentence_embedding

  # split lists which are longer than 500 strings long
  all_fine = list()

  sentences = [i.split(".") for i in text]

  for i in sentences:
    try:
      list_single_ID = list()
      for j in i:

        # ignore the string if it is shorter than 10 characters
        if len(j) < 10:
          continue

        # split the strings which are too long
        list_split = list(chunkstring(j,400))

        # append the stringe properly even if there are multiples now
        if len(list_split) > 1:
          for k in list_split:
            list_single_ID.append(unpack_if_list(k))
        else:
          list_single_ID.append(unpack_if_list(list_split))

      # append list of strings from ID to overarching list
      all_fine.append(list_single_ID)
    except:
      print("some error occurred with" + str(i[1][0]))

  # calculate embedding

  def processInput(l,j,i,single_len):
    if j % 10 == 0:
      print(str(round(j*100/single_len,2)) + "% SINGLE Progress")
    return BERT_sentence_embedding(i)

  def get_embedding(l,iter_elem, parallel_boolean=False):
    single_len = len(iter_elem)
    if parallel_boolean:
      out = Parallel(n_jobs=num_cores)(delayed(processInput)(l,j,i,single_len) for j,i in enumerate(iter_elem))
    else:
      out = [processInput(l,j,i,single_len) for j,i in enumerate(iter_elem)]
    return out

  embedding_list = list()
  # go over all elements
  for j,iter_elem in enumerate(all_fine):

    # calculate BERT
    BERT_embedding = get_embedding(j,iter_elem, parallel_boolean=parallel_boolean)
    embedding_list.append(BERT_embedding)

  embedding_formatted = list()
  for i in embedding_list:
    embedding_iter = [j.tolist() for j in i]
    embedding_formatted.append(list(np.average(np.array(embedding_iter),axis=0)))

  bert_embedding = pd.DataFrame(embedding_formatted)
  bert_embedding.columns = ["bert_" + str(i) for i in list(bert_embedding.columns)]

  return bert_embedding



if __name__=="__main__":
  
  print("#####################################################################################################################")
  print("start script")
  print("\n")

  path_to_pdfs = "/Users/davidkuchelmeister/Google Drive/Projektarbeit 2/Data/Documents/"
  create_text_data_file(path_to_pdfs)

  path = "/gdrive/My Drive/Projektarbeit 2/Data/"
  path = "/Users/davidkuchelmeister/Google Drive/Projektarbeit 2/Data/"
  df_large = pd.read_csv(path + "text_data.csv")
  text = list(df_large["Text"][0:2])

  tf_idf_model = tf_idf_fit(text)
  tf_idf_embedding = tf_idf_pred(text, tf_idf_model)

  word_2_vec_model = word_2_vec_fit(text)
  word_2_vec_embedding = word_2_vec_pred(text, word_2_vec_model)

  bert_pred(text)

  print("\n")
  print("end script")
  print("#####################################################################################################################")