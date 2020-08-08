#!/usr/bin/env python

# ./NLR.py -pdf_path /Users/davidkuchelmeister/Google\ Drive/Projektarbeit\ 2/Data/Documents/Aghaie\ et\ al._2014.pdf -nlr_path /Users/davidkuchelmeister/Google\ Drive/Projektarbeit\ 2/NLR/ -advanced False -save_past_runs False -install_packages False

import argparse
import os
import sys

def get_recommendation(pdf_path, nlr_path, advanced=False, save_past_runs=False):
    
    # set working directory
    os.chdir(nlr_path)
    
    # install packages
    sys.path.append(nlr_path + "/bin")
    import Data_Management as mgt
    import pandas as pd

    # hide all warnings
    import warnings
    warnings.filterwarnings('ignore')

    ### Load in Document
    text_pred = mgt.extract_pdf(pdf_path)
    if not text_pred: return "This pdf seems to be not in english. Please only select english texts.",None

    if advanced:
        ### fit BERT
        embedding = mgt.fun_save(mgt.bert_pred, {"text":text_pred,"parallel_boolean":True},\
                            identifier="bert", only_save_one = not save_past_runs)

        # Get trained classifiers
        embeddings_chosen = pd.read_csv("./data/bert_embeddings.csv",index_col=False)
    else:
        ### fit tf-idf
        text_data = pd.read_csv("./data/text_data.csv")
        text = list(text_data["Text"])

        tf_idf_model = mgt.fun_save(mgt.tf_idf_fit, {"text":text}, only_save_one = not save_past_runs)
        embedding = mgt.fun_save(mgt.tf_idf_pred, {"text":text_pred,"tfidf_model":tf_idf_model},
                            identifier="input", only_save_one = not save_past_runs)

        # Get trained classifiers
        embeddings_chosen = mgt.fun_save(mgt.tf_idf_pred, {"text":list(text_data["Text"]),"tfidf_model":tf_idf_model},
                                    identifier="fit", only_save_one = not save_past_runs)
        embeddings_chosen = pd.concat([text_data[["Thema","ID"]],embeddings_chosen], axis=1)

    ### create trained classifiers
    identifier_str = "bert" if advanced else "tf-idf"
    classifier_df = mgt.fun_save(fun_input=mgt.create_classifier_df,attr_input={"embedding":embeddings_chosen,"identifier_str":identifier_str},
                            identifier= identifier_str, only_save_one = not save_past_runs)


    ### predict labels
    predict_df = mgt.predict_top_n(embedding = embedding, embeddings_chosen = embeddings_chosen, classifier_df = classifier_df,
                                    top_n = 5, cut_under_n = 0)

    return predict_df, classifier_df, embeddings_chosen

def flush_memory(nlr_path):
    """
    This function will remove all saved memory files from the NLR application.
    """
    full_path = nlr_path + "/.fun_save/"
    files = os.listdir(full_path)
    del_files = [i for i in files if ".pkl" in i]

    for i in del_files:
        os.remove(full_path + i)

def install_packages(nlr_path):
    """
    Install packages using my own python code.

    Subprocess did not always work for some reason.
    """
    
    # import my own functions
    sys.path.append(nlr_path + "/bin")
    import Data_Management as mgt

    # install needed packages if specified
    mgt.pip_install(["pandas","numpy","pickle","stop_words","matplotlib","seaborn","sklearn","progressbar","nltk","gensim",\
                    "tika","cleantext","langdetect","joblib","multiprocessing","torch","pytorch_pretrained_bert"])


if __name__=="__main__":

    # Argument parser
    def parse_arguments():
        def file_path(path):
            if os.path.isfile(path):
                return path
            else:
                raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
        def dir_path(path):
            if os.path.isdir(path):
                return path
            else:
                raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
        def boolean(input_advanced):
            if isinstance(input_advanced,str):
                return "TRUE" in input_advanced.upper()
            else:
                raise argparse.ArgumentTypeError(f"advanced attribute can not be converted to boolean: {boolean}")

        parser = argparse.ArgumentParser(description='Process command line arguments.')
        parser.add_argument('-pdf_path', type=file_path)
        parser.add_argument('-nlr_path', type=dir_path)
        parser.add_argument('-advanced', type=boolean)
        parser.add_argument('-install_packages', type=boolean)
        parser.add_argument('-save_past_runs', type=boolean)

        parsed_args = parser.parse_args()

        pdf_path = parsed_args.pdf_path
        nlr_path = parsed_args.nlr_path
        advanced = parsed_args.advanced
        save_past_runs = parsed_args.save_past_runs
        install_packages = parsed_args.install_packages

        return pdf_path, nlr_path, advanced, save_past_runs, install_packages

    # arg parse
    pdf_path, nlr_path, advanced, save_past_runs, install_packages = parse_arguments()

    # if run in IDE this weill replace the arg input
    if not nlr_path and not pdf_path:
        # attributes
        pdf_path = "/Users/davidkuchelmeister/Google Drive/Projektarbeit 2/Data/Documents/Albaladejo et al._2010.pdf"
        pdf_path = "/Users/davidkuchelmeister/Google Drive/Projektarbeit 2/Data/Documents/Coelho et al._2016.pdf"

        nlr_path = "/Users/davidkuchelmeister/Google Drive/Projektarbeit 2/NLR"
        advanced = True
        save_past_runs = False
        install_packages = False

    def print_input_attributes(attr_names):
        print()
        print("Input attributes")
        print("================================")
        for i in attr_names:
            print(f"{i}: {globals()[i]}")
        print("================================")
        print()

    print_input_attributes(["pdf_path", "nlr_path", "advanced", "save_past_runs", "install_packages"])

    # Python version warning
    if sys.version_info.major < 3:
        sys.exit("This project was built on python 3.6, please run this project at least with a python version greater than 3")

    # import my own functions
    sys.path.append(nlr_path + "/bin")
    import Data_Management as mgt

    # install needed packages if specified
    if install_packages:
        mgt.pip_install(["pandas","numpy","pickle","stop_words","matplotlib","seaborn","sklearn","progressbar","nltk","gensim",\
                        "tika","cleantext","langdetect","joblib","multiprocessing","torch","pytorch_pretrained_bert"])

    import pandas as pd
    import pickle

    # create recommendation
    predict_df, classifier_df, embeddings_chosen = get_recommendation(pdf_path=pdf_path, nlr_path=nlr_path, advanced=advanced)

    # print results
    print("\n")
    print("Results of prediction")
    print(predict_df.sort_values("Probability",ascending=False).reset_index(drop=True))

    ### get scores
    scores = mgt.get_scores(classifier_df, embeddings_chosen, top_n = 3, cut_under_n = 0, plot_boolean=True)