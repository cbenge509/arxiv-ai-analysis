############################################################################
# IMPORTS
############################################################################

import pandas as pd
import numpy as np
import spacy
import h5py
import os
import glob
from langdetect import detect, DetectorFactory
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

############################################################################
# CONSTANTS
############################################################################

PAPER_CATEGORIES = ['cs.AI', 'cs.LG', 'stat.ML']

############################################################################
# CLASS : LoadCategories
############################################################################

class LoadCategories(object):

    #---------------------------------------------------------------------------
    # CLass Initialization
    #---------------------------------------------------------------------------
    def __init__(self, base_categories_path, paper_categories = PAPER_CATEGORIES, verbose = False):
        """LoadCategories class initialization

        Args:
            base_categories_path (str): Path to the location of the base pre-processed arXiv categories data.
            paper_categories (list(str), optional): List of categories to retrieve from the pre-processed dataset. Defaults to ['cs.AI', 'cs.LG', 'stat.ML'].
            verbose (bool, optional): Enables verbose logging. Defaults to False.
        """

        self.__verbose = verbose

        # validate that the constructor parameters were provided by caller
        if (not base_categories_path) | (not paper_categories):
            raise ValueError('Input parameters `base_categories_path` and `paper_categories` must be specified.')
        if (not isinstance(paper_categories, list)):
            raise ValueError('Input parameter ``paper_categories`` must be of type list.')
        paper_categories = list(filter(None, [s.strip() for s in paper_categories]))
        if len(paper_categories) < 1:
            raise ValueError('Input parameter ``paper_categories`` must contain at least one non-empty element in the list.')

        # clean and validate the path strings
        self.__base_categories_path = self.__clean_path(base_categories_path, "Path to raw arXiv data")

        # validate existence of the expected base arXiv data in the caller-specified base data path
        files = glob.glob(self.__base_categories_path + "*.tsv.xz")
        if len(files) < 1:
            raise RuntimeError(f"Base categories *.tsv.xz files not found in specified path {self.__base_categories_path}.")
        files.sort()

        # verify that the category files exist; create array of category-to-file
        category_files = []
        for cat in paper_categories:
            file_found = False
            for f in files:
                if cat.lower() in f.lower():
                    file_found = True
                    category_files.append(f)
                    break
            if (not file_found):
                raise RuntimeError(f"No corresponding file found in directory '{self.__base_categories_path}' for category '{cat}'.")
        
        self.__category_base_files = {
            "categories":paper_categories,
            "files":category_files
        }

        if verbose: print("All specified categories and base files located.")

    #---------------------------------------------------------------------------
    # (Utility) __clean_path
    #---------------------------------------------------------------------------
    def __clean_path(self, clean_path, path_title = "Unspecified"):
        """Common utility function for cleaning and validating path strings

        Args:
            clean_path (str): the path string to clean & validate
            path_title (str): the title (or name) of the path; used on error only.

        Returns:
            str: the cleaned path string
        """

        clean_path = str(clean_path).replace('\\', '/').strip()
        if (not clean_path.endswith('/')): clean_path = ''.join((clean_path, '/'))

        if (not os.path.isdir(clean_path)):
            raise RuntimeError("'%s' path specified '%s' is invalid." % (path_title, clean_path))

        return clean_path

    #---------------------------------------------------------------------------
    # (Public Method) GetDataframeFromBase
    #---------------------------------------------------------------------------
    def GetDataframeFromBase(self, verbose = False):
        """Retrieves a Pandas Datafreame object from the pre-processed arXiv category dataset.

        Args:
            verbose (bool, optional): Enables verbose logging on the console. Defaults to False.

        Returns:
            Pandas.Dataframe: Dataframe of the arXiV categories data specified by the caller during class initialization.
        """

        dtypes = {
            "abstract": object,
            "acm_class": object,
            "arxiv_id": object,
            "author_text": object,
            "categories": object,
            "comments": object,
            "created": object,
            "doi": object,
            "num_authors": int,
            "num_categories": int,
            "primary_cat": object,
            "title": object,
            "updated": object
        }
        
        df_all = pd.DataFrame()
        for i, c in enumerate(self.__category_base_files["categories"]):
            fn = self.__category_base_files["files"][i]
            if verbose: print(f"Loading category {c} from file '{fn}...")
            
            category_df = pd.read_csv(fn, 
                                      sep = "\t", 
                                      index_col = 0, 
                                      compression = 'xz', 
                                      usecols = ['abstract','acm_class','arxiv_id','author_text','categories',
                                        'comments','created','doi','num_authors','num_categories','primary_cat',
                                        'title','updated'],
                                      dtype = dtypes, 
                                      parse_dates = ["created", "updated"])
            
            df_all = df_all.append(category_df)
            if verbose: print("\trecords process for category: {cat_len}, cumulative total: {df_len}.".format(cat_len = category_df.shape[0], df_len = df_all.shape[0]))

        if verbose: print("\nDataframe processing complete for all categories.\n")

        return df_all.reset_index()

    #---------------------------------------------------------------------------
    # (Public Method) GetDataframeFromBase
    #---------------------------------------------------------------------------
    def StoreCategoriesDataframe(self, categories_df, store_path, filename, verbose = False):
        """Stores the arXiv categories dataframe to HDFF5 file under dataset name 'arxiv_categories'

        Args:
            categories_df (Pandas.Dataframe): Dataframe containing the arXiv categories data.
            store_path (str): User-specified path to store categories dataframe to.
            filename (str): Name of serialized pandas dataframe file (recommend format: filename.hdf5)
            verbose (bool, optional): Enables verbose logging on the console. Defaults to False.
        """

        # Input validation
        if (categories_df is None) | (not store_path) | (not filename):
            raise RuntimeError("Input variables `categories_df`, `store_path`, and `filename` are required.")
        if categories_df.shape[0] < 1:
            raise RuntimeError("Dataframe `categories_df` must contain at least one row.")
        if (not isinstance(categories_df, pd.DataFrame)):
            raise RuntimeError("Input variable `categories_df` expected to be of type Pandas.Dataframe.")
        if (not isinstance(store_path, str)) | (not isinstance(filename, str)):
            raise RuntimeError("Input variables `store_path` and `filename` expected to be of type str.")
        filename = filename.strip().lower()
        if len(filename) < 1:
            raise RuntimeError("Filename must not be an empty string; recommended format is '<filename>.hdf5'")

        store_path = self.__clean_path(store_path, "User-specified path to store categories dataframe to.")
        
        # Store the dataframe
        filename = os.path.join(store_path, filename)
        dt = h5py.string_dtype()
        with h5py.File(filename, "w") as f:
            f.create_dataset("arxiv_categories", data = categories_df, dtype = dt)
        
        if verbose: print(f"HDF5 file '{filename}' written to disk.\n")

        return

############################################################################
# CLASS : CleanData
############################################################################

class CleanData(object):

    #---------------------------------------------------------------------------
    # CLass Initialization
    #---------------------------------------------------------------------------
    def __init__(self, input_file_path, input_filename, verbose = False):
        """CleanData class initialization

        Args:
            input_file_path (str): filename of the pre-processed arXiv categories data file.
            input_filename (str): Path to the location of the pre-processed arXiv categories data file.
            verbose (bool, optional): Enables verbose logging. Defaults to False.
        """

        self.__verbose = verbose

        # validate that the constructor parameters were provided by caller
        if (not input_file_path) | (not input_filename):
            raise ValueError('Input parameters `input_file_path` and `input_filename` must be specified.')

        # clean and validate the path strings
        input_file_path = self.__clean_path(input_file_path, "Path to processed arXiv categories data file")

        # validate existence of the expected base arXiv data in the caller-specified base data path
        input_file_and_path = os.path.join(input_file_path, input_filename)
        if (not os.path.isfile(input_file_and_path)):
            raise RuntimeError(f"File {input_file_and_path} is not found.")

        # load the raw dataframe from categories file (no processing)
        self.__data_raw = self.__process_hdf5(input_file_and_path = input_file_and_path)

        if verbose: print(f"Categories file '{input_file_and_path}' successfully loaded with shape: {self.__data_raw.shape}.")

    #---------------------------------------------------------------------------
    # (Utility) __clean_path
    #---------------------------------------------------------------------------
    def __clean_path(self, clean_path, path_title = "Unspecified"):
        """Common utility function for cleaning and validating path strings

        Args:
            clean_path (str): the path string to clean & validate
            path_title (str): the title (or name) of the path; used on error only.

        Returns:
            str: the cleaned path string
        """

        clean_path = str(clean_path).replace('\\', '/').strip()
        if (not clean_path.endswith('/')): clean_path = ''.join((clean_path, '/'))

        if (not os.path.isdir(clean_path)):
            raise RuntimeError("'%s' path specified '%s' is invalid." % (path_title, clean_path))

        return clean_path
    
    #---------------------------------------------------------------------------
    # (Utility) __process_hdf5
    #---------------------------------------------------------------------------

    def __process_hdf5(self, input_file_and_path, verbose = False):
        """Loads the raw arXiv categories dataframe from HDF5 format.

        Args:
            input_file_and_path (str): Path and filename of the HDF5 file containing arXiv categories data.
            verbose (bool, optional): Enables verbose logging. Defaults to False.
        """

        # specifies the base column names for the processed arXiv categories file
        category_cols = ['abstract','acm_class','arxiv_id','author_text','categories',
                    'comments','created','doi','num_authors','num_categories','primary_cat',
                    'title','updated']

        # load the dataframe
        with h5py.File(input_file_and_path, 'r') as f:
            df = pd.DataFrame(np.array(f['arxiv_categories']), columns = category_cols)

        # adjust the data types for numeric and date data
        df[['num_authors', 'num_categories']] = df[['num_authors', 'num_categories']].apply(pd.to_numeric)
        df[['created','updated']] =df[['created', 'updated']].apply(pd.to_datetime)

        return df

    #---------------------------------------------------------------------------
    # (Public Method) Process
    #---------------------------------------------------------------------------
    def Process(self, verbose = False):
        """Cleans the pre-loaded categories dataframe data, preparing
           it for exploratory data analysis.

        Args:
            verbose (bool, optional): Enables verbose logging. Defaults to False.

        Raises:
            RuntimeError: categories dataframe was not properly loaded during class init.

        Returns:
            [Pandas.DataFrame]: Dataframe containing cleaned arXiv categories data
        """

        # validate the categories data has been loaded at init time
        if (not isinstance(self.__data_raw, pd.DataFrame)):
            raise RuntimeError("Initial HDF5 categories data failed to load; reload the class.")
    
        df = self.__data_raw
        
        ##################################################################
        ### CLEAN-UP
        ##################################################################

        # STEP. 1 : Create categoires columns for cs.AI, cs.LG, and stat.ML
        if verbose: print("Clean.Process - Step 1 : Create category one-hot encodings...")

        # one-hot encoding for categories
        def identify_categories(cat, target_cat):
            return (target_cat in cat)

        df['category_cs_AI'] = df.categories.apply(lambda x: identify_categories(x, 'cs.AI'))
        df['category_cs_LG'] = df.categories.apply(lambda x: identify_categories(x, 'cs.LG'))
        df['category_stat_ML'] = df.categories.apply(lambda x: identify_categories(x, 'stat.ML'))
        df.drop(columns = ['categories'], inplace = True)

        # STEP. 2 : Calculate the sentence count
        if verbose: print("Clean.Process - Step 2 : Calculate the sentence counts...")
        # count the number of sentences per abstract using sentence detection with spaCy
        # this takes a few minutes...
        sp = spacy.load('en_core_web_lg')
        def sentence_count(t):
            return len(list(sp(t.strip().lower()).sents))
        df['abstract_sentence_count'] = df.abstract.apply(lambda x: sentence_count(x))

        # STEP. 3 : Calculate word counts in abstracts and titles
        if verbose: print("Clean.Process - Step 3 : Calculate word counts...")
        # capture word count in abstract (separted by spaces)
        df['abstract_word_count'] = df.abstract.apply(lambda x: len(x.strip().split())) 
        # capture number of unique words in abstract body
        df['abstract_unique_word_count'] = df.abstract.apply(lambda x:len(set(str(x).split())))
        # capture the word count in title
        df['title_word_count'] = df.title.apply(lambda x: len(x.strip().split()))


        # STEP. 4 : language detection
        if verbose: print("Clean.Process - Step 4 : Stopword, space, and numeric data removal from abstract...")
        # create a "clean" version of the abstract for word-cloud visualization and perhaps topic modeling
        def clean_text(t):
            clean_tokens = [token.lemma_ for token in sp(t) if token.is_alpha & (not token.is_stop) & (not token.is_space) & token.is_alpha]
            return " ".join(clean_tokens)

        df['abstract_clean'] = df.abstract.apply(lambda x: clean_text(x.strip().lower()))

        # downcast variables
        dtypes = {}
        for i in ['num_authors', 'num_categories', 'title_word_count']:
            dtypes[i] = np.uint8
        for i in ['abstract_sentence_count', 'abstract_word_count', 'abstract_unique_word_count']:
            dtypes[i] = np.uint16

        # STEP. 5 : language detection
        if verbose: print("Clean.Process - Step 5 : Perform language detection...")

        DetectorFactory.seed = 0
        languages = []

        for i in range(0, len(df)):
            if np.max([df.iloc[i]['abstract_word_count'], len(df.iloc[i]['title'].strip().split(" "))]) > 1:
                if df.iloc[i]['abstract_word_count'] > 1:
                    text, lang = df.iloc[i]['abstract'].split(" ")[:50], "en"
                else:
                    text, lang = df.iloc[i]['title'].split(" ")[:50], "en"
            try:
                if len(text) > 1:
                    lang = detect(" ".join(text[:len(text)]))
                else:
                    lang = "unknown"
            except Exception as e:
                all_words = set(text)
                try:
                    lang = detect(" ".join(all_words))
                except Exception as e:
                    lang = "unknown"
                    pass
            
            languages.append(lang)
        df['language'] = languages

        # STEP 6 : KMeans Clustering
        if verbose: print("Clean.Process - Step 6 : TfIdf (top 4k) -> Kmeans clustering assignment...")
        
        def vectorize(text, max_features):
            vectorizer = TfidfVectorizer(max_features = max_features)
            v = vectorizer.fit_transform(text)
            return v
        
        # set the dropped rows aside (not used during clustering)
        df_lang = df[(df.language != 'en')].copy()
        df_noabstract = df[(df.abstract_clean.isna())].copy()

        df = df[(df.language == 'en')]
        df = df[(~df.abstract_clean.isna())]

        # tf-idf
        clean_text = vectorize(text = df.abstract_clean.values, max_features = 2 ** 12)

        # utilize PCA dimensionality reduction, retaining 95% components
        pca = PCA(n_components = 0.95, random_state = 42)
        reduced = pca.fit_transform(clean_text.toarray())

        # K-mean
        k = 15
        km = KMeans(n_clusters = k, random_state = 42).fit(reduced)
        df['cluster'] = km.predict(reduced)

        # add the dropped values back; put them in cluster nan
        df_lang['cluster'] = np.nan
        df_noabstract['cluster'] = np.nan

        df = pd.concat([df, df_lang, df_noabstract], axis = 0)

        # STEP 7. TSNE embeddings
        if verbose: print("Clean.Process - Step 7 : TSNE embeddings...")
        
        tsne = TSNE(verbose = verbose, perplexity = 100, random_state = 42)
        tsne_embedded = tsne.fit_transform(clean_text.toarray())



        if verbose: print("Clean.Process() complete.")

        return df