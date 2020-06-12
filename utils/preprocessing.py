############################################################################
# IMPORTS
############################################################################

import pandas as pd
import h5py
import os
import glob

############################################################################
# CONSTANTS
############################################################################

PAPER_CATEGORIES = ['cs.AI', 'cs.LG', 'stat.ML']

############################################################################
# CLASS : SQuADv2Utils
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
            verbose (bool, optional): [description]. Defaults to False.
        """

        self.__verbose = verbose

        # validate that the constructor parameters were provided by caller
        if (not base_categories_path) | (not paper_categories):
            raise RuntimeError('Input parameters `base_categories_path` and `paper_categories` must be specified.')
        if (not isinstance(paper_categories, list)):
            raise RuntimeError('Input parameter ``paper_categories`` must be of type list.')
        paper_categories = list(filter(None, [s.strip() for s in paper_categories]))
        if len(paper_categories) < 1:
            raise RuntimeError('Input parameter ``paper_categories`` must contain at least one non-empty element in the list.')

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

        return df_all

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
