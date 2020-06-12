############################################################################
# IMPORTS
############################################################################

import pandas as pd
import numpy as np
from utils import preprocessing
import argparse
import os

############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# Default file Locations (parameters)
CATEGORY_FILE_PATH = "processed_data/20200101/per_category"
OUTPUT_FILE_PATH = "processed_data/"
OUTPUT_FILENAME = "w299_categories.hdf5"
PAPER_CATEGORIES = ['cs.AI', 'cs.LG', 'stat.ML']

# Processing behavior (parameters)
VERBOSE = True

############################################################################
# ARGUMENT SPECIFICATION
############################################################################
parser = argparse.ArgumentParser(description = "Generates SQuAD v2 features from raw data and stores them in binary files.")
# Commandline arguments
parser.add_argument('-nv', '--no_verbose', action = 'store_true', help = 'Disables verbose output mode for more detailed descriptions of process.')
parser.add_argument('-bdp', '--base_category_filepath', type = str, default = CATEGORY_FILE_PATH, help = "Path to location of arXiv pre-processed category files.")
parser.add_argument('-ofp', '--output_filepath', type = str, default = OUTPUT_FILE_PATH, help = "Path of location to store the output dataframe in HDF5 format.")
parser.add_argument('-ofn', '--output_filename', type = str, default = OUTPUT_FILENAME, help = "Filename of HDF5 categories dataframe output file (recommended format: <filename>.hdf5).")
parser.add_argument('-pc', '--paper_categories', nargs = '+', type = str, default = PAPER_CATEGORIES, help = "List of paper categories to load into dataframe for analysis.")

############################################################################
# ARGUMENT PARSING
############################################################################
def process_arguments(parsed_args, display_args = False):
    
    global VERBOSE, CATEGORY_FILE_PATH, OUTPUT_FILE_PATH, OUTPUT_FILENAME, PAPER_CATEGORIES
    
    args = vars(parser.parse_args())
    if display_args:
        print("".join(["*" * 30, "\narguments in use:\n", "*" * 30, "\n"]))
        for arg in args:
            print("Parameter '%s' == %s" % (arg, str(getattr(parser.parse_args(), arg))))
        print("\n")

    # Assign arguments to globals
    VERBOSE = not args['no_verbose']
    CATEGORY_FILE_PATH = args['base_category_filepath'].strip().lower().replace('\\', '/')
    OUTPUT_FILE_PATH = args['output_filepath'].strip().lower().replace('\\', '/')
    OUTPUT_FILENAME = args['output_filename'].strip().lower()
    PAPER_CATEGORIES = args['paper_categories']
    
    # validate the existence of the caller-specified paths
    for p, v, l in zip([CATEGORY_FILE_PATH, OUTPUT_FILE_PATH], ['base_category_filepath', 'output_filepath'], ['Location of base pre-processed arXiv category data.', 'Directory of output HDF5 dataframe file.']):
        if not os.path.exists(p):
            raise RuntimeError(" ".join([l, "'%s'" % p, "specified in parameter `%s` does not exist." % v]))
    
    # Filename validation
    if len(OUTPUT_FILENAME) < 1: raise RuntimeError("Filename provided may not be zero length.")

    # Paper categories input validation
    if (not isinstance(PAPER_CATEGORIES, list)): raise RuntimeError("Input parameter `paper_categories` must be of type list.")
    if (not len(PAPER_CATEGORIES) > 0): raise RuntimeError("Input list `paper_categories` must contain at least one value.")
    if (not all(isinstance(x, str) for x in PAPER_CATEGORIES)): raise RuntimeError("All values in `paper_categories` list must be of type str.")


############################################################################
# LOAD THE BASE CATEGORIES DATA
############################################################################

def LoadBaseData(processor, verbose = False):

    categories_df = processor.GetDataframeFromBase(verbose = verbose)
    return categories_df

############################################################################
# STORE THE BASE CATEGORIES DATA
############################################################################
def StoreBaseData(processor, categories_df, store_path, filename, verbose = False):

    processor.StoreCategoriesDataframe(categories_df = categories_df, store_path = store_path, filename = filename, verbose = verbose)
    return

############################################################################
# MAIN FUNCTION
############################################################################

if __name__ == "__main__":

    # Clear the screen
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
    
    # Process command-line arguments and set parameters
    process_arguments(parser.parse_args(), display_args = True)

    print("".join(["-" * 100, "\n>>> ARXIV CATEGORIES DATAFRAME GENERATION BEGIN <<<\n", "-" * 100, "\n"]))

    # Genrate the feature data and store as binary
    processor = preprocessing.LoadCategories(base_categories_path = CATEGORY_FILE_PATH,
        paper_categories = PAPER_CATEGORIES, verbose = VERBOSE)

    # Load the categories dataframe from the base category files
    categories_df = LoadBaseData(processor = processor, verbose = VERBOSE)

    categories_df = categories_df.reset_index()
    for i in categories_df.columns:
        print(i, "-", categories_df[i][0])

    # TODO: Add in data cleaning or other preprocessing steps, if needed.

    # Save the base data
    StoreBaseData(processor = processor, categories_df = categories_df, store_path = OUTPUT_FILE_PATH,
        filename = OUTPUT_FILENAME, verbose = VERBOSE)

    print("".join(["-" * 100, "\n>>> ARXIV CATEGORIES DATAFRAME GENERATION COMPLETE <<<\n", "-" * 100, "\n"]))

