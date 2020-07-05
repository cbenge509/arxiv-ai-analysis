############################################################################
# IMPORTS
############################################################################

import os
import argparse
from utils import preprocessing


############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# Default file Locations (parameters)
OUTPUT_FILE_PATH = "processed_data/"
OUTPUT_FILENAME = "preprocessed_data.csv"
INPUT_FILE_PATH = "processed_data/"
INPUT_FILENAME = "w299_categories.hdf5"

# Processing behavior (parameters)
VERBOSE = True

############################################################################
# ARGUMENT SPECIFICATION
############################################################################
parser = argparse.ArgumentParser(description = "Generates SQuAD v2 features from raw data and stores them in binary files.")
# Commandline arguments
parser.add_argument('-nv', '--no_verbose', action = 'store_true', help = 'Disables verbose output mode for more detailed descriptions of process.')
parser.add_argument('-ofp', '--output_filepath', type = str, default = OUTPUT_FILE_PATH, help = "Path of location to store the output dataframe in CSV format.")
parser.add_argument('-ofn', '--output_filename', type = str, default = OUTPUT_FILENAME, help = "Filename of HDF5 categories dataframe output file (recommended format: <filename>.csv).")
parser.add_argument('-ifp', '--input_filepath', type = str, default = INPUT_FILE_PATH, help = "Path of location to the input file in HDF5 format.")
parser.add_argument('-ifn', '--input_filename', type = str, default = INPUT_FILENAME, help = "Filename of HDF5 file containing categories dataframe output file (recommended format: <filename>.hdf5).")

############################################################################
# ARGUMENT PARSING
############################################################################
def process_arguments(parsed_args, display_args = False):
    
    global VERBOSE, OUTPUT_FILE_PATH, OUTPUT_FILENAME, INPUT_FILE_PATH, INPUT_FILENAME
    
    args = vars(parser.parse_args())
    if display_args:
        print("".join(["*" * 30, "\narguments in use:\n", "*" * 30, "\n"]))
        for arg in args:
            print("Parameter '%s' == %s" % (arg, str(getattr(parser.parse_args(), arg))))
        print("\n")

    # Assign arguments to globals
    VERBOSE = not args['no_verbose']
    OUTPUT_FILE_PATH = args['output_filepath'].strip().lower().replace('\\', '/')
    OUTPUT_FILENAME = args['output_filename'].strip().lower()
    INPUT_FILE_PATH = args['input_filepath'].strip().lower().replace('\\', '/')
    INPUT_FILENAME = args['input_filename'].strip().lower()
    
    # validate the existence of the caller-specified paths
    for p, v, l in zip([OUTPUT_FILE_PATH, INPUT_FILE_PATH], ['output_filepath', 'input_filepath'], ['Directory of output CSV dataframe file.', 'Directory of input HDF5 categories file.']):
        if not os.path.exists(p):
            raise RuntimeError(" ".join([l, "'%s'" % p, "specified in parameter `%s` does not exist." % v]))
    
    # Filename validation
    if len(OUTPUT_FILENAME) < 1: raise RuntimeError("Output filename provided may not be zero length.")
    if len(INPUT_FILENAME) < 1: raise RuntimeError("Input filename provided may not be zero length.")

    # Paper categories input validation

############################################################################
# CLEANSE DATA
############################################################################

def LoadCleanedData(processor, verbose = False):

    df = processor.Process(verbose = verbose)
    return df

############################################################################
# STORE DATA
############################################################################

def StoreCleanedData(df, output_file_path, output_filename, verbose = False):

    fname = os.path.join(output_file_path, output_filename)
    df.to_csv(fname, index = False, sep = ',', mode = 'w')
    if verbose: print(f"Output file written to '{fname}'.")

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

    print("".join(["-" * 100, "\n>>> DATA PROCESSING INITIATED <<<\n", "-" * 100, "\n"]))

    # Genrate the feature data and store as binary
    processor = preprocessing.CleanData(input_file_path = INPUT_FILE_PATH, input_filename = INPUT_FILENAME, verbose = VERBOSE)

    # Load Clean Data
    df = LoadCleanedData(processor, verbose = VERBOSE)

    # Store the Clean Data
    StoreCleanedData(df, output_file_path = OUTPUT_FILE_PATH, output_filename = OUTPUT_FILENAME, verbose = VERBOSE)

    print("".join(["-" * 100, "\n>>> DATA PROCESSING COMPLETE <<<\n", "-" * 100, "\n"]))

