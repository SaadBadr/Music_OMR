
import argparse
import os
import datetime
from project import *
# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")

args = parser.parse_args()

main_program(args.inputfolder, args.outputfolder)

with open(f"{args.outputfolder}/Output.txt", "w") as text_file:
    text_file.write("Input Folder: %s" % args.inputfolder)
    text_file.write("Output Folder: %s" % args.outputfolder)
    text_file.write("Date: %s" % datetime.datetime.now())
    


print('Finished !!') 
