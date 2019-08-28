### get sentiment of each line using SentiStrength

import subprocess
import shlex
import os.path
import sys
from tqdm import tqdm ## package to show progress
from pathos.multiprocessing import ProcessingPool as Pool

SentiStrengthLocation = "/research/lyu1/cygao/workspace/sentiment_tool/SentiStrengthCom.jar" #The location of SentiStrength on your computer
SentiStrengthLanguageFolder = "/research/lyu1/cygao/workspace/sentiment_tool/SentStrength_Data/" #The location of the unzipped SentiStrength data files on your computer
train_file = "/research/lyu1/cygao/workspace/data/single_train_data.txt" #The location of the file that you want classified.
test_file = "/research/lyu1/cygao/workspace/data/single_test_data.txt"
valid_file = "/research/lyu1/cygao/workspace/data/single_valid_data.txt"

output_train = "/research/lyu1/cygao/workspace/data/single_train_senti.txt"
output_test = "/research/lyu1/cygao/workspace/data/single_test_senti.txt"
output_valid = "/research/lyu1/cygao/workspace/data/single_valid_senti.txt"


## The code below allows SentiStrength to be called and run on a single line of text.
def RateSentiment(sentiString):
    #open a subprocess using shlex to get the command line string into the correct args list format
    p = subprocess.Popen(shlex.split("java -jar '" + SentiStrengthLocation + "' stdin sentidata '" + SentiStrengthLanguageFolder + "'"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #communicate via stdin the string to be rated. Note that all spaces are replaced with +
    new_sentistr = sentiString.replace(" ","+")
    #b = bytes(new_sentistr, 'utf-8') #Can't send string in Python 3, must send bytes
    b = new_sentistr
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")  #convert from byte
    stdout_text = stdout_text.rstrip().replace("\t"," ") #remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1 -5
    return stdout_text
    # return stdout_text + " " + sentiString

def FileSentiment(fpath):
    if not os.path.isfile(fpath):
        print("File to classify not found at: ", fpath)

    print("Running SentiStrength on file " + fpath + " with command:")
    cmd = 'java -jar "' + SentiStrengthLocation + '" sentidata "' + SentiStrengthLanguageFolder + '" input "' + fpath + '"'
    print(cmd)
    p = subprocess.Popen(shlex.split(cmd),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    classifiedSentimentFile = os.path.splitext(fpath)[0] + "0_out.txt"
    print("Finished! The results will be in:\n" + classifiedSentimentFile)

def get_data(file):
    fr = open(file)
    lines = fr.readlines()
    fr.close()
    return lines

def sub_process(row):
    terms = row.split('***')
    review = terms[4]
    output_senti = RateSentiment(review)
    return row.strip()+'***'+output_senti+'\n'


def main():
    output_review = []
    lines = get_data(valid_file)
    pool = Pool(8)
    block_num = 1000
    block_size = len(lines)//block_num

    for i in tqdm(range(block_num+1)):
        if i == block_num:
            block = lines[i*block_size:]
        else:
            block = lines[i*block_size:(i+1)*block_size]
        tunnel = pool.amap(sub_process, block)
        output = tunnel.get()
        output_review += output

    fw = open(output_valid, "w")
    fw.writelines(output_review)
    fw.close()

if __name__ == "__main__":
    main()





