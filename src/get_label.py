### Get the categories of each review

import os, re, time
from dicttoxml import dicttoxml
import xmltodict
from tqdm import tqdm ## package to show progress
from pathos.multiprocessing import ProcessingPool as Pool
import traceback
import itertools
import subprocess, shlex
from bs4 import BeautifulSoup


train_file = "/research/lyu1/cygao/workspace/data/train_senti.txt" #The location of the file that you want classified.
valid_file = "/research/lyu1/cygao/workspace/data/valid_senti.txt"
test_file = "/research/lyu1/cygao/workspace/data/test_senti.txt"

train_dir = "/research/lyu1/cygao/workspace/data/train"
valid_dir = "/research/lyu1/cygao/workspace/data/valid"
test_dir = "/research/lyu1/cygao/workspace/data/test"


train_xml = "/research/lyu1/cygao/workspace/data/xml_input/train"
valid_xml = "/research/lyu1/cygao/workspace/data/xml_input/valid"
test_xml = "/research/lyu1/cygao/workspace/data/xml_input/test"

train_surf = "/research/lyu1/cygao/workspace/data/surf/train"
valid_surf = "/research/lyu1/cygao/workspace/data/surf/valid"
test_surf = "/research/lyu1/cygao/workspace/data/surf/test"

output_train = "/research/lyu1/cygao/workspace/data/train_label.txt"
output_valid = "/research/lyu1/cygao/workspace/data/valid_label.txt"
output_test = "/research/lyu1/cygao/workspace/data/test_label.txt"


def convert_xml(data, foutpath):
    review_sum = {}
    review_sum["reviews"] = {}
    for idx, lines in enumerate(data):
        line = lines.split("***")
        key_name = "review id=" + str(idx)
        review_sum["reviews"][key_name] = {}
        review_sum["reviews"][key_name]["app_version"] = "NA"
        review_sum["reviews"][key_name]["user"] = "NA"
        review_sum["reviews"][key_name]["date"] = "NA"
        review_sum["reviews"][key_name]["star_rating"] = line[1]
        review_sum["reviews"][key_name]["review_title"] = "NA"
        review_sum["reviews"][key_name]["review_text"] = line[4]

    xml = dicttoxml(review_sum, custom_root='reviews_summary', attr_type=False)
    filter_xml = re.sub('key name="', '', xml)
    filter_xml = re.sub('id=', 'id="', filter_xml)
    filter_xml = re.sub("key", "review", filter_xml)

    soup = BeautifulSoup(filter_xml, 'xml')
    fw = open(foutpath, "w")
    fw.write(soup.prettify())
    fw.close()

def get_data(file):
    fr = open(file)
    lines = fr.readlines()
    fr.close()
    return lines

def run_surf(finpath, foutpath):
    command_script = "java -Xmx256m -classpath './lib/*:./SURF.jar' org.surf.Main " + finpath + " " + foutpath +" &>/dev/null"
    output = os.system(command_script)  #wait for the command line to finish
    print("output of running surf is ", output)
    if output:
        return "We finish running surf."

def modify_xml(finpath):
    fr = open(finpath)
    soup = BeautifulSoup(fr.read(), 'xml')
    fr.close()
    fw = open(finpath, "w")
    fw.write(soup.prettify())
    fw.close()



def get_label(line, rows):
    try:
        topics = xmltodict.parse(line)
        # topics = BeautifulSoup(line, 'xml')
        rev_topics = topics["reviews_summary"]["topic"]
        for rev_topic in rev_topics:
            topic = str(rev_topic["@name"])
            sentences = rev_topic["sentences"]["sentence"]
            if not isinstance(sentences, list):
                sen_id = int(dict(sentences)["from_review"])
                sen_type = str(dict(sentences)["sentence_type"])
                rows[sen_id] = rows[sen_id].strip() + "***" + topic + " " + sen_type + "\n"
            else:
                for sentence in sentences:
                    sen_id = int(sentence["from_review"])
                    sen_type = str(sentence["sentence_type"])
                    try:
                        rows[sen_id] = rows[sen_id].strip() + "***" + topic + " " + sen_type + "\n"
                    except IndexError:
                        print("Cannot find this index!")
                        continue
    except:
        traceback.print_exc()
    return rows

def sub_process(row):
    rows = row.split("\t")
    time_str = time.strftime("%Y%m%d_%H%M%S")
    convert_xml(rows, train_xml+time_str)
    run_surf(train_xml+time_str, train_surf+time_str)
    # new_data = get_label(rows, test_surf+time_str)

    try:
        Flag = True
        while Flag:
            try:
                fr_input = open(train_surf+time_str)    # test_surf + finpath
                Flag = False
            except IOError as e:
                print("IOError when running java!")
                if os.path.exists(train_xml+time_str):
                    print("xml file exists.")
                    modify_xml(train_xml+time_str)
                    output = run_surf(train_xml + time_str, train_surf + time_str)
                if os.path.exists(train_surf + time_str):
                    print("surf file exists")
                    fr_input = open(train_surf + time_str)
                    Flag = False
                time.sleep(10)
        line = fr_input.read()
        fr_input.close()
    except:
        traceback.print_exc()

    rows = get_label(line, rows)
    return rows

def split_data(data, n):
    # split data by n and save to file by its id
    def sub_save(lines, fn):
        fw = open(os.path.join(valid_dir, fn), "w")
        fw.writelines(lines)
        fw.close()

    sub_lines = []
    for idx, line in enumerate(data):
        if idx%n == 0 and idx!=0:
            print("build the %s dir"%(str(idx/n)))
            sub_lines.append(line)
            sub_save(sub_lines, str(idx/n))
            sub_lines = []
        else:
            sub_lines.append(line)
    sub_save(sub_lines, str(idx/n+1))

def sub_fn_process(fn):
    fr = open(os.path.join(train_dir, fn))
    lines = fr.readlines()
    fr.close()
    convert_xml(lines, os.path.join(train_xml, fn))
    run_surf(os.path.join(train_xml, fn), os.path.join(train_surf, fn))
    return

def get_fn_label(fn):
    xml_fr = open(os.path.join(train_xml, fn))
    xml_input = xml_fr.read()
    xml_fr.close()
    reviews = xmltodict.parse(xml_input)
    review_list = reviews["reviews_summary"]["reviews"]["review"]
    id_dict = {}
    for idx, review in enumerate(review_list):
        id_dict[review["@id"]] = idx


    fdata = open(os.path.join(train_dir, fn))
    lines = fdata.readlines()
    fdata.close()
    fr_input = open(os.path.join(train_surf, fn))
    surf_result = fr_input.read()
    fr_input.close()

    topics = xmltodict.parse(surf_result)
    # topics = BeautifulSoup(line, 'xml')
    rev_topics = topics["reviews_summary"]["topic"]
    for rev_topic in rev_topics:
        topic = str(rev_topic["@name"])
        sentences = rev_topic["sentences"]["sentence"]
        if not isinstance(sentences, list):
            sen_id = id_dict[dict(sentences)["from_review"]]
            sen_type = str(dict(sentences)["sentence_type"])
            lines[sen_id] = lines[sen_id].strip() + "***" + topic + " " + sen_type + "\n"
        else:
            for sentence in sentences:
                sen_id = id_dict[sentence["from_review"]]
                sen_type = str(sentence["sentence_type"])
                try:
                    lines[sen_id] = lines[sen_id].strip() + "***" + topic + " " + sen_type + "\n"
                except IndexError:
                    print("Cannot find this index!")
                    print(sen_id)
                    continue

    return lines



def main():
    # total_lines = get_data(train_file)
    # split data by 1000
    # split_data(total_lines, 1000)

    ### process surf results and save to file
    results = []
    sub_fn = sorted(os.listdir(train_surf), key=lambda x: int(x))
    for fn in sub_fn:
        print("process %s file"%fn)
        rows = get_fn_label(fn)
        results += rows
    fw = open(output_train, "w")
    fw.writelines(results)
    fw.close()



    ### conver each xml_input to surf results
    # pool = Pool(8)
    # block_num = 50
    # sub_fn = os.listdir(os.path.join(train_dir))
    # fnum = len(sub_fn)
    # block_size = fnum // block_num
    # for i in tqdm(range(block_num+1)):
    #     if i != block_num:
    #         block = sub_fn[i * block_size:(i + 1) * block_size]
    #     else:
    #         block = sub_fn[i * block_size:]
    #     tunnel = pool.amap(sub_fn_process, block)
    #     tunnel.get()


    ### read lines and then get xml and surf at the same time
    # f = lambda A, n=100: ["\t".join(A[i:i + n]) for i in range(0, len(A), n)]  # split dataset by n
    # pool = Pool(8)
    # block_num = 10
    # for i in range(242500, len(total_lines), 5000):
    #     for file in os.listdir("/research/lyu1/cygao/workspace/data/xml_input/"):
    #         os.remove(os.path.join("/research/lyu1/cygao/workspace/data/xml_input", file))
    #     for file in os.listdir("/research/lyu1/cygao/workspace/data/surf/"):
    #         os.remove(os.path.join("/research/lyu1/cygao/workspace/data/surf", file))
    #
    #
    #     start = i
    #     end = start + 5000
    #     if end <len(total_lines):
    #         lines = f(total_lines[start:end])
    #     else:
    #         lines = f(total_lines[start:])
    #
    #     block_size = len(lines) // block_num
    #
    #     fw = open(output_train, "a")
    #     for i in tqdm(range(block_num+1)):
    #         if i != block_num:
    #             block = lines[i * block_size:(i + 1) * block_size]
    #         else:
    #             block = lines[i * block_size:]
    #         tunnel = pool.amap(sub_process, block)
    #         output = tunnel.get()
    #         flat_output = list(itertools.chain(*output))
    #         fw.writelines(flat_output)
    #     fw.close()
    # pool.close()




    # fr_input = open("/research/lyu1/cygao/workspace/data/surf/test_20190111_215557")
    # line = fr_input.read()
    # fr_input.close()
    # rows = get_label(line, lines)
    # fw = open(output_test, "a")
    # fw.writelines(rows)
    # fw.close()

if __name__ == "__main__":
    main()

