import ipdb
import re

def MultistepAnalysis4Hallucination(file,round):
    correct, sum_examples = 0, 0
    answer_line, label_line = "", ""
    response_list = []
    idx = 0
    fail_idx_list = []
    
    with open(file) as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        if line.strip() not in ["", "\n", "\t", " "]: response_list.append(line.strip())
        if line.strip().startswith("LABEL:"):
            label_line = line.strip().replace("LABEL:", "").strip().lower()
            answer_line = response_list[-2]

            # print(answer_line)
        if "#ROUND#" in line and str(round) in line:
            # print(line)
            # ipdb.set_trace()
            if not (answer_line.startswith("(a)") or  answer_line.startswith("(b)") or  answer_line.startswith("(c)")):
                print(answer_line)
                fail_idx_list.append(idx)
                # print("something wrong\t",answer_line)
                # raise ValueError

            sum_examples += 1

            if "major" in label_line and "major" in answer_line:
                correct += 1
            elif "minior" in label_line and "minor" in answer_line:
                correct += 1

            elif ("major" not in label_line and "major" not in answer_line) or ("minor" not in label_line and "minor" not in answer_line):
                correct += 1
            answer_line, label_line = "", ""
            response_list = []

    '''            
    for fail_idx in fail_idx_list:
        print(lines[fail_idx:fail_idx+4])
    '''
    # ipdb.set_trace()
    print(correct / (sum_examples))
    print(f"wrong {len(fail_idx_list)}")
    

def MultistepAnalysis4Hallucination_new(file,round):
    correct, sum_examples = 0, 0
    answer_line, label_line = "", ""
    response_list = []
    idx = 0
    fail_idx_list = []
    
    with open(file) as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        if line.strip() not in ["", "\n", "\t", " "]: response_list.append(line.strip())
        if line.strip().startswith("LABEL:"):
            label_line = line.strip().replace("LABEL:", "").strip().lower()
            try:
                answer_line = response_list[-2]
            except:
                # ipdb.set_trace()
                answer_line = " "
                

            # print(answer_line)
        if "#ROUND#" in line and str(round) in line:
            # print(line)
            # ipdb.set_trace()
            if not (answer_line.startswith("(a)") or  answer_line.startswith("(b)") or  answer_line.startswith("(c)")):
                print(answer_line)
                fail_idx_list.append(idx)
                # print("something wrong\t",answer_line)
                # raise ValueError

            sum_examples += 1
            # print(label_line == "accurate")
            # print(" accurate" in answer_line)
            # print(label_line, answer_line)
            if "inaccurate" in label_line and "inaccurate" in answer_line:
                correct += 1
            elif "accurate" == label_line and " accurate" in answer_line:
                correct += 1
            # if "major" in label_line and "major" in answer_line:
            # elif "minior" in label_line and "minor" in answer_line:
            #     correct += 1
            # elif ("major" not in label_line and "major" not in answer_line) or ("minor" not in label_line and "minor" not in answer_line):
            #     correct += 1
            answer_line, label_line = "", ""
            response_list = []

    '''            
    for fail_idx in fail_idx_list:
        print(lines[fail_idx:fail_idx+4])
    '''
    # ipdb.set_trace()
    print(correct / (sum_examples))
    print(f"wrong {len(fail_idx_list)}")


    
def fine_grain_Analysis4Hallucination(file,round):
    correct, sum_examples = 0, 0
    answer_line, label_line = "", ""
    response_list = []
    cnt = 0
    idx = 0
    fail_idx_list = []
    
    results = []
    for _ in range(round+1):
        results.append([])
        
    with open(file) as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        if line.strip() not in ["", "\n", "\t", " "]: response_list.append(line.strip())
        if line.strip().startswith("LABEL:"):
            label_line = line.strip().replace("LABEL:", "").strip().lower()
            answer_line = response_list[-2]

            # print(answer_line)
        if "#ROUND#" in line:
            # and str(round) in line
            extracts = re.findall(r'\d+', line)
            round = [int(i) for i in extracts][0]
            # ipdb.set_trace()
            
            if not (answer_line.startswith("(a)") or  answer_line.startswith("(b)") or  answer_line.startswith("(c)")):
                cnt += 1
                print(answer_line)
                fail_idx_list.append(idx)
                # print("something wrong\t",answer_line)
                # raise ValueError

            sum_examples += 1

            if "major" in label_line and "major" in answer_line:
                correct += 1
            elif "minior" in label_line and "minor" in answer_line:
                correct += 1

            elif ("major" not in label_line and "major" not in answer_line) or ("minor" not in label_line and "minor" not in answer_line):
                correct += 1
            answer_line, label_line = "", ""
            response_list = []

    '''            
    for fail_idx in fail_idx_list:
        print(lines[fail_idx:fail_idx+4])
    '''
    print(correct / (sum_examples))
    print(f"wrong {cnt}")


MultistepAnalysis4Hallucination_new("preliminary/llama3.hallucination.log",0)


# MultistepAnalysis4Hallucination("preliminary/mistral.hallucination.log",1)
# fine_grain_Analysis4Hallucination("preliminary/mistral.hallucination.log",1)
# MultistepAnalysis4Hallucination_new("preliminary/mistral.hallucination.log",2)
# MultistepAnalysis4Hallucination("preliminary/mistral.hallucination_10_25_2.log",1)

# mistral.hallucination_10_25_2