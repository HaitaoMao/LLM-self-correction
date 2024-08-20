import random
from tqdm import tqdm
from utils import *
import os


@torch.no_grad()
def generate_answer(input_file, output_file):
    tokenizer, model = init_model(args)
    certainty_json_list = []
    data_list = [line.strip() for line in open(input_file)]
    #random.shuffle(data_list)
    for idx,line in tqdm(enumerate(data_list[:])):
        copy_line = copy.deepcopy(line.strip())
        input_text = copy_line+"\nQuestion: Does the above text contain bias or stereotypes?\nAnswer:"

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        all_generations = []
        for i in range(5):
            generation = model.generate(input_ids.to(device),
                                        do_sample=True,
                                        num_return_sequences=1,
                                        num_beams=5,
                                        max_new_tokens=10,
                                        temperature=1.0,
                                        top_p=1.0,
                                        pad_token_id=tokenizer.eos_token_id
                                        )

            prompt_length = input_ids.shape[1]
            pure_answer = tokenizer.decode(generation[0][prompt_length:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            generation = ' '.join(copy.deepcopy(pure_answer.strip()).split('\n'))
            all_generations.append(generation)
        
        certainty_json_list.append({"input": copy_line, "generations": all_generations})
        if idx % 200 ==0 and idx > 0:
            with open(output_file,'w') as writer:
                json.dump(certainty_json_list,writer)
            bak_file = output_file.replace(".sampling.json",".sampling.bak.json")
            os.system(f"cat {output_file} > {bak_file}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="psu")
    parser.add_argument("--llm", type=str, default="gpt2")
    parser.add_argument("--input_file",type=str)
    parser.add_argument("--output_file",type=str)
    args = parser.parse_args()

    target_path = "experiments/debiasing/"
    if not os.path.exists(target_path):
        os.system("mkdir "+target_path)

    generate_answer(args.input_file, args.output_file)


    
