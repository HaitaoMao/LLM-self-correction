import os
for i in range(8):  
    os.system(f"CUDA_VISIBLE_DEVICES={i} nohup  python -u gpt2_generate.py --input_file datasets/debiasing/{i}.txt --output_file experiments/debiasing/debiasing.{i}.sampling.json --cluster voodoo > experiments/debiasing/sampling.{i}.log 3>&1&")
    
