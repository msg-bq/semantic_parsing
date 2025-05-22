import argparse
import json

from generate_individual_func.from_gpt import call_openai_threads

parser = argparse.ArgumentParser(description="Natural Statements Generation")
parser.add_argument("--input_file_path",type=str,help="Path of the input file with formals.")
parser.add_argument("--output_file_path",type=str,help="Path of the output file.")
parser.add_argument("--area",type=str,choices=['financial','weather'],help="Choose the area of the formals.")
args = parser.parse_args()


if __name__ == "__main__":
    f = open(args.input_file_path,'r',encoding='utf-8')
    if args.area == 'financial':
        datas = f.readlines()
        res = call_openai_threads.generate_from_gpt(datas,1)
    else:
        datas = json.load(f)
        res = call_openai_threads.generate_from_gpt(datas,2)

    final = open(args.output_file_path,'a',encoding='utf-8')
    for r in res:
        json.dump(res,final,ensure_ascii=False)
        f.write('\n')


