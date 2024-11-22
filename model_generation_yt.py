import argparse
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

#import debugpy
#try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#    debugpy.listen(("localhost", 9501))
#    print("Waiting for debugger attach")
#    debugpy.wait_for_client()
#except Exception as e:
#    pass


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, prompt):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt = prompt

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["prompt"]
        
        qs = qs.replace('<image>','').strip()
        #yt
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        # qs ="<image>\n" + qs
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # self.model_config.image_aspect_ratio = 'pad_and_divide'
        # image_tensor = process_images([image], self.image_processor, self.model_config)
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, prompt=''):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, prompt)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    # rm_model_path = os.path.expanduser(args.rm_model_path)
    # rm_model_name = get_model_name_from_path(rm_model_path)

    # model_name = "llava"
    device_map = "auto"
    # rm_tokenizer, rm_model, rm_image_processor, _ = load_pretrained_model(rm_model_path, None, rm_model_name, device_map=device_map)
    # model.eval()
    
    model_path = os.path.expanduser(args.model_path)
    #yt
    model_name = "llava_qwen"
    device = "cuda"
    # model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map)
    model.eval()

    questions = []
    with open(args.question_file, 'r') as file:
        for line in file:
            questions.append(json.loads(line))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file_with_chunks = f"{args.answers_file.rsplit('.', 1)[0]}_chunk{args.chunk_idx}_of_{args.num_chunks}.jsonl"
    existing_ids = set()
    try:
        with open(answers_file_with_chunks, 'r') as file:
            for line in file:
                answer = json.loads(line)
                existing_ids.add(answer.get("id"))
    except FileNotFoundError:
        print(f"No existing file found: {answers_file_with_chunks}, proceeding with empty set of existing IDs.")

    questions = [q for q in questions if q.get("id") not in existing_ids]
    print(f'saving all the answers to {answers_file_with_chunks}')
    answers_file = os.path.expanduser(answers_file_with_chunks)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, prompt=args.test_prompt)

    index, cnt_images = 0, []
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["id"]
        cur_prompt = line["prompt"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        cnt_images.append(image_tensor.shape[0])
        output_texts = []

        for t in range(5):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
                
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            output_texts.append(outputs)
        
        index += 1
        ans_id = shortuuid.uuid()
        line['outputs'] = output_texts
        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling strategy follows simpo
    # nohup python tmp/model_generation.py --num-chunks 6 --chunk-idx 5 >> tmp/llava16_chunk-6-outof-6.log 2>&1 &
    # parser.add_argument("--rm-model-path", type=str, default="../model/alignment/llava16_llava_rlhf_reward_model_lr1e_5_bsz128_freevision_reward_model_coefficent")
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-critic-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./coco/train2017/train2017")
    parser.add_argument("--question-file", type=str, default="llava_7b_v1_preference.jsonl")
    parser.add_argument("--answers-file", type=str, default="llava_critic_7b_iter1.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    parser.add_argument("--use-qlora", type=bool, default=False)
    parser.add_argument("--qlora-path", type=str, default="")

    parser.add_argument(
        "--test-prompt",
        type=str,
        default="",
    )
    args = parser.parse_args()
    print(args)

    eval_model(args)
