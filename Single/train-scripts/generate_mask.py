import os
from tqdm import tqdm
import torch

def gradients_to_mask(gradients, threshold=0.5):
    hard_dict = {}
    for key, tensor in gradients.items():
        tensor = tensor.to('cuda')
        abs_tensor = torch.abs(tensor).flatten()
        num_elements = abs_tensor.numel()
        threshold_index = int(num_elements * threshold)
        positions = torch.argsort(abs_tensor)
        ranks = torch.argsort(positions)
        threshold_tensor = torch.zeros_like(ranks, dtype=torch.bool)
        threshold_tensor[ranks < threshold_index] = True
        hard_dict[key] = threshold_tensor.reshape(tensor.shape).cpu()
    torch.cuda.empty_cache()
    return hard_dict

def multi_tensors(tensors1, tensors2):
    result = []
    device = 'cuda'
    for t1, t2 in zip(tensors1, tensors2):
        result.append((t1.to(device) * t2.to(device)).cpu())
    return result

def save_tensors_as_pt(tensors, keys, save_path):
    tensor_dict = {key: tensor for key, tensor in zip(keys, tensors)}
    torch.save(tensor_dict, save_path)

def main():
    figure_name = 'i2p_5prompt_5seed_full_0.01'
    threshold = 0.5
    base_dir = f'../gradient/{figure_name}'

    all_tensors = []
    keys = None

    for root, dirs, files in os.walk(base_dir):
        for file in tqdm(files):
            if file.endswith(".pt"):
                file_path = os.path.join(root, file)
                gradients = torch.load(file_path)
                if keys is None:
                    keys = list(gradients.keys())

                hard_dict = gradients_to_mask(gradients, threshold=threshold)
                tensor_list = [hard_dict[k] for k in keys]
                all_tensors.append(tensor_list)

    if not all_tensors:
        print("Failed to find any valid mask")
        return

    print(f"Totally processed {len(all_tensors)} mask, now start to compute the intersection...")
    result_tensors = all_tensors[0]
    #count_one(result_tensors, keys, threshold, 0, figure_name)
    for i in tqdm(range(1, len(all_tensors))):
        result_tensors = multi_tensors(result_tensors, all_tensors[i])

    save_path = f"../count_one/final_mask/{figure_name}_{threshold}_intersection.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_tensors_as_pt(result_tensors, keys, save_path)
    print(f"Results saved to:{save_path}")

if __name__ == "__main__":
    main()
