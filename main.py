import os

def main(expirments:str):
    """Run the expirments
    
    Args:
        expirments: expirments name
    """
    if expirments == 'compare_data_sim':
        os.system('python experiments/compare_data_similarity_for_segmentation.py')
    elif expirments == 'compare_data_size_ft':
        os.system('python experiments/compare_pretrained_model_finetuning_sizes.py')
    elif expirments == 'compare_baseline':
        os.system('python experiments/compare_pretrained_with_baseline.py')
    else:
        raise ValueError(f'unknown expirments : {expirments}')


if __name__ == '__main__':
    main(expirments='compare_data_sim_aug')