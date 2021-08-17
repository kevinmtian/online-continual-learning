# mtian_dev test train ODI
# nohup python -u general_main.py --data mini_imagenet --cl_type ni --agent ER --retrieve MIR --update random --mem_size 5000&>train.out&

python general_main.py --data mini_imagenet --cl_type ni --agent ER --retrieve MIR --update random --mem_size 5000

