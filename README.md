# AI506_TP

To download dependencies and activate conda environment:

    conda env create -f env.yml
    conda activate ai506tp

To generate Node2Vec keyed vectors:

    python gen_n2v_kv.py

To generate HyperNode2Vec keyed vectors:
    
    python gen_kv.py

To train classifier1 and evaluate:

    python main.py
    
To train classifier2 and evaluate:
    
    python classify.py
    (change filename of keyed vector and directory...)