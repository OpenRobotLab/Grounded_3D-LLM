export PYTHONPATH="./"
python models/metrics/evaluate_LLM.py\
        --directory_path="$1"\
        --statistics=true\
        --test_scanrefer=true\
        --test_m3drefer=true\
        --test_lan=true\
        --test_detection=true
