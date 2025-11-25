# Meta-Memory
> <b>Meta-Memory: Retrieving and Integrating Semantic-Spatial Memories for Robot Spatial Reasoning</b> <br>
> Yufan Mao, Hanjing Ye, Wenlong Dong, Chengjie Zhang, and Hong Zhang <br>
> [<u>Project page</u>](https://itsbaymax.github.io/meta-memory.github.io/)

## Setup

1. Install Python dependencies

    ```
    conda create -n metamemory python=3.10 -y
    conda activate metamemory
    python -m pip install -r requirements.txt
    ```

2. Set the OpenAI API key in environment variables

    ```
    export OPENAI_API_BASE=...
    export OPENAI_API_KEY=...
    ```

3. Install MilvusDB

    ```
    curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o launch_milvus_container.sh
    ```

        > `docker` must be installed on the system to easily use Milvus by simply running the command below. This script will automatically launch MilvusDB on a docker container. Otherwise, the user must install MilvusDB from scratch themselves

        ```
        bash launch_milvus_container.sh start
        ```

## Dataset
You can download SpaceLocQA dataset from [Hugging Face](https://huggingface.co/datasets/Baymax-12/SpaceLocQA/resolve/main/SpaceLocQA.zip?download=true). 
After downloading, please unzip the archive and organize the dataset into the following directory structure:

```
|-- data
    |-- spacelocqa
        |-- 0
            |-- spacelocqa_unfilled.json
            |-- camera
            |-- odometry.txt
        |-- 1
        ......
        |-- spacelocqa.csv

```

## Usage

1. Generate captions

    ```
    bash meta_memory/scripts/bash_scripts/preprocess_captions_all.sh
    ```
2. Generate questions

    ```
    python meta_memory/scripts/question_scripts/form_question_jsons.py
    ```
3. Run the evaluation

    ```
    bash meta_memory/scripts/bash_scripts/run_all_evals.sh
    ```
4. Real-world Deployment
    - First, you need to use a camera to capture images and employ SLAM to obtain the pose corresponding to these images.
    - Next, process the metadata using preprocess_caption.py to generate captions.
    - Finally, refer to eval_real_world.py to deploy Meta-Memory on a robot.


## Acknowledgement
- Thanks to these great repositories: [remembr](https://github.com/NVIDIA-AI-IOT/remembr/tree/main), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file), [MilvusDB](https://github.com/milvus-io/milvus) and many other inspiring works in the community.


