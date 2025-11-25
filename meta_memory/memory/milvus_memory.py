from dataclasses import dataclass, asdict

import datetime, time
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np
import os
import base64
import re
from heapq import heappop, heappush
import math
import matplotlib.pyplot as plt
from adjustText import adjust_text

from memory.memory import Memory, MemoryItem

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from openai import OpenAI


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970


class MilvusWrapper:
    def __init__(self, collection_name='test', ip_address='127.0.0.1', port=19530, drop_collection=False):
        self.collection_name = collection_name
        self.collection = self.connect_to_milvus_collection(collection_name, 1024, address=ip_address, port=port, drop_collection=drop_collection)

    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, address='127.0.0.1', port=19530, drop_collection=False):
        connections.connect(host=address, port=port)
        
        if drop_collection:
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', is_primary=True, auto_id=False, max_length=1000),
            FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
            FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, description='position of robot', dim=3),
            FieldSchema(name='theta', dtype=DataType.FLOAT, description='rotation of robot', dim=1),
            FieldSchema(name='caption', dtype=DataType.VARCHAR, description='caption string', max_length=3000),
            FieldSchema(name='images_path', dtype=DataType.VARCHAR, description='images path', max_length=200)
        ]
        schema = CollectionSchema(fields=fields, description='text image search')
        collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":1024}
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="position", index_params=index_params)

        return collection
    
    def insert(self, data_list):
        res = self.collection.insert(data_list)

    def search(self, data):

        self.collection.load()

        BATCH_SIZE = 2
        LIMIT = 10

        param = {
            "metric_type": "L2",
            "params": {
                "nprobe": 1024,
            }
        }

        res = self.collection.search(
            data=[data],
            anns_field="text_embedding",
            param=param,
            batch_size=BATCH_SIZE,
            limit=LIMIT,
            # expr="id > 3",
            output_fields=["id", "text_embedding"]
        )

        return res


class MilvusMemory(Memory):

    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530, time_offset=FIXED_SUBTRACT, save_path='', save_reasoning_path=''):

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset

        self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_api_base = os.environ.get("OPENAI_API_BASE")
        self.openai_client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_api_base)  
        self.save_visual_prompt_path = save_path
        self.save_reasoning_path = save_reasoning_path


    def insert(self, item: MemoryItem, text_embedding=None):

        memory_dict = asdict(item)
        memory_dict['id'] = str(time.time())

        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption'])

        memory_dict['text_embedding'] = text_embedding

        self.milv_wrapper.insert([memory_dict])


    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory


    def reset(self, drop_collection=True):

        if drop_collection:
            print("Resetting memory. We are dropping the current collection")

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)

        text_vector_db = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
        )
        self.text_retriever = text_vector_db.as_retriever(search_kwargs={"k": 5})

        self.position_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='position',
            text_field='caption',
        )
           

    def spatial_range_retrieval(self, query) -> str:
        print(f"Using spatial_range_retrieval tool. Input: {query}\n")

        question_pattern = r"question: (.*?),"
        position_pattern = r"position: [\[(]([^)\]]+)[\])]"
        range_pattern = r"search range: (\d+)"

        question_match = re.search(question_pattern, query)
        question = question_match.group(1)

        position_match = re.search(position_pattern, query)
        position_str = position_match.group(1).replace(' ', '')
        position = tuple(float(coord) for coord in position_str.split(','))

        range_match = re.search(range_pattern, query)
        search_range = int(range_match.group(1))

        docs = self._similarity_search_with_score_by_vector(self.position_vector_db, np.array(position).astype(float), k=max(int(search_range/2), 3))
        docs = self._memory_to_string(docs, out_img_path=True)

        messages = f"You are a five-star agent. Based on the given question and the information visible to the robot, please reason about the most likely image paths where the target object might appear. \
The given question is: {question}. The robot's observations are as follows: {docs}\n \
Output up to five best guesses in the following format, where position indicates the coordinates corresponding to each image path: 'Reasoning: <Explain why you chose these image paths and the corresponding positions for each image path.>, Image_paths: ['out/visual_prompt/25-05-30_16:18.png', 'out/visual_prompt/25-05-30_16:14.png', <...more image paths...>], Position: [(1, 1, 0), (2, 1, 4), <...more position...>]'. \
Here is an example of the output format: 'Reasoning: To determine the most likely locations where a statue in front of the Bill and Melinda Gates Computer Science building might be visible, I focused on observations that described outdoor settings near the building with clear views and fewer obstructive elements like walls or indoor settings. The questions suggest the statue would likely be located in a recognizable outdoor area possibly leading to or directly in front of the building. Locations with open plazas, pathways directly mentioning the building, or distinct outdoor features (such as stone tiles or plazas) near the building are ideal for discovering a statue. \
1. The image at position [-80.674, 203.412, -1.987] (./data/images/0/images/sampled_images_132.png) is significant because it describes an open plaza with a building bearing the words COMPUTER SCIENCE visible. Such a location is strategic for positioning a statue. \
2. The position [-79.618, 201.338, -2.053] (./data/images/0/images/sampled_images_131.png) shows a large, open courtyard, which aligns with typical locations where one would expect a statue, as courtyards often feature public artworks. \
3. The position [-76.845, 197.558, -2.031] (./data/images/0/images/sampled_images_129.png) provides a view of a sidewalk next to a glass building with a sign reading Bill & Melinda Gates Foundation, hinting at proximity to the main building and potential statue placement. \
4. The position [-74.092, 197.391, -1.883] (./data/images/0/images/sampled_images_128.png) focuses on a pathway with visible pathways and marble tiles, providing another area where a statue could logically be placed due to open visibility. \
Image_paths: ['./data/images/0/images/sampled_images_132.png', './data/images/0/images/sampled_images_131.png', './data/images/0/images/sampled_images_129.png', './data/images/0/images/sampled_images_128.png'], \
Position: [(-80.674, 203.412, -1.987), (-79.618, 201.338, -2.053), (-76.845, 197.558, -2.031), (-74.092, 197.391, -1.883)]'"
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                "role": "user",
                "content": messages,
            },
            ])
        chat_output = completion.choices[0].message.content
        print(f"chat_output: {chat_output}\n")

        image_paths_pattern = r"Image_paths:\s*\[([^\]]+)\]"
        position_pattern = r"Position:\s*\[([^\]]+)\]"
        image_paths_matches = re.findall(image_paths_pattern, chat_output)
        position_matches = re.findall(position_pattern, chat_output)

        image_paths_list = []
        if image_paths_matches:
            paths = image_paths_matches[0].strip().split(", ")
            image_paths_list = [path.strip("'") for path in paths]

        position_list = []
        if position_matches:
            positions = position_matches[0].strip("()").split("), (")
            for pos in positions:
                coords = pos.split(", ")
                coords = [float(coord) for coord in coords]
                position_list.append(tuple(coords))

        base64_images = [self._encode_image(image_path) for image_path in image_paths_list]
        response = self._gpt4v_QA(question, base64_images)

        out_string = ''
        for position, content in zip(position_list, response):
            out_string += f"The robot was at an average position of {position}. The robot saw the following: {content}\n\n"

        print(f"The output of the spatial_range_retrieval tool is: \n{out_string}\n")

        with open(self.save_reasoning_path, 'a') as f:
            f.write(f"Using spatial_range_retrieval tool. Input: {query}\n\n")
            f.write(f"chat_output: {chat_output}\n\n")
            f.write(f"image_paths_list: {image_paths_list}\n\n")
            f.write(f"position_list: {position_list}\n\n")
            f.write(f"response: {response}\n\n")
            f.write(f"The output of the spatial_range_retrieval tool is: {out_string}\n\n")

        return out_string
    

    def semantic_similarity_retrieval(self, query: str) -> str:
        print(f"Using semantic_similarity_retrieval tool. Input: {query}\n")
        docs = self.text_retriever.invoke(query)

        base64_images = [self._encode_image(doc.metadata['images_path']) for doc in docs]
        response = self._gpt4v_QA(query, base64_images)

        for doc, content in zip(docs, response):
            doc.page_content = content

        docs = self._memory_to_string(docs) 
        print(f"The output of the semantic_similarity_retrieval tool is: \n{docs}\n")

        with open(self.save_reasoning_path, 'a') as f:
            f.write(f"Using semantic_similarity_retrieval tool. Input: {query}\n\n")
            f.write(f"The output of the semantic_similarity_retrieval tool is: \n{docs}\n")

        return docs
    

    def memory_integration(self, query: str):
        print(f"Using memory_integration tool. Input: {query}\n")

        collection = Collection(name=self.db_collection_name)
        all_position = collection.query(expr='', limit=16384, output_fields=['position'])
        all_position = [tuple(item['position'][:2]) for item in all_position]
        path_graph = self._build_graph(all_position, 8.0)

        question, start_to_end, candidate_positions = self._parse_text(query)

        print(f"question: {question}")
        print(f"start_to_end: {start_to_end}")

        start_end_dicts = []
        candi_position = []
        text_prompt = ''
        agent_prompt = "You are an agent equipped with spatial perception capabilities. You will be provided with a two-dimensional map. \
On this map, the positive direction of the y-axis indicates North, while the positive direction of the x-axis indicates East. "
        if len(start_to_end)!=0:
            label_prompt_0 = ''
            for i, item in enumerate(start_to_end):
                start_end_name = []
                start_end_position = []
                for name, position in item.items():
                    start_end_name.append(name)
                    start_end_position.append(position[:2])
                    candi_position.append(position)

                print(f"start_end_position: {start_end_position}")
                path, distance = self._dijkstra(path_graph, start_end_position[0], start_end_position[1])
                print(f"path: {path}")

                start_end_dicts.append({start_end_name[0]: start_end_position[0], start_end_name[1]: start_end_position[1], 'path': path})

                if start_end_name[0]=="current position" and i==0:
                    label_prompt_0 += f'Label S0 represents your current position, with coordinates at {start_end_position[0]}. '
                elif start_end_name[0]!="current position":
                    label_prompt_0 += f'Label S{i} represents {start_end_name[0]}, with coordinates at {start_end_position[0]}. '

                label_prompt_0 += f'Label E{i} represents {start_end_name[1]}, with coordinates at {start_end_position[1]}. '
                label_prompt_0 += f'The distance between {start_end_position[0]} and {start_end_position[1]} is {distance} meters. '

            text_prompt += agent_prompt
            text_prompt += label_prompt_0
            text_prompt += 'The green dot indicates the path. '

        if len(candidate_positions)!=0:
            label_prompt_1 = ''
            for i, (name, position) in enumerate(candidate_positions.items()):
                label_prompt_1 += f'Label C{i} represents {name}, with coordinates at {position[:2]}. '
                candi_position.append(position)
            if len(start_to_end)!=0:
                text_prompt += label_prompt_1
            else:
                text_prompt += agent_prompt
                text_prompt += label_prompt_1

        print("##start_end_dicts##", start_end_dicts)
        print("##candidate_positions##", candidate_positions)
        visual_prompt_image_path = self._build_visual_prompt(start_end_dicts, candidate_positions, self.save_visual_prompt_path)
        print("##visual_prompt_image_path##", visual_prompt_image_path)

        text_prompt += f"Please answer the question: {question}. Your output format is as follows: \
'reasoning: <Carefully consider the given question and image, and input the reasons for selecting the coordinate points here. If you are uncertain of the exact answer, please provide your best estimate.>, position: (x, y)'"
        print("##text_prompt##", text_prompt)

        base64_images = [self._encode_image(visual_prompt_image_path)]
        response = self._gpt4o(text_prompt, base64_images)
        print("##response##", response)

        pattern = r"position: \(([-+]?\d+\.?\d*), ([-+]?\d+\.?\d*)\)"
        match = re.search(pattern, response)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
        else:
            raise ValueError("No valid position found in the input string")

        for coord in candi_position:
            if coord[0] == x and coord[1] == y:
                select_coord = list(coord)
                break 
        out_string = f"The question is: {question} Based on the previous observations, the answer is: The coordinates are {select_coord}."

        print(f"The output of the memory_integration tool is: \n{out_string}\n")

        with open(self.save_reasoning_path, 'a') as f:
            f.write(f"Using memory_integration tool. Input: {query}\n\n")
            f.write(f"visual_prompt_image_path: {visual_prompt_image_path}\n\n")
            f.write(f"text prompt: {text_prompt}\n\n")
            f.write(f"response: {response}\n\n")
            f.write(f"The output of the memory_integration tool is: \n{out_string}\n\n")


        return out_string


    ### Doc formatting for the last LLM
    def _memory_to_string(self, memory_list: list[MemoryItem], out_img_path=False):
        out_string = ""
        for doc in memory_list:
            s = f"The robot's observation at position {np.array(doc.metadata['position']).round(3).tolist()} is as follows: {doc.page_content} "
            if out_img_path: 
                s += f"The image path corresponding to this position is: {doc.metadata['images_path']}.\n\n "
            else:
                s += "\n\n"
            out_string += s
        return out_string


    # NOTE: This version of the code returns the vector
    def _similarity_search_with_score_by_vector(
            self,
            pos_db,
            embedding: List[float],
            k: int = 4,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
        ) -> List[Tuple[Document, float]]:
            
            if pos_db.col is None:
                print("No existing collection to search.")
                return []

            if param is None:
                param = pos_db.search_params

            # Determine result metadata fields with PK.
            output_fields = pos_db.fields[:]
            # output_fields.remove(pos_db._vector_field)
            timeout = pos_db.timeout or timeout
            # Perform the search.
            res = pos_db.col.search(
                data=[embedding],
                anns_field=pos_db._vector_field,
                param=param,
                limit=k,
                expr=expr,
                output_fields=output_fields,
                timeout=timeout,
                **kwargs,
            )
            # Organize results.
            ret = []
            for result in res[0]:
                data = {x: result.entity.get(x) for x in output_fields}
                doc = pos_db._parse_document(data)
                pair = (doc, result.score)
                ret.append(pair)

            return [doc for doc, _ in ret]
    

    def _gpt4o(self, query: str, base64_imgs: list):
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}} for image_base64 in base64_imgs[:]],
                    ],
                }
            ],
            max_tokens=1024,
        )
        response = response.choices[0].message.content

        return response
    

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def _gpt4v_QA(self, query: str, base64_images: list): 

        query = f"Given a query and several images, where each image comprises four consecutive video frames depicting a scene, please provide a description of each image. \
Pay particular attention to the objects, text, environmental features, and any other notable details. Additionally, carefully determine whether the specific object mentioned in the query is present in each image. \
Query: {query}. Formatting Rules: 1. Do not provide a summary description for all the images. Output the description of each image directly without numbering. \
2. Separate the description of each image with a line break."

        response = self._gpt4o(query, base64_images)
        response = re.split(r'\n+', response.strip())

        return response


    def _euclidean_distance(self, point_a, point_b):
        return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


    def _build_graph(self, nodes, threshold):
        graph = {}
        for i, node in enumerate(nodes):
            graph[node] = {}
            for j, other_node in enumerate(nodes):
                if i != j and self._euclidean_distance(node, other_node) <= threshold:
                    graph[node][other_node] = self._euclidean_distance(node, other_node)
        return graph

    def _add_node_to_graph(self, graph, new_node):
        nearest_node = None
        min_distance = float('inf')
        for node in graph:
            distance = self._euclidean_distance(node, new_node)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        graph[new_node] = {nearest_node: self._euclidean_distance(new_node, nearest_node)}
        graph[nearest_node][new_node] = self._euclidean_distance(nearest_node, new_node)

    def _dijkstra(self, graph, start_node, end_node):
        if start_node not in graph:
            self._add_node_to_graph(graph, start_node)
        if end_node not in graph:
            self._add_node_to_graph(graph, end_node)

        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0
        previous_nodes = {node: None for node in graph}
        priority_queue = [(0, start_node)]

        while priority_queue:
            current_distance, current_node = heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heappush(priority_queue, (distance, neighbor))

        path = []
        current_node = end_node
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()

        return path, distances[end_node]


    def _parse_text(self, text: str):
        question_match = re.search(r'question:\s*(.*?)(?=\s*(?:,?\s*start_to_end:|$))', text, re.DOTALL)
        question = question_match.group(1).strip() if question_match else None

        # Extract start_to_end paths
        start_to_end = []
        start_end_section = re.search(r'start_to_end: (.*?)(?=, candidate position:|$)', text, re.DOTALL)
        if start_end_section:
            paths = re.finditer(r'\[(.*?)\]', start_end_section.group(1))
            for path in paths:
                items = re.finditer(r'([^:,\[\]]+):\s*\(([^)]+)\)', path.group(1))
                path_dict = {}
                for item in items:
                    key = item.group(1).strip()
                    coords = tuple(map(float, re.split(r',\s*', item.group(2))))
                    path_dict[key] = coords
                if path_dict:
                    start_to_end.append(path_dict)

        # Extract candidate positions
        candidate_positions = {}
        candidate_section = re.search(r'candidate position:\s*\[(.*?)\]', text, re.DOTALL)
        if candidate_section:
            items = re.finditer(r'([^:,\[\]]+):\s*\(([^)]+)\)', candidate_section.group(1))
            for item in items:
                key = item.group(1).strip()
                coords = tuple(map(float, re.split(r',\s*', item.group(2))))
                candidate_positions[key] = coords

        return question, start_to_end, candidate_positions


    def _build_visual_prompt(self, start_to_end: list[dict], candidate_position: dict, save_path: str):
        plt.figure(figsize=(10, 8))

        all_x = []
        all_y = []
        texts = []  

        def check_and_adjust_position(x, y, existing_positions, margin=2.0):
            """Check if the position overlaps with any existing positions and adjust if necessary."""
            adjusted_x, adjusted_y = x, y
            while any(abs(adjusted_x - ex) < margin and abs(adjusted_y - ey) < margin for ex, ey in existing_positions):
                adjusted_x += margin / 2
                adjusted_y += margin / 2
            existing_positions.append((adjusted_x, adjusted_y))
            return adjusted_x, adjusted_y

        existing_positions = []

        if len(start_to_end) != 0:
            for i, item in enumerate(start_to_end):
                path = item['path']  
                
                path_xy = [(x, y) for x, y in path]
                
                all_x.extend([p[0] for p in path_xy])
                all_y.extend([p[1] for p in path_xy])

                seen_x = set()
                unique_full_path = []
                for x, y in sorted(path_xy, key=lambda p: p[0]):
                    if x not in seen_x:
                        seen_x.add(x)
                        unique_full_path.append((x, y))

                for (x, y) in unique_full_path:
                    plt.scatter(x, y, color='g', s=20, alpha=0.7)

            for i, item in enumerate(start_to_end):
                item_keys = list(item.keys())
                cp_x, cp_y = item[item_keys[0]]
                el_x, el_y = item[item_keys[1]]

                if item_keys[0] == "current position" and i == 0:
                    plt.scatter(cp_x, cp_y, color='r', s=80)
                    texts.append(plt.text(cp_x, cp_y, f'S{i}', color='black', ha='right', va='bottom', fontsize=10))
                elif item_keys[0] != "current position":
                    cp_x, cp_y = check_and_adjust_position(cp_x, cp_y, existing_positions)
                    plt.scatter(cp_x, cp_y, color='r', s=80)
                    texts.append(plt.text(cp_x, cp_y, f'S{i}', color='black', ha='right', va='bottom', fontsize=10))
                
                el_x, el_y = check_and_adjust_position(el_x, el_y, existing_positions)
                plt.scatter(el_x, el_y, color='b', s=80)
                texts.append(plt.text(el_x, el_y, f'E{i}', color='black', ha='left', va='bottom', fontsize=10))

        if len(candidate_position) != 0:
            for i, (vm_name, (vm_x, vm_y, _)) in enumerate(candidate_position.items()):
                vm_x, vm_y = check_and_adjust_position(vm_x, vm_y, existing_positions)
                plt.scatter(vm_x, vm_y, color='y', s=80)
                texts.append(plt.text(vm_x, vm_y, f'C{i}', color='black', ha='right', va='top', fontsize=10))
                all_x.append(vm_x)
                all_y.append(vm_y)

        margin_x = (max(all_x) - min(all_x)) * 0.2
        margin_y = (max(all_y) - min(all_y)) * 0.2
        plt.xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        plt.ylim(min(all_y) - margin_y, max(all_y) + margin_y)

        plt.xlabel('X Axis', fontsize=12)
        plt.ylabel('Y Axis', fontsize=12)

        adjust_text(texts)

        plt.grid(True, linestyle='--', alpha=0.5)

        import time
        current_time = time.localtime()
        formatted_time = time.strftime('%y-%m-%d_%H:%M:%S', current_time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_path = os.path.join(save_path, formatted_time+".png") 
        plt.savefig(img_path, dpi=300, bbox_inches='tight')

        plt.close()

        return img_path