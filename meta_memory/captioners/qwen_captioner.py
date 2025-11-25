from transformers import AutoModelForImageTextToText, AutoProcessor


class QwenCaptioner:

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        
        # default: Load the model on the available device(s)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map="auto"
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained(model_name)


    def video_caption(self, images_path: list[str], query: str):
        # Messages containing multiple images and a text query
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": [*images_path],
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

        return output_text
    