import os
import requests
import folder_paths
import json
import base64
import io
from PIL import Image
import comfy.utils
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURATION ---
ENABLE_REMOTE_MODELS = True  # Set to False to use only the hardcoded list
# ---------------------

with open(os.path.join(script_dir, "API-key.txt"), "r") as file:
    API_key = file.read().strip()

FALLBACK_MODELS = [
    "pixtral-large-latest",
    "pixtral-12b-latest",
    "ministral-3b-latest",
    "ministral-8b-latest",
    "ministral-14b-latest",
    "open-mistral-nemo",
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
]


def get_mistral_models():
    if not ENABLE_REMOTE_MODELS:
        return FALLBACK_MODELS

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_key}",
    }

    try:
        response = requests.get(
            "https://api.mistral.ai/v1/models", headers=headers, timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            model_list = []
            for model in data.get("data", []):
                model_name = model["id"]
                if model.get("capabilities", {}).get("vision", False):
                    model_name += " 🖼️"
                model_list.append(model_name)

            return sorted(model_list)

    except Exception as e:
        print(
            f"MistralAPI Node: Failed to fetch models from API, using fallback. Error: {e}"
        )

    return FALLBACK_MODELS


def _add_prompts_folder_path():
    prompts_folder_name = "prompts"
    prompts_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), prompts_folder_name
    )

    folders, extensions = folder_paths.folder_names_and_paths.get(
        prompts_folder_name, ([], set())
    )

    if prompts_dir not in folders:
        folders.append(prompts_dir)

    extensions.add(".json")

    folder_paths.folder_names_and_paths[prompts_folder_name] = (folders, extensions)


_add_prompts_folder_path()  # Call the function to register the path


class InvokeMistralEndpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (get_mistral_models(), {}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "max_tokens": ("INT", {"default": 512}),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0, "min": -2.0, "max": 2.0, "step": 0.1},
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0, "min": -2.0, "max": 2.0, "step": 0.1},
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Your response is directly used as a prompt: avoid quotation marks, brackets, etc! Focus on visually percievable descriptors, not touchy feely fluff. Give me a single SD prompt for the following:\n\n",
                        "multiline": True,
                        "placeholder": "There is no hidden prompt injection.",
                    },
                ),
            },
            "optional": {
                "context": (
                    "STRING",
                    {
                        "multiline": True,
                        "defaultInput": True,
                        "tooltip": "Use Mistral AI prompt loader node to supply the context for few-shot prompting.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {"tooltip": "Only Pixtral models support image input."},
                ),
                "random_seed": (
                    "INT",
                    {
                        "defaultInput": True,
                        "tooltip": "If you don't supply a fixed seed, it's randomized on the server.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat_complete"
    CATEGORY = "utils/text"
    DESCRIPTION = (
        "Makes a single call to the Mistral AI API. Set the API key in API-key.txt"
    )

    def prepare_image_for_mistral(self, image):
        samples = image.movedim(-1, 1)
        height = samples.shape[2]
        width = samples.shape[3]

        if height > 1024 or width > 1024:
            if height > width:
                width = round(width * 1024 / height)
                height = 1024
            else:
                height = round(height * 1024 / width)
                width = 1024
            # comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
            samples = comfy.utils.common_upscale(
                samples, width, height, "bilinear", "disabled"
            )
        samples = samples.movedim(1, -1)

        i = 255.0 * samples.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        with io.BytesIO() as output_bytes:
            img.save(output_bytes, format="JPEG")
            bytes_data = output_bytes.getvalue()

        base64_str = base64.b64encode(bytes_data).decode("utf-8")
        return base64_str

    def chat_complete(
        self,
        model,
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        prompt,
        random_seed=None,
        context=None,
        image=None,
    ):

        target_model = model.split()[0]  # Remove the emoji and whitespace

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_key}",  # Using the API key from the file
        }

        messages = []

        if context:
            try:
                context_messages = json.loads(context)
                if isinstance(context_messages, list):
                    messages.extend(context_messages)
                else:
                    print("Warning: Context input is not a valid JSON list.")
            except json.JSONDecodeError:
                print("Error: Context input is not valid JSON.")

        if image is not None:
            base64_image = self.prepare_image_for_mistral(image)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        data = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        if random_seed is not None:
            data["random_seed"] = random_seed

        try:
            # print(f"Request Data: {data}")  # Print the request data
            print(f"Mistral AI API request sent.")
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions", headers=headers, json=data
            )
            response.raise_for_status()
            # print(f"Raw Response: {response.text}")  # Print the raw response
            print(f"Mistral AI API response received.")
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return (content,)

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            raise


class LoadFewShotPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_file": (folder_paths.get_filename_list("prompts"),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("few_shot_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "utils/text"
    DESCRIPTION = "Loads a JSON file with the Messages."

    def load_prompt(self, prompt_file):
        prompts_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prompts"
        )
        file_path = os.path.join(prompts_dir, prompt_file)

        try:
            with open(file_path, "r") as f:
                content = f.read()
            return (content,)
        except Exception as e:
            print(f"Error loading prompt file: {e}")
            return ("",)


NODE_CLASS_MAPPINGS = {
    # API
    "InvokeMistralEndpoint": InvokeMistralEndpoint,
    # loader
    "LoadFewShotPrompt": LoadFewShotPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # API
    "InvokeMistralEndpoint": "Mistral AI completion",
    # loader
    "LoadFewShotPrompt": "Mistral AI prompt loader",
}
