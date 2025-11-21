import os

import websocket
import json
import uuid
import urllib.parse
import urllib.request
from typing import Dict, Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class ComfyUISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="comfyui_", env_file='.env', extra='ignore')

    host: str = "localhost"
    port: int = 8188
    authentication: str | None = None
    workflow_dir: str = "workflows"


class ComfyUI:
    def __init__(self):
        self.settings = ComfyUISettings()
        self.http_url = f'http://{self.settings.host}:{self.settings.port}'
        self.client_id = str(uuid.uuid4())
        self.ws_url = f'ws://{self.settings.host}:{self.settings.port}/ws?clientId={self.client_id}'
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.settings.authentication:
            self.headers["Authorization"] = self.settings.authentication

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        url = f"{self.http_url}/view?{url_values}"
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req) as response:
            return response.read()

    def get_history(self, prompt_id):
        url = f"{self.http_url}/history/{prompt_id}"
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    
    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"{self.http_url}/prompt", headers=self.headers, data=data)
        with urllib.request.urlopen(req) as resp:
            resp_data = resp.read()
            resp_json = json.loads(resp_data)
            return resp_json

    async def process_workflow(self, workflow: Any, params: Dict[str, Any]):
        if isinstance(workflow, str):
            workflow_path = os.path.join(self.settings.workflow_dir, f"{workflow}.json")
            if not os.path.exists(workflow_path):
                raise Exception(f"Workflow {workflow} not found")
            with open(workflow_path, "r", encoding='utf-8') as f:
                prompt = json.load(f)
        else:
            prompt = workflow

        self.update_workflow_params(prompt, params)

        ws = websocket.WebSocket()
        if self.settings.authentication:
            ws.connect(self.ws_url, header=[f"Authorization: {self.settings.authentication}"])
        else:
            ws.connect(self.ws_url)

        try:
            images = self.get_images(ws, prompt)
            return images
        finally:
            ws.close()

    def get_images(self, ws, prompt):
        prompt_id = self.queue_prompt(prompt)["prompt_id"]
        output_images = {}
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break
            else:
                continue

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            if "images" in node_output:
                output_images[node_id] = [
                    self.get_image(image["filename"], image["subfolder"], image["type"])
                    for image in node_output["images"]
                ]

        return output_images

    def update_workflow_params(self, prompt, params):
        if not params:
            return

        for node in prompt.values():
            node_title = node.get("_meta", {}).get("title", "")
            
            # Handle positive prompt - support both new and old parameter names
            if node_title == "Positive Prompt":
                if "positive_prompt" in params:
                    if isinstance(node["inputs"]["text"], str):
                        node["inputs"]["text"] = params["positive_prompt"]
                elif "prompt" in params:
                    # Backward compatibility
                    if isinstance(node["inputs"]["text"], str):
                        node["inputs"]["text"] = params["prompt"]
                elif "text" in params:
                    # Backward compatibility
                    if isinstance(node["inputs"]["text"], str):
                        node["inputs"]["text"] = params["text"]
            # Handle negative prompt
            elif node_title == "Negative Prompt" and "negative_prompt" in params:
                if isinstance(node["inputs"]["text"], str):
                    node["inputs"]["text"] = params["negative_prompt"]
            # Handle sampler parameters
            elif node_title == "Sampler":
                if "seed" in params:
                    node["inputs"]["seed"] = params["seed"]
                if "steps" in params:
                    node["inputs"]["steps"] = params["steps"]
                if "cfg" in params:
                    node["inputs"]["cfg"] = params["cfg"]
                if "denoise" in params:
                    node["inputs"]["denoise"] = params["denoise"]
            # Handle image loading
            elif node_title == "Load Image" and "image" in params:
                node["inputs"]["image"] = params["image"]
            # Handle empty latent image (resolution)
            elif node_title == "Empty Latent Image":
                if "width" in params:
                    node["inputs"]["width"] = params["width"]
                if "height" in params:
                    node["inputs"]["height"] = params["height"]
