import os
import logging

import websocket
import json
import uuid
import urllib.parse
import urllib.request
import io
import base64
from typing import Dict, Any

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ComfyUISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="comfyui_", env_file='.env', extra='ignore')

    host: str = "localhost"
    port: int = 8188
    authentication: str | None = None
    workflow_dir: str = "workflows"
    upload_subfolder: str = "uploads"


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
        
        # Log connection information
        logger.info(f"Connecting to ComfyUI at HTTP: {self.http_url}, WebSocket: {self.ws_url}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if the ComfyUI server is reachable."""
        # Try to access the system_stats endpoint to test connectivity
        test_url = f"{self.http_url}/system_stats"
        req = urllib.request.Request(test_url, headers=self.headers)
        
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                logger.info(f"Successfully connected to ComfyUI at {self.http_url}")
                return
        except urllib.error.HTTPError as e:
            # HTTP error means server is reachable, but endpoint might not exist
            # Try root endpoint as fallback
            if e.code in (404, 405):
                try:
                    root_req = urllib.request.Request(self.http_url, headers=self.headers)
                    with urllib.request.urlopen(root_req, timeout=5):
                        logger.info(f"Successfully connected to ComfyUI at {self.http_url}")
                        return
                except Exception:
                    pass
            # Server is reachable but returned an error
            logger.warning(f"ComfyUI server at {self.http_url} is reachable but returned HTTP {e.code}: {e.reason}")
            return
        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to ComfyUI at {self.http_url}: {e.reason}")
            return
        except Exception as e:
            logger.error(f"Unexpected error while testing connection to ComfyUI at {self.http_url}: {str(e)}")
            return

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
    
    def upload_image_from_content(self, image_content) -> Dict[str, Any]:
        """Upload an image from MCP ImageContent to the ComfyUI server.
        
        Args:
            image_content: MCP ImageContent object containing the image data.
            
        Returns:
            Dictionary containing the upload result. The 'name' field contains the actual filename
            used by ComfyUI (which may differ from the requested name if there was a conflict).
            This name should be used when referencing the image in workflows.
        """
        # Extract image data from ImageContent/Image object
        image_data = image_content.data
        
        # Handle different data formats
        if isinstance(image_data, bytes):
            # FastMCP Image.data contains raw image bytes (as created in extract_images)
            # Use the bytes directly as they are already raw image data
            image_bytes = image_data
        elif isinstance(image_data, str):
            # Handle data URI format (data:image/png;base64,...)
            if image_data.startswith("data:"):
                # Extract base64 part from data URI
                base64_data = image_data.split(",", 1)[1]
            else:
                # Assume it's already base64
                base64_data = image_data
            
            # Strip whitespace and newlines from base64 string
            base64_data = base64_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')
            
            # Add padding if needed (base64 strings must be multiple of 4)
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)
            
            # Decode base64 to bytes
            try:
                image_bytes = base64.b64decode(base64_data, validate=True)
            except Exception as e:
                raise ValueError(f"Failed to decode image data: {str(e)}")
        else:
            raise ValueError(f"Image data must be a string or bytes, got {type(image_data)}")
        
        # Determine filename from mime type
        # Try mimeType first, then format attribute
        mime_type = getattr(image_content, 'mimeType', None)
        if not mime_type:
            format_attr = getattr(image_content, 'format', 'png')
            mime_type = f'image/{format_attr}' if format_attr else 'image/png'
        
        if 'png' in mime_type:
            filename = "image.png"
        elif 'jpeg' in mime_type or 'jpg' in mime_type:
            filename = "image.jpg"
        else:
            filename = "image.png"
        
        return self.upload_image(image_bytes, filename, subfolder=self.settings.upload_subfolder)
    
    def upload_image(self, image_bytes: bytes, filename: str = "image.png", upload_type: str = "input", subfolder: str = "") -> Dict[str, Any]:
        """Upload an image to the ComfyUI server.
        
        Args:
            image_bytes: The image data as bytes.
            filename: The requested filename for the uploaded image (may be renamed by ComfyUI to avoid conflicts).
            upload_type: The type of upload - "input", "temp", or "output" (default: "input").
            subfolder: The subfolder name for the uploaded image (default: empty string for no subfolder).
            
        Returns:
            Dictionary containing the upload result. The response includes the actual filename used by ComfyUI
            (which may differ from the requested filename if there was a conflict). Expected keys:
            - 'name' or 'filename': The actual filename used by ComfyUI (use this in workflows)
            - 'subfolder': The subfolder where the image was stored (may be empty string or the subfolder from response)
            - 'type': The upload type
        """
        # Create multipart form data matching ComfyUI's expected format
        boundary = uuid.uuid4().hex
        form_data = []
        
        # Add image file - ComfyUI expects "image" field
        form_data.append(f'--{boundary}'.encode())
        form_data.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode())
        form_data.append(b'Content-Type: image/png')
        form_data.append(b'')
        form_data.append(image_bytes)
        
        # Add type field if specified (ComfyUI may use this)
        if upload_type:
            form_data.append(f'--{boundary}'.encode())
            form_data.append(f'Content-Disposition: form-data; name="type"'.encode())
            form_data.append(b'')
            form_data.append(upload_type.encode())
        
        # Add subfolder field if specified
        if subfolder:
            form_data.append(f'--{boundary}'.encode())
            form_data.append(f'Content-Disposition: form-data; name="subfolder"'.encode())
            form_data.append(b'')
            form_data.append(subfolder.encode())
        
        form_data.append(f'--{boundary}--'.encode())
        
        body = b'\r\n'.join(form_data)
        
        # Create request with multipart headers
        upload_headers = {}
        if self.settings.authentication:
            upload_headers["Authorization"] = self.settings.authentication
        upload_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        
        req = urllib.request.Request(
            f"{self.http_url}/upload/image",
            headers=upload_headers,
            data=body
        )
        
        response = urllib.request.urlopen(req)
        upload_response = json.loads(response.read())
        
        # Ensure the response has the expected structure
        # ComfyUI may return the actual filename used (which could be renamed to avoid conflicts)
        # Normalize the response to always have 'name' and 'subfolder' fields
        if isinstance(upload_response, dict):
            # If response has 'filename' but not 'name', use 'filename' as 'name'
            if 'name' not in upload_response and 'filename' in upload_response:
                upload_response['name'] = upload_response['filename']
            # Ensure 'name' exists (fallback to requested filename if not in response)
            if 'name' not in upload_response:
                upload_response['name'] = filename
            
            # Ensure 'subfolder' exists - prefer response value (ComfyUI may override it)
            # If not in response, use the one we sent, or empty string
            if 'subfolder' not in upload_response:
                upload_response['subfolder'] = subfolder if subfolder else ""
            # Note: If 'subfolder' is in response (even if empty string), we use that value
            # as ComfyUI may have modified it
        
        return upload_response
    
    async def process_workflow(self, workflow: Any, params: Dict[str, Any], return_url: bool = False):
        logger.info(f"process_workflow called: workflow type={type(workflow).__name__}, params={list(params.keys())}, return_url={return_url}")
        
        if isinstance(workflow, str):
            workflow_path = os.path.join(self.settings.workflow_dir, f"{workflow}.json")
            logger.info(f"Loading workflow from file: {workflow_path}")
            if not os.path.exists(workflow_path):
                logger.error(f"Workflow file not found: {workflow_path}")
                raise Exception(f"Workflow {workflow} not found")
            with open(workflow_path, "r", encoding='utf-8') as f:
                prompt = json.load(f)
            logger.info(f"Workflow loaded from file, contains {len(prompt)} node(s)")
        else:
            prompt = workflow
            logger.info(f"Using provided workflow object, contains {len(prompt) if isinstance(prompt, dict) else 'unknown'} node(s)")

        logger.info(f"Updating workflow parameters: {list(params.keys())}")
        self.update_workflow_params(prompt, params)
        logger.info("Workflow parameters updated")

        logger.info(f"Connecting to WebSocket: {self.ws_url}")
        ws = websocket.WebSocket()
        try:
            if self.settings.authentication:
                ws.connect(self.ws_url, header=[f"Authorization: {self.settings.authentication}"])
            else:
                ws.connect(self.ws_url)
            logger.info("WebSocket connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {str(e)}", exc_info=True)
            raise

        try:
            logger.info("Starting image retrieval from workflow...")
            images = self.get_images(ws, prompt, return_url)
            logger.info(f"process_workflow: Returning images dict with {len(images)} node(s)")
            return images
        except Exception as e:
            logger.error(f"Error in process_workflow: {str(e)}", exc_info=True)
            raise
        finally:
            logger.info("Closing WebSocket connection")
            ws.close()
            logger.info("WebSocket connection closed")

    def get_images(self, ws, prompt, return_url):
        logger.info("Starting get_images: queueing prompt...")
        queue_result = self.queue_prompt(prompt)
        prompt_id = queue_result["prompt_id"]
        logger.info(f"Prompt queued successfully, prompt_id: {prompt_id}")
        
        output_images = {}
        
        logger.info("Waiting for workflow execution to complete...")
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    node = data.get("node")
                    if node is None and data["prompt_id"] == prompt_id:
                        logger.info(f"Workflow execution completed for prompt_id: {prompt_id}")
                        break
                    elif node is not None:
                        logger.debug(f"Executing node: {node}, prompt_id: {data['prompt_id']}")
            else:
                logger.debug(f"Received non-string message from websocket, type: {type(out)}")
                continue

        logger.info(f"Fetching history for prompt_id: {prompt_id}")
        history_data = self.get_history(prompt_id)
        
        if prompt_id not in history_data:
            logger.error(f"Prompt {prompt_id} not found in history")
            return output_images
        
        history = history_data[prompt_id]
        logger.info(f"History retrieved, found {len(history.get('outputs', {}))} output node(s)")
        
        if "outputs" not in history:
            logger.warning(f"No outputs found in history for prompt_id: {prompt_id}")
            return output_images
        
        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            logger.info(f"Processing output node {node_id}, keys: {list(node_output.keys())}")
            
            if "images" in node_output:
                image_list = node_output["images"]
                logger.info(f"Node {node_id} has {len(image_list)} image(s)")
                
                if return_url:
                    output_images[node_id] = []
                    for idx, image in enumerate(image_list):
                        data = {"filename": image["filename"], "subfolder": image["subfolder"], "type": image["type"]}
                        url_values = urllib.parse.urlencode(data)
                        url = f"{self.http_url}/view?{url_values}"
                        output_images[node_id].append(url)
                        logger.info(f"Node {node_id}, image {idx + 1}: Added URL {url}")
                else:
                    output_images[node_id] = []
                    for idx, image in enumerate(image_list):
                        logger.info(f"Node {node_id}, image {idx + 1}: Fetching image - filename='{image['filename']}', subfolder='{image.get('subfolder', '')}', type='{image['type']}'")
                        try:
                            image_bytes = self.get_image(image["filename"], image["subfolder"], image["type"])
                            image_size = len(image_bytes) if image_bytes else 0
                            logger.info(f"Node {node_id}, image {idx + 1}: Successfully fetched {image_size} bytes")
                            output_images[node_id].append(image_bytes)
                        except Exception as e:
                            logger.error(f"Node {node_id}, image {idx + 1}: Failed to fetch image: {str(e)}", exc_info=True)
                            raise
            else:
                logger.warning(f"Node {node_id} has no 'images' key in output")

        total_images = sum(len(images) for images in output_images.values())
        logger.info(f"get_images: Successfully retrieved {total_images} total image(s) from {len(output_images)} node(s)")
        return output_images

    def update_workflow_params(self, prompt, params):
        if not params:
            return

        for node_id, node in prompt.items():
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
            # Handle parameter nodes - inject values into parameter nodes only
            elif node_title == "Param Seed" and "seed" in params:
                if node.get("class_type") == "PrimitiveInt":
                    node["inputs"]["value"] = params["seed"]
            elif node_title == "Param Steps" and "steps" in params:
                if node.get("class_type") == "PrimitiveInt":
                    node["inputs"]["value"] = params["steps"]
            elif node_title == "Param CFG" and "cfg" in params:
                if node.get("class_type") == "PrimitiveFloat":
                    node["inputs"]["value"] = params["cfg"]
                elif node.get("class_type") == "PrimitiveInt":
                    node["inputs"]["value"] = int(params["cfg"])
            # Handle LoadImage nodes - inject directly into LoadImage nodes using title to match images
            elif node_title and "Load Image" in node_title:
                # Extract index from title (e.g., "Load Image 1" -> 1, "Load Image 2" -> 2)
                # Title format is "Load Image" followed by optional number
                index = 1  # Default to 1 if no number specified
                title_parts = node_title.split()
                if len(title_parts) >= 3 and title_parts[2].isdigit():
                    index = int(title_parts[2])
                elif len(title_parts) == 2:
                    # Just "Load Image" without a number, default to 1
                    index = 1
                
                # Convert to 0-based index for list access
                list_index = index - 1 if index > 0 else 0
                
                # Support multiple uploaded images via list
                if "uploaded_images" in params:
                    uploaded_images = params["uploaded_images"]
                    if isinstance(uploaded_images, list) and len(uploaded_images) > 0:
                        # Use the image at the corresponding index
                        uploaded = uploaded_images[list_index] if list_index < len(uploaded_images) else uploaded_images[0]
                        if isinstance(uploaded, dict):
                            # Use the actual name returned by ComfyUI (may be renamed to avoid conflicts)
                            # Priority: 'name' > 'filename' > empty string
                            image_name = uploaded.get("name") or uploaded.get("filename") or ""
                            if image_name:
                                node["inputs"]["image"] = image_name
                                # Set subfolder from response (ComfyUI may have modified it)
                                # Only set if it's a non-empty string
                                if "subfolder" in uploaded:
                                    subfolder_value = uploaded["subfolder"]
                                    if subfolder_value and isinstance(subfolder_value, str):
                                        node["inputs"]["subfolder"] = subfolder_value
                        elif isinstance(uploaded, str):
                            # If it's just a string, treat it as the image name
                            node["inputs"]["image"] = uploaded
                # Support single uploaded image (for first LoadImage node only, index == 1)
                elif "uploaded_image" in params and index == 1:
                    uploaded = params["uploaded_image"]
                    if isinstance(uploaded, dict):
                        image_name = uploaded.get("name") or uploaded.get("filename") or ""
                        if image_name:
                            node["inputs"]["image"] = image_name
                            # Set subfolder from response (ComfyUI may have modified it)
                            # Only set if it's a non-empty string
                            if "subfolder" in uploaded:
                                subfolder_value = uploaded["subfolder"]
                                if subfolder_value and isinstance(subfolder_value, str):
                                    node["inputs"]["subfolder"] = subfolder_value
                    elif isinstance(uploaded, str):
                        node["inputs"]["image"] = uploaded
            # Handle empty latent image (resolution)
            elif node_title == "Empty Latent Image":
                if "width" in params:
                    node["inputs"]["width"] = params["width"]
                if "height" in params:
                    node["inputs"]["height"] = params["height"]
