from __future__ import annotations

import json
import base64
import logging
from enum import Enum
from typing import Optional

from client.comfyui import ComfyUI
from fastmcp.utilities.types import Image
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageInput(BaseModel):
    """Pydantic-compatible image input model for MCP tools.
    
    This model represents an image that can be passed as a parameter to MCP tools.
    The Image type from FastMCP is not directly compatible with Pydantic schema generation
    for tool parameters, so this model serves as an intermediary.
    """
    data: str = Field(..., description="Base64-encoded image data")
    format: str = Field(default="png", description="Image format (png, jpeg, etc.)")
    
    def to_image(self) -> Image:
        """Convert this Pydantic model to a FastMCP Image object."""
        # FastMCP Image expects raw image bytes (not base64-encoded)
        # FastMCP handles base64 encoding internally when serializing to JSON
        if isinstance(self.data, str):
            # Data is a base64-encoded string, decode it to get raw image bytes
            # Handle data URI format (data:image/png;base64,...)
            if self.data.startswith("data:"):
                # Extract base64 part from data URI
                base64_data = self.data.split(",", 1)[1]
            else:
                # Assume it's already base64 string
                base64_data = self.data
            
            # Strip whitespace and newlines from base64 string
            base64_data = base64_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')
            
            # Add padding if needed (base64 strings must be multiple of 4)
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)
            
            # Decode base64 string to raw image bytes
            try:
                data_bytes = base64.b64decode(base64_data, validate=True)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image data: {str(e)}")
        else:
            # Assume it's already raw image bytes
            data_bytes = self.data
        
        # Create Image object with raw image bytes (FastMCP handles base64 encoding internally)
        return Image(data=data_bytes, format=self.format)


class MCPTransport(str, Enum):
    stdio = "stdio"
    sse = "sse"
    http = "http"


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="mcp_", env_file=".env", extra='ignore')

    host: str = "0.0.0.0"
    port: int = 8000
    transport: MCPTransport = MCPTransport.http


settings = ServerSettings()
mcp = FastMCP("comfyui", host=settings.host, port=settings.port, stateless_http=True)
logger.info("Initializing ComfyUI client...")
comfyui_client = ComfyUI()

def extract_image_urls(images: dict) -> list[str]:
    """Extract the image URLs from a workflow result.

    Args:
        images: Dictionary mapping node IDs to lists of image URLs.

    Returns:
        list[str]: List of ComfyUI download URLs for the generated images.
    """
    logger.info(f"extract_image_urls called with {len(images)} node(s) containing images")
    
    if not images:
        logger.error("No images were generated - images dict is empty")
        raise ValueError("No images were generated")

    result = []
    total_images = 0
    
    for node_id, url_list in images.items():
        logger.info(f"Processing node {node_id} with {len(url_list) if url_list else 0} URL(s)")
        
        if not url_list:
            logger.warning(f"Node {node_id} has empty URL list, skipping")
            continue

        for idx, url in enumerate(url_list):
            total_images += 1
            logger.info(f"Processing URL {total_images} from node {node_id} (index {idx}): {url}")
            
            # Validate URL
            if not url or not isinstance(url, str):
                logger.error(f"URL {total_images} from node {node_id} is empty or invalid")
                raise ValueError("Image URL is empty or invalid")
            
            result.append(url)
    
    logger.info(f"extract_image_urls: Successfully extracted {len(result)} URL(s) from {total_images} total images")
    return result


@mcp.tool()
async def text_to_image(prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: float = 8.0, denoise: float = 1.0, width: int = 1024, height: int = 1024) -> list[str]:
    """Generate an image from a prompt. Returns ComfyUI download links for the generated images.
    
    Args:
        prompt: The prompt to generate the image from.  Uses natural language prompt.
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: The height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        list[str]: List of ComfyUI download URLs for the generated images. Files can be downloaded from these links.
    """
    logger.info(f"text_to_image called with prompt='{prompt[:50]}...', seed={seed}, steps={steps}, cfg={cfg}, width={width}, height={height}")
    
    params = {"prompt": prompt, "cfg": cfg, "denoise": denoise, "width": width, "height": height}
    if seed is not None:
        params["seed"] = seed
    if steps is not None:
        params["steps"] = steps
    
    logger.info(f"Processing workflow 'text_to_image' with params: {params}")
    images = await comfyui_client.process_workflow("text_to_image", params, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    extracted = extract_image_urls(images)
    logger.info(f"text_to_image: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    logger.info(f"Returning {len(extracted)} download URL(s)")
    return extracted


@mcp.tool()
async def text_to_image_placeholder(prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: float = 7.0, denoise: float = 1.0, width: int = 1024, height: int = 1024) -> list[str]:
    """Generate a placeholder image from a prompt. Optimized for quick placeholder generation. Returns ComfyUI download links for the generated images.
    
    Args:
        prompt: The prompt to generate the placeholder image from. Uses comma separated tags such as "dog, walking, sunset"
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: The height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        list[str]: List of ComfyUI download URLs for the generated placeholder images. Files can be downloaded from these links.
    """
    logger.info(f"text_to_image_placeholder called with prompt='{prompt[:50]}...', seed={seed}, steps={steps}, cfg={cfg}, width={width}, height={height}")
    
    params = {"prompt": prompt, "cfg": cfg, "denoise": denoise, "width": width, "height": height}
    if seed is not None:
        params["seed"] = seed
    if steps is not None:
        params["steps"] = steps
    
    logger.info(f"Processing workflow 'text_to_image_placeholder' with params: {params}")
    workflow_images = await comfyui_client.process_workflow("text_to_image_placeholder", params, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(workflow_images)} node(s)")
    
    extracted = extract_image_urls(workflow_images)
    logger.info(f"text_to_image_placeholder: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    logger.info(f"Returning {len(extracted)} download URL(s)")
    return extracted


@mcp.tool()
async def edit_image(images: list[ImageInput], prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: float = 8.0, denoise: float = 1.0, width: int = 1024, height: int = 1024) -> list[str]:
    """Edit one or more images using a prompt. Returns ComfyUI download links for the edited images.
    
    Args:
        images: List of images to edit. Each image should have base64-encoded data and format.
        prompt: The prompt to guide the image editing. Uses natural language prompt in the form of requests for changes (ex: "Change the person's hair color to blonde")
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The target width of the generated image in pixels.
        height: The target height of the generated image in pixels.
    
    Returns:
        list[str]: List of ComfyUI download URLs for the edited images. Files can be downloaded from these links.
    """
    if not images:
        raise ValueError("At least one image must be provided")
    
    # Convert Pydantic ImageInput models to FastMCP Image objects
    mcp_images = [img_input.to_image() for img_input in images]
    
    # Upload images to ComfyUI (client handles Image conversion)
    # Note: ComfyUI may rename uploaded images to avoid conflicts. The upload response contains
    # the actual filename used, which is automatically used in the workflow.
    upload_results = []
    for img in mcp_images:
        upload_result = comfyui_client.upload_image_from_content(img)
        upload_results.append(upload_result)
    
    # Process workflow with uploaded images (all images available via uploaded_images list)
    # The workflow parameterization will use the actual filenames from upload_results,
    # which may differ from the requested names if ComfyUI renamed them to avoid conflicts.
    params = {
        "prompt": prompt,
        "cfg": cfg,
        "denoise": denoise,
        "uploaded_images": upload_results,  # Pass all uploaded images with their actual names
        "width": width,
        "height": height
    }
    if seed is not None:
        params["seed"] = seed
    if steps is not None:
        params["steps"] = steps
    logger.info(f"Processing workflow 'edit_image' with params: prompt='{prompt[:50]}...', cfg={cfg}, width={width}, height={height}, {len(upload_results)} uploaded image(s)")
    workflow_images = await comfyui_client.process_workflow("edit_image", params, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(workflow_images)} node(s)")
    
    extracted = extract_image_urls(workflow_images)
    logger.info(f"edit_image: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    logger.info(f"Returning {len(extracted)} download URL(s)")
    return extracted


@mcp.tool()
async def run_workflow_from_file(file_path: str = "") -> list[str]:
    """Run a workflow from a file. Returns ComfyUI download links for the generated images.
    
    Args:
        file_path: The absolute path to the file to run.
    
    Returns:
        list[str]: List of ComfyUI download URLs for the generated images. Files can be downloaded from these links.
    """
    logger.info(f"run_workflow_from_file called with file_path='{file_path}'")
    
    with open(file_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    logger.info(f"Loaded workflow from file, processing...")
    images = await comfyui_client.process_workflow(workflow, {}, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    extracted = extract_image_urls(images)
    logger.info(f"run_workflow_from_file: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    logger.info(f"Returning {len(extracted)} download URL(s)")
    return extracted


@mcp.tool()
async def run_workflow_from_json(json_data: Optional[dict] = None) -> list[str]:
    """Run a workflow from JSON data. Returns ComfyUI download links for the generated images.
    
    Args:
        json_data: The JSON workflow to run.
    
    Returns:
        list[str]: List of ComfyUI download URLs for the generated images. Files can be downloaded from these links.
    """
    logger.info(f"run_workflow_from_json called with json_data={'provided' if json_data else 'None'}")
    
    if json_data is None:
        json_data = {}

    logger.info(f"Processing workflow from JSON...")
    images = await comfyui_client.process_workflow(json_data, {}, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    extracted = extract_image_urls(images)
    logger.info(f"run_workflow_from_json: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    logger.info(f"Returning {len(extracted)} download URL(s)")
    return extracted


if __name__ == "__main__":
    mcp.run(transport=settings.transport.value)
