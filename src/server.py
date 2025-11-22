from __future__ import annotations

import json
import base64
import logging
import os
import urllib.request
import urllib.parse
from datetime import datetime
from enum import Enum
from pathlib import Path
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
    transport: MCPTransport = MCPTransport.stdio


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


def download_single_image(url: str, target_directory: str, filename: Optional[str] = None) -> Image:
    """Download a single image from a URL, save it to disk, and return an Image object.
    
    Args:
        url: Image URL to download.
        target_directory: Full local file path (not relative) to a directory where the image will be saved.
        filename: Optional filename for the downloaded image. If not provided, a timestamped filename will be generated.
    
    Returns:
        Image: Image object with raw image bytes (FastMCP handles base64 encoding internally).
    """
    if not url:
        raise ValueError("Image URL is required")
    if not target_directory:
        raise ValueError("target_directory is required")
    
    # Ensure target directory exists
    target_path = Path(target_directory)
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory: {target_path.absolute()}")
    
    logger.info(f"Downloading image from {url}")
    
    try:
        # Download image
        req = urllib.request.Request(url)
        if comfyui_client.settings.authentication:
            req.add_header("Authorization", comfyui_client.settings.authentication)
        
        with urllib.request.urlopen(req) as response:
            image_bytes = response.read()
            logger.info(f"Downloaded {len(image_bytes)} bytes")
        
        # Determine file extension from URL or content
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        # Determine filename and extension
        if filename:
            # Use provided filename
            save_filename = filename
            # Extract extension from provided filename if it has one
            if '.' in save_filename:
                ext = save_filename.rsplit('.', 1)[1].lower()
            else:
                # If no extension in provided filename, try to get it from URL
                url_filename = query_params.get('filename', ['image.png'])[0]
                if '.' in url_filename:
                    ext = url_filename.rsplit('.', 1)[1].lower()
                    save_filename = f"{save_filename}.{ext}"
                else:
                    ext = 'png'
                    save_filename = f"{save_filename}.png"
        else:
            # Generate unique filename with timestamp
            url_filename = query_params.get('filename', ['image.png'])[0]
            if '.' in url_filename:
                ext = url_filename.rsplit('.', 1)[1].lower()
            else:
                # Default to png if we can't determine
                ext = 'png'
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_filename = f"image_{timestamp}.{ext}"
        
        save_path = target_path / save_filename
        
        # Save image to disk
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"Saved image to {save_path.absolute()}")
        
        # Create Image object with raw bytes (FastMCP handles base64 encoding internally)
        # Determine format from extension
        format_map = {
            'png': 'png',
            'jpg': 'jpeg',
            'jpeg': 'jpeg',
            'webp': 'webp',
            'gif': 'gif'
        }
        image_format = format_map.get(ext, 'png')
        
        image_obj = Image(data=image_bytes, format=image_format)
        return image_obj
        
    except Exception as e:
        logger.error(f"Failed to download/save image from {url}: {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def download_image(image_url: str, target_directory: str, filename: Optional[str] = None) -> Image:
    """Download a single image from a URL and save it to the target directory. Returns Image object for preview.
    
    Args:
        image_url: The URL of the image to download.
        target_directory: Required. Full local file path (not relative) to a directory where the image will be saved. Generated files can be found there.
        filename: Optional filename for the downloaded image. If not provided, a timestamped filename will be generated automatically.
    
    Returns:
        Image: Image object for preview. The image file is saved to the target directory.
    """
    logger.info(f"download_image called with image_url='{image_url}', target_directory={target_directory}, filename={filename}")
    return download_single_image(image_url, target_directory, filename)


@mcp.tool()
async def text_to_image(prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: Optional[float] = None, denoise: Optional[float] = None, width: Optional[int] = None, height: Optional[int] = None) -> list[str]:
    """Generate an image from a prompt. Returns ComfyUI download URLs for the generated images. Use the download_image tool to download and save the images.
    
    Args:
        prompt: The prompt to generate the image from.  Uses natural language prompt.
        seed: Optional integer seed to use for the image generation.
        steps: Optional integer number of steps to use for the image generation.
        cfg: Optional float CFG scale to use for the image generation.
        denoise: Optional float denoise strength to use for the image generation.
        width: Optional integer width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: Optional integer height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        list[str]: List of ComfyUI download URLs for the generated images. Use the download_image tool to download and save these images.
    """
    logger.info(f"text_to_image called with prompt='{prompt[:50]}...', seed={seed}, steps={steps}, cfg={cfg}, denoise={denoise}, width={width}, height={height}")
    
    params = {"prompt": prompt}
    if seed is not None:
        params["seed"] = seed
    if steps is not None:
        params["steps"] = steps
    if cfg is not None:
        params["cfg"] = cfg
    if denoise is not None:
        params["denoise"] = denoise
    if width is not None:
        params["width"] = width
    if height is not None:
        params["height"] = height
    
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
async def text_to_image_placeholder(prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: Optional[float] = None, denoise: Optional[float] = None, width: Optional[int] = None, height: Optional[int] = None) -> list[str]:
    """Generate a placeholder image from a prompt. Optimized for quick placeholder generation. Returns ComfyUI download URLs for the generated images. Use the download_image tool to download and save the images.
    
    Args:
        prompt: The prompt to generate the placeholder image from. Uses comma separated tags such as "dog, walking, sunset"
        seed: Optional integer seed to use for the image generation.
        steps: Optional integer number of steps to use for the image generation.
        cfg: Optional float CFG scale to use for the image generation.
        denoise: Optional float denoise strength to use for the image generation.
        width: Optional integer width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: Optional integer height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        list[str]: List of ComfyUI download URLs for the generated placeholder images. Use the download_image tool to download and save these images.
    """
    logger.info(f"text_to_image_placeholder called with prompt='{prompt[:50]}...', seed={seed}, steps={steps}, cfg={cfg}, denoise={denoise}, width={width}, height={height}")
    
    params = {"prompt": prompt}
    if seed is not None:
        params["seed"] = seed
    if steps is not None:
        params["steps"] = steps
    if cfg is not None:
        params["cfg"] = cfg
    if denoise is not None:
        params["denoise"] = denoise
    if width is not None:
        params["width"] = width
    if height is not None:
        params["height"] = height
    
    logger.info(f"Processing workflow 'text_to_image_placeholder' with params: {params}")
    workflow_images = await comfyui_client.process_workflow("text_to_image_placeholder", params, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(workflow_images)} node(s)")
    
    extracted = extract_image_urls(workflow_images)
    logger.info(f"text_to_image_placeholder: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    logger.info(f"Returning {len(extracted)} download URL(s)")
    return extracted


def extract_filename_from_url(url: str) -> str:
    """Extract filename from a ComfyUI download URL.
    
    Args:
        url: ComfyUI download URL (e.g., "http://localhost:8188/view?filename=ComfyUI_00115_.png&subfolder=&type=output")
    
    Returns:
        str: Filename extracted from URL (e.g., "ComfyUI_00115_.png")
    """
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    filename = query_params.get('filename', [''])[0]
    if not filename:
        raise ValueError(f"Could not extract filename from URL: {url}")
    return filename


@mcp.tool()
async def edit_image(image: str, prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: Optional[float] = None, denoise: Optional[float] = None, width: Optional[int] = None, height: Optional[int] = None) -> list[str]:
    """Edit an image using a prompt. The image must already exist on the ComfyUI server (from a previous generation). Returns ComfyUI download URLs for the edited images. Use the download_image tool to download and save the images.
    
    Args:
        image: Filename of an image already on the ComfyUI server (e.g., "ComfyUI_00115_.png") or a download URL from which the filename will be extracted.
        prompt: The prompt to guide the image editing. Uses natural language prompt in the form of requests for changes (ex: "Change the person's hair color to blonde")
        seed: Optional integer seed to use for the image generation.
        steps: Optional integer number of steps to use for the image generation.
        cfg: Optional float CFG scale to use for the image generation.
        denoise: Optional float denoise strength to use for the image generation.
        width: Optional integer target width of the generated image in pixels.
        height: Optional integer target height of the generated image in pixels.
    
    Returns:
        list[str]: List of ComfyUI download URLs for the edited images. Use the download_image tool to download and save these images.
    """
    if not image:
        raise ValueError("image parameter is required")
    
    # Extract filename from URL if it's a URL, otherwise use it as-is
    if image.startswith("http://") or image.startswith("https://"):
        filename = extract_filename_from_url(image)
        logger.info(f"Extracted filename '{filename}' from URL: {image}")
    else:
        filename = image
        logger.info(f"Using provided filename: {filename}")
    
    # Format filename for LoadImageOutput node: "filename [output]"
    image_reference = f"{filename} [output]"
    
    params = {
        "prompt": prompt,
        "image": image_reference,  # Pass image reference for LoadImageOutput node
    }
    if seed is not None:
        params["seed"] = seed
    if steps is not None:
        params["steps"] = steps
    if cfg is not None:
        params["cfg"] = cfg
    if denoise is not None:
        params["denoise"] = denoise
    if width is not None:
        params["width"] = width
    if height is not None:
        params["height"] = height
    logger.info(f"Processing workflow 'edit_image' with params: prompt='{prompt[:50]}...', cfg={cfg}, denoise={denoise}, width={width}, height={height}, image='{image_reference}'")
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
    """Run a workflow from a file. Returns ComfyUI download URLs for the generated images. Use the download_image tool to download and save the images.
    
    Args:
        file_path: The absolute path to the file to run.
    
    Returns:
        list[str]: List of ComfyUI download URLs for the generated images. Use the download_image tool to download and save these images.
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
    """Run a workflow from JSON data. Returns ComfyUI download URLs for the generated images. Use the download_image tool to download and save the images.
    
    Args:
        json_data: The JSON workflow to run.
    
    Returns:
        list[str]: List of ComfyUI download URLs for the generated images. Use the download_image tool to download and save these images.
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
