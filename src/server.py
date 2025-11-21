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


def download_and_save_images(urls: list[str], output_directory: str) -> list[Image]:
    """Download images from URLs, save them to disk, and return Image objects.
    
    Args:
        urls: List of image URLs to download.
        output_directory: Directory where images will be saved.
    
    Returns:
        list[Image]: List of Image objects with raw image bytes (FastMCP handles base64 encoding).
    """
    if not urls:
        raise ValueError("No image URLs provided")
    
    # Ensure output directory exists
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")
    
    image_objects = []
    
    for idx, url in enumerate(urls):
        logger.info(f"Downloading image {idx + 1}/{len(urls)} from {url}")
        
        try:
            # Download image
            req = urllib.request.Request(url)
            if comfyui_client.settings.authentication:
                req.add_header("Authorization", comfyui_client.settings.authentication)
            
            with urllib.request.urlopen(req) as response:
                image_bytes = response.read()
                logger.info(f"Downloaded {len(image_bytes)} bytes for image {idx + 1}")
            
            # Determine file extension from URL or content
            parsed_url = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Try to get format from filename in URL
            filename = query_params.get('filename', ['image.png'])[0]
            if '.' in filename:
                ext = filename.rsplit('.', 1)[1].lower()
            else:
                # Default to png if we can't determine
                ext = 'png'
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"image_{timestamp}_{idx + 1}.{ext}"
            save_path = output_path / save_filename
            
            # Save image to disk
            with open(save_path, 'wb') as f:
                f.write(image_bytes)
            logger.info(f"Saved image {idx + 1} to {save_path.absolute()}")
            
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
            image_objects.append(image_obj)
            
        except Exception as e:
            logger.error(f"Failed to download/save image {idx + 1} from {url}: {str(e)}", exc_info=True)
            raise
    
    logger.info(f"Successfully downloaded and saved {len(image_objects)} image(s) to {output_path.absolute()}")
    return image_objects


@mcp.tool()
async def text_to_image(output_directory: str, prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: Optional[float] = None, denoise: Optional[float] = None, width: Optional[int] = None, height: Optional[int] = None) -> list[Image]:
    """Generate an image from a prompt. Returns Image objects for preview and saves files to the output directory.
    
    Args:
        output_directory: Required. Full local file path (not relative) to a directory where generated image files will be saved. Generated files can be found there.
        prompt: The prompt to generate the image from.  Uses natural language prompt.
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: The height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        list[Image]: List of Image objects for preview. Generated files are saved to the output directory.
    """
    if not output_directory:
        raise ValueError("output_directory is required")
    
    logger.info(f"text_to_image called with prompt='{prompt[:50]}...', seed={seed}, steps={steps}, cfg={cfg}, denoise={denoise}, width={width}, height={height}, output_directory={output_directory}")
    
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
    
    # Download, save, and convert to Image objects
    image_objects = download_and_save_images(extracted, output_directory)
    logger.info(f"Returning {len(image_objects)} Image object(s)")
    return image_objects


@mcp.tool()
async def text_to_image_placeholder(output_directory: str, prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: Optional[float] = None, denoise: Optional[float] = None, width: Optional[int] = None, height: Optional[int] = None) -> list[Image]:
    """Generate a placeholder image from a prompt. Optimized for quick placeholder generation. Returns Image objects for preview and saves files to the output directory.
    
    Args:
        output_directory: Required. Full local file path (not relative) to a directory where generated image files will be saved. Generated files can be found there.
        prompt: The prompt to generate the placeholder image from. Uses comma separated tags such as "dog, walking, sunset"
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: The height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        list[Image]: List of Image objects for preview. Generated files are saved to the output directory.
    """
    if not output_directory:
        raise ValueError("output_directory is required")
    
    logger.info(f"text_to_image_placeholder called with prompt='{prompt[:50]}...', seed={seed}, steps={steps}, cfg={cfg}, denoise={denoise}, width={width}, height={height}, output_directory={output_directory}")
    
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
    
    # Download, save, and convert to Image objects
    image_objects = download_and_save_images(extracted, output_directory)
    logger.info(f"Returning {len(image_objects)} Image object(s)")
    return image_objects


@mcp.tool()
async def edit_image(output_directory: str, images: list[ImageInput], prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: Optional[float] = None, denoise: Optional[float] = None, width: Optional[int] = None, height: Optional[int] = None) -> list[Image]:
    """Edit one or more images using a prompt. Returns Image objects for preview and saves files to the output directory.
    
    Args:
        output_directory: Required. Full local file path (not relative) to a directory where generated image files will be saved. Generated files can be found there.
        images: List of images to edit. Each image should have base64-encoded data and format.
        prompt: The prompt to guide the image editing. Uses natural language prompt in the form of requests for changes (ex: "Change the person's hair color to blonde")
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The target width of the generated image in pixels.
        height: The target height of the generated image in pixels.
    
    Returns:
        list[Image]: List of Image objects for preview. Generated files are saved to the output directory.
    """
    if not images:
        raise ValueError("At least one image must be provided")
    if not output_directory:
        raise ValueError("output_directory is required")
    
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
        "uploaded_images": upload_results,  # Pass all uploaded images with their actual names
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
    logger.info(f"Processing workflow 'edit_image' with params: prompt='{prompt[:50]}...', cfg={cfg}, denoise={denoise}, width={width}, height={height}, {len(upload_results)} uploaded image(s), output_directory={output_directory}")
    workflow_images = await comfyui_client.process_workflow("edit_image", params, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(workflow_images)} node(s)")
    
    extracted = extract_image_urls(workflow_images)
    logger.info(f"edit_image: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    # Download, save, and convert to Image objects
    image_objects = download_and_save_images(extracted, output_directory)
    logger.info(f"Returning {len(image_objects)} Image object(s)")
    return image_objects


@mcp.tool()
async def run_workflow_from_file(output_directory: str, file_path: str = "") -> list[Image]:
    """Run a workflow from a file. Returns Image objects for preview and saves files to the output directory.
    
    Args:
        output_directory: Required. Full local file path (not relative) to a directory where generated image files will be saved. Generated files can be found there.
        file_path: The absolute path to the file to run.
    
    Returns:
        list[Image]: List of Image objects for preview. Generated files are saved to the output directory.
    """
    if not output_directory:
        raise ValueError("output_directory is required")
    
    logger.info(f"run_workflow_from_file called with file_path='{file_path}', output_directory={output_directory}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    logger.info(f"Loaded workflow from file, processing...")
    images = await comfyui_client.process_workflow(workflow, {}, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    extracted = extract_image_urls(images)
    logger.info(f"run_workflow_from_file: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    # Download, save, and convert to Image objects
    image_objects = download_and_save_images(extracted, output_directory)
    logger.info(f"Returning {len(image_objects)} Image object(s)")
    return image_objects


@mcp.tool()
async def run_workflow_from_json(output_directory: str, json_data: Optional[dict] = None) -> list[Image]:
    """Run a workflow from JSON data. Returns Image objects for preview and saves files to the output directory.
    
    Args:
        output_directory: Required. Full local file path (not relative) to a directory where generated image files will be saved. Generated files can be found there.
        json_data: The JSON workflow to run.
    
    Returns:
        list[Image]: List of Image objects for preview. Generated files are saved to the output directory.
    """
    if not output_directory:
        raise ValueError("output_directory is required")
    
    logger.info(f"run_workflow_from_json called with json_data={'provided' if json_data else 'None'}, output_directory={output_directory}")
    
    if json_data is None:
        json_data = {}

    logger.info(f"Processing workflow from JSON...")
    images = await comfyui_client.process_workflow(json_data, {}, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    extracted = extract_image_urls(images)
    logger.info(f"run_workflow_from_json: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    # Download, save, and convert to Image objects
    image_objects = download_and_save_images(extracted, output_directory)
    logger.info(f"Returning {len(image_objects)} Image object(s)")
    return image_objects


if __name__ == "__main__":
    mcp.run(transport=settings.transport.value)
