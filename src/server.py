from __future__ import annotations

import json
import logging
import random
import urllib.request
import urllib.parse
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Annotated

from pydantic import Field

from client.comfyui import ComfyUI
from fastmcp.utilities.types import Image
from fastmcp import FastMCP
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPTransport(str, Enum):
    stdio = "stdio"
    sse = "sse"
    http = "http"


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="mcp_", env_file=".env", extra='ignore')

    host: str = "0.0.0.0"
    port: int = 8000
    transport: MCPTransport = MCPTransport.stdio
    log_level: str = "INFO"


settings = ServerSettings()
logging.basicConfig(level=settings.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
mcp = FastMCP("comfyui", host=settings.host, port=settings.port, stateless_http=True)
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


def download_image_from_url(url: str) -> tuple[bytes, str]:
    """Download an image from a URL and return the image bytes and format.
    
    Args:
        url: Image URL to download.
    
    Returns:
        tuple[bytes, str]: Image bytes and format (e.g., 'png', 'jpeg').
    """
    if not url:
        raise ValueError("Image URL is required")
    
    logger.info(f"Downloading image from {url}")
    
    try:
        # Download image
        req = urllib.request.Request(url)
        if comfyui_client.settings.authentication:
            req.add_header("Authorization", comfyui_client.settings.authentication)
        
        with urllib.request.urlopen(req) as response:
            image_bytes = response.read()
            logger.info(f"Downloaded {len(image_bytes)} bytes")
        
        # Determine file extension from URL
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        url_filename = query_params.get('filename', ['image.png'])[0]
        if '.' in url_filename:
            ext = url_filename.rsplit('.', 1)[1].lower()
        else:
            ext = 'png'
        
        # Map extension to format
        format_map = {
            'png': 'png',
            'jpg': 'jpeg',
            'jpeg': 'jpeg',
            'webp': 'webp',
            'gif': 'gif'
        }
        image_format = format_map.get(ext, 'png')
        
        return image_bytes, image_format
        
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {str(e)}", exc_info=True)
        raise


def save_image_with_path(image_bytes: bytes, save_path: str, filename_hint: Optional[str] = None) -> Path:
    """Save an image to disk with flexible path handling.
    
    Args:
        image_bytes: Image data as bytes.
        save_path: Either a directory path or a full file path. If directory, filename will be generated.
        filename_hint: Optional filename hint (used if save_path is a directory).
    
    Returns:
        Path: The full path where the image was saved.
    """
    save_path_obj = Path(save_path)
    
    # Check if save_path is a directory or a file path
    if save_path_obj.is_dir() or (not save_path_obj.exists() and not save_path_obj.suffix):
        # It's a directory (or intended to be one)
        save_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if filename_hint:
            if '.' in filename_hint:
                save_filename = filename_hint
            else:
                # Try to get extension from filename_hint or default to png
                ext = 'png'
                if '.' in filename_hint:
                    ext = filename_hint.rsplit('.', 1)[1].lower()
                save_filename = f"{filename_hint}.{ext}"
        else:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_filename = f"image_{timestamp}.png"
        
        final_path = save_path_obj / save_filename
    else:
        # It's a file path
        final_path = save_path_obj
        # Ensure parent directory exists
        final_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image to disk
    with open(final_path, 'wb') as f:
        f.write(image_bytes)
    logger.info(f"Saved image to {final_path.absolute()}")
    
    return final_path


def url_to_image(url: str, save_path: Optional[str] = None, filename_hint: Optional[str] = None) -> Image:
    """Download an image from URL, optionally save it, and return Image object.
    
    Args:
        url: Image URL to download.
        save_path: Optional path to save the image (directory or full file path).
        filename_hint: Optional filename hint (used if save_path is a directory).
    
    Returns:
        Image: Image object with raw image bytes.
    """
    image_bytes, image_format = download_image_from_url(url)
    
    # Save if save_path is provided
    if save_path:
        # Extract filename hint from URL if not provided
        if not filename_hint:
            parsed_url = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            filename_hint = query_params.get('filename', [None])[0]
        
        save_image_with_path(image_bytes, save_path, filename_hint)
    
    # Create Image object with raw bytes (FastMCP handles base64 encoding internally)
    image_obj = Image(data=image_bytes, format=image_format)
    return image_obj


def handle_image_response(images: dict, save_path: Optional[str] = None) -> Image:
    """Handle a workflow response that should contain a single image.
    
    Extracts image URLs, downloads the first image, optionally saves it, and returns an Image object.
    
    Args:
        images: Dictionary mapping node IDs to lists of image URLs from workflow execution.
        save_path: Optional path to save the image (directory or full file path).
    
    Returns:
        Image: The generated image.
    
    Raises:
        ValueError: If no images were generated.
    """
    extracted = extract_image_urls(images)
    logger.info(f"handle_image_response: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    # Warn if more than one image (single image expected)
    if len(extracted) > 1:
        logger.warning(f"Workflow returned {len(extracted)} images, but expected 1. Using first image.")
    
    # Download and optionally save the image
    image_obj = url_to_image(extracted[0], save_path)
    logger.info(f"handle_image_response: Returning generated image")
    return image_obj


def handle_images_response(images: dict, save_path: Optional[str] = None) -> list[Image]:
    """Handle a workflow response that may contain multiple images.
    
    Extracts image URLs, downloads all images, optionally saves them, and returns a list of Image objects.
    
    Args:
        images: Dictionary mapping node IDs to lists of image URLs from workflow execution.
        save_path: Optional path to save the image(s). Can be a directory path or a full file path.
                   If directory and multiple images are generated, each will be saved with a unique filename.
                   If file path, only the first image will be saved to that path.
    
    Returns:
        list[Image]: List of generated images.
    
    Raises:
        ValueError: If no images were generated.
    """
    extracted = extract_image_urls(images)
    logger.info(f"handle_images_response: Extracted {len(extracted)} URL(s)")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    # Determine if save_path is a directory or file path
    is_directory = None
    if save_path:
        save_path_obj = Path(save_path)
        # Check if it's an existing directory, or if it doesn't exist and has no file extension
        is_directory = save_path_obj.is_dir() or (not save_path_obj.exists() and not save_path_obj.suffix)
    
    # Download and optionally save all images
    result_images = []
    for idx, url in enumerate(extracted):
        # If save_path is provided and it's a directory, save all images with unique filenames
        # If save_path is a file path, only save the first image to that path
        current_save_path = None
        if save_path:
            if is_directory or idx == 0:
                # Directory (save all) or first image with file path
                current_save_path = save_path
        
        image_obj = url_to_image(url, current_save_path)
        result_images.append(image_obj)
    
    logger.info(f"handle_images_response: Returning {len(result_images)} generated image(s)")
    return result_images


@mcp.tool()
async def tags_to_image(
        tags: Annotated[str, Field(
            description="comma separated tags (e.g. \"dog, walking, sunset\")")] = "",
        width: Annotated[int, Field(
            description="pixel width of the image. Best results are at approximately 1 megapixel (e.g., 1024x1024).")] = 1024,
        height: Annotated[int, Field(
            description="pixel height of the image. Best results are at approximately 1 megapixel (e.g., 1024x1024).")] = 1024,
        save_path: Annotated[Optional[str], Field(
            description="Optional path to save the image. Can be a directory path or a full file path. If not provided, the image will not be saved.")] = None,
) -> Image:
    """Quickly generate an image from a list of tags, and save it. Returns the image."""
    seed = random.randint(0, 2 ** 32 - 1)
    logger.info(
        f"text_to_image_placeholder called with prompt='{tags[:50]}...', seed={seed}, width={width}, height={height}, save_path={save_path}")

    params = {"prompt": tags, "width": width, "height": height, "seed": seed}

    logger.info(f"Processing workflow 'text_to_image_placeholder' with params: {params}")
    workflow_images = await comfyui_client.process_workflow("text_to_image_placeholder", params, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(workflow_images)} node(s)")

    return handle_image_response(workflow_images, save_path)


@mcp.tool()
async def text_to_image(
        prompt: Annotated[str, Field(description="prompt to generate the image from (uses natural language prompt)")] = "",
        width: Annotated[int, Field(description="pixel width of the image. Best results are at approximately 1 megapixel (e.g., 1024x1024).")] = 1024,
        height: Annotated[int, Field(description="pixel height of the image. Best results are at approximately 1 megapixel (e.g., 1024x1024).")] = 1024,
        save_path: Annotated[Optional[str], Field(description="Optional path to save the image. Can be a directory path or a full file path. If not provided, the image will not be saved.")] = None,
) -> Image:
    """Slowly generate an image from a natural language prompt, and save it. Returns the generated image."""
    seed = random.randint(0, 2**32 - 1)
    logger.info(f"text_to_image called with prompt='{prompt[:50]}...', seed={seed}, width={width}, height={height}, save_path={save_path}")
    
    params = {"prompt": prompt, "width": width, "height": height, "seed": seed}
    
    logger.info(f"Processing workflow 'text_to_image' with params: {params}")
    images = await comfyui_client.process_workflow("text_to_image", params, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    return handle_image_response(images, save_path)


def extract_filename_from_url(url: Annotated[str, Field(description="ComfyUI Download URL")]) -> str:
    """Extract filename from a ComfyUI download URL.  Returns filename extracted from URL"""
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    filename = query_params.get('filename', [''])[0]
    if not filename:
        raise ValueError(f"Could not extract filename from URL: {url}")
    return filename


@mcp.tool()
async def edit_image(
        image: Annotated[str, Field(description="Filename of an image already on the ComfyUI server (e.g., \"ComfyUI_00115_.png\") or a download URL from which the filename will be extracted.")],
        prompt: Annotated[str, Field(description="The prompt to guide the image editing. Uses natural language prompt in the form of requests for changes (ex: \"Change the person's hair color to blonde\")")] = "",
        width: Annotated[int, Field(description="target pixel width of the edited image.")] = 1024,
        height: Annotated[int, Field(description="target pixel height of the edited image.")] = 1024,
        save_path: Annotated[Optional[str], Field(description="Optional path to save the image. Can be a directory path or a full file path. If not provided, the image will not be saved.")] = None,
) -> Image:
    """Edit an image using a prompt. The image must already exist on the ComfyUI server (from a previous generation). Returns the generated image."""
    if not image:
        raise ValueError("image parameter is required")
    
    seed = random.randint(0, 2**32 - 1)
    logger.info(f"edit_image called with image='{image}', prompt='{prompt[:50]}...', seed={seed}, width={width}, height={height}, save_path={save_path}")
    
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
        "width": width,
        "height": height,
        "seed": seed,
    }
    logger.info(f"Processing workflow 'edit_image' with params: prompt='{prompt[:50]}...', width={width}, height={height}, image='{image_reference}'")
    workflow_images = await comfyui_client.process_workflow("edit_image", params, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(workflow_images)} node(s)")

    return handle_image_response(workflow_images, save_path)


@mcp.tool()
async def run_workflow_from_file(
        file_path: Annotated[str, Field(description="The absolute path to the file to run.")] = "",
        save_path: Annotated[Optional[str], Field(description="Optional path to save the image(s). Can be a directory path or a full file path. If directory and multiple images are generated, each will be saved with a unique filename. If not provided, the images will not be saved.")] = None,
) -> list[Image]:
    """Run a workflow from a file. Returns the generated image(s)."""
    logger.info(f"run_workflow_from_file called with file_path='{file_path}', save_path={save_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    logger.info(f"Loaded workflow from file, processing...")
    images = await comfyui_client.process_workflow(workflow, {}, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")

    return handle_images_response(images, save_path)


@mcp.tool()
async def run_workflow_from_json(
        json_data: Annotated[Optional[dict], Field(description="The JSON workflow to run.")] = None,
        save_path: Annotated[Optional[str], Field(description="Optional path to save the image(s). Can be a directory path or a full file path. If directory and multiple images are generated, each will be saved with a unique filename. If not provided, the images will not be saved.")] = None,
) -> list[Image]:
    """Run a workflow from JSON data. Returns the generated image(s)."""
    logger.info(f"run_workflow_from_json called with json_data={'provided' if json_data else 'None'}, save_path={save_path}")

    if json_data is None:
        json_data = {}

    logger.info(f"Processing workflow from JSON...")
    images = await comfyui_client.process_workflow(json_data, {}, return_url=True)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")

    return handle_images_response(images, save_path)


if __name__ == "__main__":
    mcp.run(transport=settings.transport.value)
