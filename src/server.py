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


def extract_images(images: dict) -> list[Image]:
    """Extract the images from a workflow result.

    Args:
        images: Dictionary mapping node IDs to lists of image bytes.

    Returns:
        list[Image]: The images as an MCP Image objects.
    """
    logger.info(f"extract_images called with {len(images)} node(s) containing images")
    
    if not images:
        logger.error("No images were generated - images dict is empty")
        raise ValueError("No images were generated")

    result = []
    total_images = 0
    
    for node_id, resp_image_list in images.items():
        logger.info(f"Processing node {node_id} with {len(resp_image_list) if resp_image_list else 0} image(s)")
        
        if not resp_image_list:
            logger.warning(f"Node {node_id} has empty image list, skipping")
            continue

        for idx, image_bytes in enumerate(resp_image_list):
            total_images += 1
            logger.info(f"Processing image {total_images} from node {node_id} (index {idx}): {len(image_bytes)} bytes")
            
            # Validate image bytes
            if not image_bytes or len(image_bytes) == 0:
                logger.error(f"Image {total_images} from node {node_id} is empty or invalid")
                raise ValueError("Image bytes are empty or invalid")

            # Detect image format
            image_format = "png"
            if len(image_bytes) >= 8:
                if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                    image_format = "png"
                    logger.debug(f"Image {total_images}: Detected PNG format (signature match)")
                elif image_bytes[:2] == b'\xff\xd8':
                    image_format = "jpeg"
                    logger.debug(f"Image {total_images}: Detected JPEG format")
                elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
                    image_format = "webp"
                    logger.debug(f"Image {total_images}: Detected WebP format")
                else:
                    logger.warning(f"Image {total_images}: Unknown format, defaulting to PNG. First 8 bytes: {image_bytes[:8].hex()}")
            
            # FastMCP Image expects raw image bytes (not base64-encoded)
            # FastMCP handles base64 encoding internally when serializing to JSON
            logger.info(f"Image {total_images}: Using raw image bytes, size: {len(image_bytes)} bytes")
            
            # Verify image bytes are actually bytes
            if not isinstance(image_bytes, bytes):
                logger.error(f"Image {total_images}: image_bytes is not bytes, type: {type(image_bytes)}")
                raise ValueError(f"Image data must be bytes, got type {type(image_bytes)}")
            
            mcp_image = Image(data=image_bytes, format=image_format)
            logger.info(f"Image {total_images}: Created MCP Image object with format='{image_format}', data type={type(mcp_image.data)}, data length={len(mcp_image.data) if hasattr(mcp_image.data, '__len__') else 'unknown'}")
            
            result.append(mcp_image)
    
    logger.info(f"extract_images: Successfully extracted {len(result)} image(s) from {total_images} total images")
    return result


def extract_first_image(images: dict) -> Image:
    """Extract the first image from a workflow result.
    
    Args:
        images: Dictionary mapping node IDs to lists of image bytes.
    
    Returns:
        Image: The first image as an MCP Image object.
    """
    all_images = extract_images(images)
    if not all_images:
        raise ValueError("No images were generated")
    return all_images[0]


@mcp.tool()
async def text_to_image(prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: float = 8.0, denoise: float = 1.0, width: int = 1024, height: int = 1024) -> Image:
    """Generate an image from a prompt and return it in memory.
    
    Args:
        prompt: The prompt to generate the image from.  Uses natural language prompt.
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: The height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        Image: The generated image as an MCP Image object.
    """
    logger.info(f"text_to_image called with prompt='{prompt[:50]}...', seed={seed}, steps={steps}, cfg={cfg}, width={width}, height={height}")
    
    params = {"prompt": prompt, "cfg": cfg, "denoise": denoise, "width": width, "height": height}
    if seed is not None:
        params["seed"] = seed
    if steps is not None:
        params["steps"] = steps
    
    logger.info(f"Processing workflow 'text_to_image' with params: {params}")
    images = await comfyui_client.process_workflow("text_to_image", params)
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    extracted = extract_images(images)
    logger.info(f"text_to_image: Extracted {len(extracted)} MCP Image object(s), returning first image")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    img = extracted[0]
    img_data_len = len(img.data) if hasattr(img.data, '__len__') else 'unknown'
    logger.info(f"Returning image: format='{img._format}', data type={type(img.data)}, data length={img_data_len}")
    
    return img


@mcp.tool()
async def text_to_image_placeholder(prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: float = 7.0, denoise: float = 1.0, width: int = 1024, height: int = 1024) -> Image:
    """Generate a placeholder image from a prompt and return it in memory. Optimized for quick placeholder generation.
    
    Args:
        prompt: The prompt to generate the placeholder image from. Uses comma separated tags such as "dog, walking, sunset"
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: The height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        Image: The generated placeholder image as an MCP Image object.
    """
    logger.info(f"text_to_image_placeholder called with prompt='{prompt[:50]}...', seed={seed}, steps={steps}, cfg={cfg}, width={width}, height={height}")
    
    params = {"prompt": prompt, "cfg": cfg, "denoise": denoise, "width": width, "height": height}
    if seed is not None:
        params["seed"] = seed
    if steps is not None:
        params["steps"] = steps
    
    logger.info(f"Processing workflow 'text_to_image_placeholder' with params: {params}")
    workflow_images = await comfyui_client.process_workflow("text_to_image_placeholder", params)
    logger.info(f"Workflow completed, received images dict with {len(workflow_images)} node(s)")
    
    extracted = extract_images(workflow_images)
    logger.info(f"text_to_image_placeholder: Extracted {len(extracted)} MCP Image object(s), returning first image")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    img = extracted[0]
    img_data_len = len(img.data) if hasattr(img.data, '__len__') else 'unknown'
    logger.info(f"Returning image: format='{img._format}', data type={type(img.data)}, data length={img_data_len}")
    
    return img


@mcp.tool()
async def edit_image(images: list[ImageInput], prompt: str = "", seed: Optional[int] = None, steps: Optional[int] = None, cfg: float = 8.0, denoise: float = 1.0, width: int = 1024, height: int = 1024) -> Image:
    """Edit one or more images using a prompt and return the edited images in memory.
    
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
        Image: The edited image as an MCP Image object.
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
    workflow_images = await comfyui_client.process_workflow("edit_image", params)
    logger.info(f"Workflow completed, received images dict with {len(workflow_images)} node(s)")
    
    extracted = extract_images(workflow_images)
    logger.info(f"edit_image: Extracted {len(extracted)} MCP Image object(s), returning first image")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    img = extracted[0]
    img_data_len = len(img.data) if hasattr(img.data, '__len__') else 'unknown'
    logger.info(f"Returning image: format='{img._format}', data type={type(img.data)}, data length={img_data_len}")
    
    return img


@mcp.tool()
async def run_workflow_from_file(file_path: str = "") -> Image:
    """Run a workflow from a file and return the generated image in memory.
    
    Args:
        file_path: The absolute path to the file to run.
    
    Returns:
        Image: The generated image as an MCP Image object.
    """
    logger.info(f"run_workflow_from_file called with file_path='{file_path}'")
    
    with open(file_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    logger.info(f"Loaded workflow from file, processing...")
    # Get images as bytes
    images = await comfyui_client.process_workflow(workflow, {})
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    extracted = extract_images(images)
    logger.info(f"run_workflow_from_file: Extracted {len(extracted)} MCP Image object(s), returning first image")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    img = extracted[0]
    img_data_len = len(img.data) if hasattr(img.data, '__len__') else 'unknown'
    logger.info(f"Returning image: format='{img._format}', data type={type(img.data)}, data length={img_data_len}")
    
    return img


@mcp.tool()
async def run_workflow_from_json(json_data: Optional[dict] = None) -> Image:
    """Run a workflow from JSON data and return the generated image in memory.
    
    Args:
        json_data: The JSON workflow to run.
    
    Returns:
        Image: The generated image as an MCP Image object.
    """
    logger.info(f"run_workflow_from_json called with json_data={'provided' if json_data else 'None'}")
    
    if json_data is None:
        json_data = {}

    logger.info(f"Processing workflow from JSON...")
    images = await comfyui_client.process_workflow(json_data, {})
    logger.info(f"Workflow completed, received images dict with {len(images)} node(s)")
    
    extracted = extract_images(images)
    logger.info(f"run_workflow_from_json: Extracted {len(extracted)} MCP Image object(s), returning first image")
    
    if not extracted:
        raise ValueError("No images were generated")
    
    img = extracted[0]
    img_data_len = len(img.data) if hasattr(img.data, '__len__') else 'unknown'
    logger.info(f"Returning image: format='{img._format}', data type={type(img.data)}, data length={img_data_len}")
    
    return img


if __name__ == "__main__":
    mcp.run(transport=settings.transport.value)
