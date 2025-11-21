import json
import base64
from enum import Enum
from typing import Optional

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
    transport: MCPTransport = MCPTransport.http


settings = ServerSettings()
mcp = FastMCP("comfyui", host=settings.host, port=settings.port, stateless_http=True)
comfyui_client = ComfyUI()


def extract_images(images: dict) -> list[Image]:
    """Extract the images from a workflow result.

    Args:
        images: Dictionary mapping node IDs to lists of image bytes.

    Returns:
        list[Image]: The images as an MCP Image objects.
    """
    if not images:
        raise ValueError("No images were generated")

    result = []
    for resp_image_list in images.values():
        if not resp_image_list:
            continue

        for image_bytes in resp_image_list:
            # Validate image bytes
            if not image_bytes or len(image_bytes) == 0:
                raise ValueError("Image bytes are empty or invalid")

            # Verify it's a valid PNG (PNG files start with PNG signature: 89 50 4E 47)
            if len(image_bytes) < 8 or image_bytes[:8] != b'\x89PNG\r\n\x1a\n':
                # Not a PNG, but continue anyway - might be JPEG or other format
                # Try to detect format or default to PNG
                pass

            # Encode image bytes as base64
            base64_data = base64.b64encode(image_bytes)
            result.append(Image(data=base64_data, format="png"))
    
    return result


@mcp.tool()
async def text_to_image(prompt: str = "", seed: int = 0, steps: int = 20, cfg: float = 8.0, denoise: float = 1.0, width: int = 1024, height: int = 1024) -> list[Image]:
    """Generate an image from a prompt and return it in memory.
    
    Args:
        prompt: The prompt to generate the image from.  Uses SDXL prompting style.
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        cfg: The CFG scale to use for the image generation.
        denoise: The denoise strength to use for the image generation.
        width: The width of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
        height: The height of the generated image in pixels. Best results are at approximately 1 megapixel (e.g., 1024x1024).
    
    Returns:
        ImageContent: The generated image as an MCP ImageContent object.
    """
    images = await comfyui_client.process_workflow("text_to_image", {"prompt": prompt, "seed": seed, "steps": steps, "cfg": cfg, "denoise": denoise, "width": width, "height": height})
    return extract_images(images)


@mcp.tool()
async def run_workflow_from_file(file_path: str = "") -> list[Image]:
    """Run a workflow from a file and return the generated image in memory.
    
    Args:
        file_path: The absolute path to the file to run.
    
    Returns:
        ImageContent: The generated image as an MCP ImageContent object.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Get images as bytes
    images = await comfyui_client.process_workflow(workflow, {})
    return extract_images(images)


@mcp.tool()
async def run_workflow_from_json(json_data: Optional[dict] = None) -> list[Image]:
    """Run a workflow from JSON data and return the generated image in memory.
    
    Args:
        json_data: The JSON workflow to run.
    
    Returns:
        ImageContent: The generated image as an MCP ImageContent object.
    """
    if json_data is None:
        json_data = {}

    images = await comfyui_client.process_workflow(json_data, {})
    return extract_images(images)


if __name__ == "__main__":
    mcp.run(transport=settings.transport.value)
