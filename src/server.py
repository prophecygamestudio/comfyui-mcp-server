import json
import base64
from enum import Enum

from mcp.types import ImageContent
from pydantic import Field

from client.comfyui import ComfyUI
from fastmcp import FastMCP
from pydantic_settings import BaseSettings, SettingsConfigDict


dotenv_file = ".env"


class MCPTransport(str, Enum):
    stdio = "stdio"
    sse = "sse"
    http = "http"


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="mcp_", env_file=dotenv_file, extra='ignore')

    host: str = "0.0.0.0"
    port: int = 8000
    use_image_data_uri: bool = False
    transport: MCPTransport = MCPTransport.http


settings = ServerSettings()
mcp = FastMCP("comfyui", host=settings.host, port=settings.port, stateless_http=True)


class ComfyUISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="comfyui_", env_file=dotenv_file, extra='ignore')

    host: str = Field("localhost", description="The hostname or IP address of the ComfyUI server.")
    port: int = Field(8188, description="The port number of the ComfyUI server.")
    authentication: str | None = Field(None, description="The authentication token for the ComfyUI server, if required.")


def get_comfyui_client() -> ComfyUI:
    """Create and return a ComfyUI client based on environment variables."""
    comfy_settings = ComfyUISettings()
    return ComfyUI(
        url=f'http://{comfy_settings.host}:{comfy_settings.port}',
        authentication=comfy_settings.authentication,
    )


def extract_first_image(images: dict) -> ImageContent:
    """Extract the first image from a workflow result.
    
    Args:
        images: Dictionary mapping node IDs to lists of image bytes.
        
    Returns:
        ImageContent: The first image as an MCP ImageContent object.
        
    Raises:
        ValueError: If no images are found in the workflow output.
    """
    if not images:
        raise ValueError("No images were generated")
    
    # Get the first image from the first node
    first_node_id = next(iter(images.keys()))
    image_bytes_list = images[first_node_id]
    if not image_bytes_list:
        raise ValueError("No image data found in the workflow output")
    
    # Get the first image bytes
    image_bytes = image_bytes_list[0]
    
    # Validate image bytes
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Image bytes are empty or invalid")
    
    # Verify it's a valid PNG (PNG files start with PNG signature: 89 50 4E 47)
    if len(image_bytes) < 8 or image_bytes[:8] != b'\x89PNG\r\n\x1a\n':
        # Not a PNG, but continue anyway - might be JPEG or other format
        # Try to detect format or default to PNG
        pass
    
    # Encode image bytes as base64
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    
    # Validate base64 encoding
    if not base64_data:
        raise ValueError("Failed to encode image as base64")
    
    # Clean base64 string (remove any whitespace only - don't remove valid base64 chars)
    base64_data = base64_data.strip().replace('\n', '').replace('\r', '')
    
    # Validate base64 string format by trying to decode it
    try:
        decoded = base64.b64decode(base64_data, validate=True)
        if len(decoded) != len(image_bytes):
            raise ValueError("Base64 decode length mismatch")
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")
    
    # Some MCP clients expect just the base64 string, not the full data URI
    if settings.use_image_data_uri:
        image_data = f"data:image/png;base64,{base64_data}"
    else:
        image_data = base64_data
    
    image_content = ImageContent(
        type="image",
        data=image_data,
        mimeType="image/png"
    )
    
    return image_content


@mcp.tool()
async def text_to_image(prompt: str, seed: int, steps: int, cfg: float, denoise: float, width: int = 1024, height: int = 1024) -> ImageContent:
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
    images = await get_comfyui_client().process_workflow("text_to_image", {"prompt": prompt, "seed": seed, "steps": steps, "cfg": cfg, "denoise": denoise, "width": width, "height": height})
    return extract_first_image(images)


@mcp.tool()
async def run_workflow_from_file(file_path: str) -> ImageContent:
    """Run a workflow from a file and return the generated image in memory.
    
    Args:
        file_path: The absolute path to the file to run.
    
    Returns:
        ImageContent: The generated image as an MCP ImageContent object.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Get images as bytes
    images = await get_comfyui_client().process_workflow(workflow, {})
    return extract_first_image(images)


@mcp.tool()
async def run_workflow_from_json(json_data: dict) -> ImageContent:
    """Run a workflow from JSON data and return the generated image in memory.
    
    Args:
        json_data: The JSON data to run.
    
    Returns:
        ImageContent: The generated image as an MCP ImageContent object.
    """
    workflow = json_data
    
    images = await get_comfyui_client().process_workflow(workflow, {})
    return extract_first_image(images)


if __name__ == "__main__":
    mcp.run(transport=settings.transport.value)
