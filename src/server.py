import os
import json
from client.comfyui import ComfyUI
from mcp.server.fastmcp import FastMCP
from fastmcp.utilities.types import Image
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("comfyui")

def extract_first_image(images: dict) -> Image:
    """Extract the first image from a workflow result.
    
    Args:
        images: Dictionary mapping node IDs to lists of image bytes.
        
    Returns:
        Image: The first image as an in-memory Image object.
        
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
    
    # Return as Image type
    return Image(data=image_bytes, mimeType="image/png")

@mcp.tool()
async def text_to_image(prompt: str, seed: int, steps: int, cfg: float, denoise: float, width: int = 1024, height: int = 1024) -> Image:
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
        Image: The generated image as an in-memory Image object.
    """
    auth = os.environ.get("COMFYUI_AUTHENTICATION")
    comfy = ComfyUI(
        url=f'http://{os.environ.get("COMFYUI_HOST", "localhost")}:{os.environ.get("COMFYUI_PORT", 8188)}',
        authentication=auth
    )
    # Get images as bytes
    images = await comfy.process_workflow("text_to_image", {"prompt": prompt, "seed": seed, "steps": steps, "cfg": cfg, "denoise": denoise, "width": width, "height": height})
    
    return extract_first_image(images)

@mcp.tool()
async def run_workflow_from_file(file_path: str) -> Image:
    """Run a workflow from a file and return the generated image in memory.
    
    Args:
        file_path: The absolute path to the file to run.
    
    Returns:
        Image: The generated image as an in-memory Image object.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    
    auth = os.environ.get("COMFYUI_AUTHENTICATION")
    comfy = ComfyUI(
        url=f'http://{os.environ.get("COMFYUI_HOST", "localhost")}:{os.environ.get("COMFYUI_PORT", 8188)}',
        authentication=auth
    )
    # Get images as bytes
    images = await comfy.process_workflow(workflow, {})
    return extract_first_image(images)

@mcp.tool()
async def run_workflow_from_json(json_data: dict) -> Image:
    """Run a workflow from JSON data and return the generated image in memory.
    
    Args:
        json_data: The JSON data to run.
    
    Returns:
        Image: The generated image as an in-memory Image object.
    """
    workflow = json_data
    
    auth = os.environ.get("COMFYUI_AUTHENTICATION")
    comfy = ComfyUI(
        url=f'http://{os.environ.get("COMFYUI_HOST", "localhost")}:{os.environ.get("COMFYUI_PORT", 8188)}',
        authentication=auth
    )
    # Get images as bytes
    images = await comfy.process_workflow(workflow, {})
    return extract_first_image(images)

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    # Map "http" to "sse" for backward compatibility (FastMCP uses "sse" for HTTP-like transports)
    if transport == "http":
        transport = "sse"
    mcp.run(transport=transport)
