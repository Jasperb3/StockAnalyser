from typing import Type, List, Literal, Optional, Dict
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from contextlib import contextmanager
import os
import time
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
import json

# Constants
CHROME_BINARY_PATH = "/usr/bin/chromium-browser"
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"

class WebElement(BaseModel):
    element_type: Literal['text', 'image', 'link', 'button', 'form', 'navigation', 'other']
    content: str
    location: str
    confidence: float
    attributes: Dict[str, str] = {}
    full_text: Optional[str] = None

class WebpageDescription(BaseModel):
    summary: str
    layout: str
    main_content: List[WebElement]
    navigation: List[WebElement]
    interactive_elements: List[WebElement]
    media_elements: List[WebElement]
    color_scheme: List[str]
    accessibility_notes: Optional[str] = None
    branding_elements: Optional[List[WebElement]] = None
    page_type: Literal['Homepage', 'Article', 'Product', 'Form', 'Landing Page', 'Other']
    language: str
    metadata: Dict[str, str] = {}

class WebpageDescriptionInput(BaseModel):
    """Input schema for WebpageDescription."""
    url: str = Field(..., description="The URL of the webpage to analyze")
    screenshots_dir: str = Field(
        default=str(Path(__file__).parent.parent.parent.parent / "screenshots"),
        description="Directory to save the screenshot. Default path is [project_root]/screenshots"
    )

@contextmanager
def get_driver(window_size=(1920, 1080)):
    """Context manager for handling driver creation and cleanup"""
    chrome_options = Options()
    chrome_options.binary_location = CHROME_BINARY_PATH
    
    # Core options
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Fix for DevTools port issues
    chrome_options.add_argument('--remote-debugging-pipe')  # Use pipe instead of port
    chrome_options.add_argument('--disable-gpu')
    
    # Additional stability options
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-software-rasterizer')
    chrome_options.add_argument('--disable-features=VizDisplayCompositor')
    
    # Create a temporary directory for Chrome
    tmp_dir = '/tmp/chrome-data'
    os.makedirs(tmp_dir, exist_ok=True)
    chrome_options.add_argument(f'--user-data-dir={tmp_dir}')
    
    service = Service(
        executable_path=CHROMEDRIVER_PATH,
        service_args=['--verbose', '--log-path=/tmp/chromedriver.log']
    )
    
    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_window_size(*window_size)
        yield driver
    finally:
        if driver:
            try:
                driver.close()
                driver.quit()
            except Exception as e:
                print(f"Error during driver cleanup: {e}")
            # Cleanup temp directory
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")

class WebpageDescriptionTool(BaseTool):
    name: str = "Webpage Description Tool"
    description: str = (
        "Pass this tool a URL as input and it will capture a screenshot and provide a comprehensive analysis "
        "of the webpage including layout, content, navigation, and visual design aspects."
    )
    args_schema: Type[BaseModel] = WebpageDescriptionInput

    def _take_screenshot(self, url: str, save_path: str) -> tuple[bool, str]:
        """Takes a screenshot of the webpage and returns success status and file path"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"webpage_{timestamp}.png"
                full_path = os.path.join(save_path, filename)
                
                print(f"Taking screenshot of {url} (attempt {attempt + 1}/{max_retries})")
                with get_driver() as driver:
                    driver.get(url)
                    time.sleep(5)  # Wait for page load

                    # Try multiple common patterns for cookie consent buttons
                    consent_button_patterns = [
                        "//button[contains(translate(., 'ACCEPT', 'accept'), 'accept')]",
                        "//button[contains(translate(., 'AGREE', 'agree'), 'agree')]",
                        "//button[contains(@class, 'consent')]",
                        "//button[contains(@id, 'consent')]",
                        "//div[contains(@class, 'consent')]//button",
                        "//div[contains(@id, 'consent')]//button",
                        "//iframe[contains(@title, 'consent')]"
                    ]

                    for pattern in consent_button_patterns:
                        try:
                            # Switch to iframe if it exists
                            if 'iframe' in pattern:
                                iframe = WebDriverWait(driver, 2).until(
                                    EC.presence_of_element_located((By.XPATH, pattern))
                                )
                                driver.switch_to.frame(iframe)
                                # Reset pattern to look for button in iframe
                                pattern = "//button[contains(translate(., 'ACCEPT', 'accept'), 'accept')]"

                            consent_button = WebDriverWait(driver, 2).until(
                                EC.element_to_be_clickable((By.XPATH, pattern))
                            )
                            consent_button.click()
                            if 'iframe' in pattern:
                                driver.switch_to.default_content()
                            print("Successfully clicked cookie consent button")
                            break
                        except TimeoutException:
                            if 'iframe' in pattern:
                                driver.switch_to.default_content()
                            continue
                        except Exception as e:
                            print(f"Error clicking consent button: {str(e)}")
                            continue
                    else:
                        print("No cookie consent popup found, continuing...")
                    
                    # Set window size to full page height
                    total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
                    driver.set_window_size(1920, total_height)
                    time.sleep(2)  # Wait for resize
                    
                    driver.save_screenshot(full_path)
                    print(f"Screenshot saved to {full_path}")
                    return True, full_path

            except Exception as e:
                print(f"Screenshot error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False, str(e)

    def _analyze_screenshot(self, image_path: str, url: str) -> WebpageDescription:
        """Analyzes the screenshot using Gemini and returns a structured description"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Screenshot not found at {image_path}")

        print(f"Analyzing screenshot at {image_path}")
        
        # Configure Gemini
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
        # Setup model with lower temperature for more consistent results
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )

        try:
            # Load and analyze image
            image = genai.upload_file(image_path, mime_type="image/png")
            
            # First get a raw description
            response = model.generate_content([
                "Analyze this webpage screenshot and describe what you see in detail.",
                image
            ])
            
            # Then structure it into our format
            structured_prompt = (
                f"Based on this webpage screenshot analysis:\n{response.text}\n\n"
                "Generate a JSON response following this exact schema:\n"
                f"{json.dumps(WebpageDescription.model_json_schema(), indent=2)}"
            )
            
            json_response = model.generate_content(structured_prompt)
            
            # Clean up the response text to extract just the JSON
            response_text = json_response.text
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                # Find the first and last ``` and extract content between them
                start_idx = response_text.find("\n", response_text.find("```")) + 1
                end_idx = response_text.rfind("```")
                response_text = response_text[start_idx:end_idx].strip()
            
            # Remove any "json" or other language identifiers that might be present
            if response_text.startswith("json\n"):
                response_text = response_text[5:].strip()
            
            # Parse and validate the cleaned JSON
            try:
                webpage_description = WebpageDescription.model_validate_json(response_text)
                return webpage_description
            except Exception as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Cleaned response text: {response_text}")
                raise
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            raise

    def _run(self, url: str, screenshots_dir: str = "./screenshots") -> WebpageDescription:
        """Main execution method that coordinates screenshot capture and analysis"""
        # Ensure directories exist with proper permissions
        for directory in [screenshots_dir, '/tmp/chrome-data']:
            try:
                os.makedirs(directory, mode=0o755, exist_ok=True)
            except Exception as e:
                raise Exception(f"Failed to create directory {directory}: {str(e)}")

        # Validate chrome and chromedriver paths
        for path, name in [(CHROME_BINARY_PATH, "Chromium"), (CHROMEDRIVER_PATH, "ChromeDriver")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at {path}")
        """Main execution method that coordinates screenshot capture and analysis"""
        # Validate chrome and chromedriver paths
        for path, name in [(CHROME_BINARY_PATH, "Chromium"), (CHROMEDRIVER_PATH, "ChromeDriver")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at {path}")

        # Take screenshot
        success, result = self._take_screenshot(url, screenshots_dir)
        if not success:
            raise Exception(f"Failed to take screenshot: {result}")

        # Analyze screenshot
        try:
            description = self._analyze_screenshot(result, url)
            return description
        except Exception as e:
            raise Exception(f"Failed to analyze screenshot: {str(e)}")
        
if __name__ == "__main__":
    tool = WebpageDescriptionTool()
    result = tool.run("https://www.google.com")
    print(result)
