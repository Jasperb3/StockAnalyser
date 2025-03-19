import os
import base64
import markdown
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from markdown.inlinepatterns import InlineProcessor
from markdown.extensions import Extension
import xml.etree.ElementTree as etree

SCOPES = ['https://www.googleapis.com/auth/gmail.compose']

def authenticate_gmail():
    """
    Authenticate and return an authorized Gmail API service instance.
    Expects credentials.json and token.json in the same directory as this script.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(current_dir, 'token.json')
    credentials_path = os.path.join(current_dir, 'credentials.json')
    
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"credentials.json not found at {credentials_path}. "
                    "Download OAuth 2.0 credentials from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    service = build('gmail', 'v1', credentials=creds)
    return service

# Custom Markdown inline processor to handle images.
class InlineImageProcessor(InlineProcessor):
    def __init__(self, pattern, md, image_list):
        super().__init__(pattern, md)
        self.image_list = image_list

    def handleMatch(self, m, data):
        alt_text = m.group(1)
        img_src = m.group(2)
        # Create a unique CID using the current count; note the angle brackets will be added later in headers.
        cid = f'Image_{len(self.image_list)}'
        self.image_list.append({
            'cid': cid,
            'filename': img_src,
            'alt_text': alt_text
        })
        # Instead of returning a bare <img> tag, wrap it in a <div> (block element)
        container = etree.Element("div")
        container.set("style", "display:block;")
        img = etree.SubElement(container, "img")
        # Set the src with angle brackets around the CID (Gmail requires them in the MIME part)
        img.set("src", f"cid:{cid}")
        img.set("alt", alt_text)
        # Optionally add inline styles to force size if needed:
        img.set("style", "max-width:600px;width:100%;height:auto;margin:20px auto;display:block;")
        return container, m.start(0), m.end(0)

class ImageExtension(Extension):
    def __init__(self, **kwargs):
        self.image_list = []
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        # Standard Markdown image pattern: ![alt](src)
        IMAGE_RE = r'!\[([^\]]*)\]\(([^)]+)\)'
        md.inlinePatterns.register(InlineImageProcessor(IMAGE_RE, md, self.image_list), 'custom_image', 175)

def convert_markdown_to_html_with_images(markdown_text, image_list):
    """
    Convert Markdown to HTML while using a custom extension to process images.
    The extension will populate `image_list` with image details in the exact order encountered.
    """
    image_ext = ImageExtension()
    html = markdown.markdown(markdown_text, extensions=[image_ext, 'fenced_code', 'tables'])
    image_list.extend(image_ext.image_list)
    return html

def create_message_with_inline_image(sender: str, to: str, subject: str, markdown_text: str) -> dict:
    """
    Create an email with inline images using a custom Markdown extension.
    This version wraps each image in a block <div> to help preserve order in Gmail.
    """
    # This list will be populated in order by the Markdown processor
    image_references = []
    html_content = convert_markdown_to_html_with_images(markdown_text, image_references)

    # Wrap the content in a full HTML document
    full_html = f"""\
<html>
  <head>
    <meta charset="utf-8">
    <title>{subject}</title>
    <style>
      body {{
          font-family: Arial, sans-serif;
          line-height: 1.6;
          color: #333;
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
      }}
      h1 {{
          color: #2c3e50;
          border-bottom: 2px solid #eee;
          padding-bottom: 10px;
      }}
    </style>
  </head>
  <body>
    {html_content}
  </body>
</html>
"""

    # Build the multipart/related MIME message
    msg = MIMEMultipart('related')
    msg['From'] = sender
    msg['To'] = to
    msg['Subject'] = subject

    # Create the alternative part with plain text and HTML
    alternative_part = MIMEMultipart('alternative')
    plain_text = "This email contains HTML content. Please view it in a HTML-supporting email client."
    alternative_part.attach(MIMEText(plain_text, 'plain'))
    alternative_part.attach(MIMEText(full_html, 'html'))
    msg.attach(alternative_part)

    # Attach the images in the exact order collected
    for img_ref in image_references:
        img_path = img_ref['filename']
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} not found.")
            continue
        with open(img_path, 'rb') as f:
            img_data = f.read()
        mime_img = MIMEImage(img_data)
        # Set the Content-ID wrapped in angle brackets
        mime_img.add_header('Content-ID', f'<{img_ref["cid"]}>')
        mime_img.add_header('Content-Disposition', 'inline', filename=os.path.basename(img_path))
        msg.attach(mime_img)

    # Encode message for sending via Gmail API (base64url encoded)
    raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
    return {'raw': raw_message}


def create_draft(service, user_id: str, message_body: dict) -> dict:
    """
    Create and insert a draft email using the Gmail API.
    """
    try:
        draft = service.users().drafts().create(
            userId=user_id, 
            body={'message': message_body}
        ).execute()
        print(f"Draft id: {draft['id']}")
        return draft
    except Exception as error:
        print(f"An error occurred while creating draft: {error}")
        return None

# Example usage:
if __name__ == '__main__':
    sender = os.getenv("CLIENT_EMAIL")
    recipient = os.getenv("CLIENT_EMAIL")
    subject = "Test Email with Inline Images"
    with open("/home/j/ai/crewAI/finance/stock_analyser/final_reports/2025-03-12/MSFT_Stock_Analysis_Report_20250312_132728.md", "r") as f:
        markdown_body = f.read()
    message_body = create_message_with_inline_image(sender, recipient, subject, markdown_body)
    # Now use the Gmail API to send `message_body`
    # For example: service.users().messages().send(userId="me", body=message_body).execute()
    print("Message created. (Use Gmail API to send)")
