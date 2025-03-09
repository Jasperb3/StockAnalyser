import os
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

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

def create_message_with_inline_image(sender: str, to: str, subject: str, markdown_text: str) -> dict:
    """
    Create an email message with inline images using the Content-ID (CID) method.
    Converts markdown to HTML, replaces markdown image references with HTML <img> tags, and attaches images inline.
    """
    # Start by processing the markdown text for image references
    image_references = []
    lines = markdown_text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('![') and '](' in line and line.endswith(')'):
            # Extract the alt text and image filename
            alt_text = line[2:line.index(']')]
            img_filename = line[line.index('(')+1:-1]

            # Create a unique CID *before* we modify the line.  Crucially,
            # use a consistent, predictable naming scheme based on the
            # *order* of the images in the markdown. Using `len(image_references)`
            # inside the loop *could* cause issues if the list is modified
            # during iteration, though it's unlikely here. Using the index `i`
            # is safer and more directly reflects the image's position.
            cid = f'Image_{i}'

            # Replace the markdown image with an HTML <img> tag referencing the CID
            image_html = (
                f'<img src="cid:{cid}" alt="{alt_text}" '
                f'style="max-width: 600px; width: 100%; height: auto; display: block; margin: 20px auto;">'
            )
            lines[i] = image_html

            # Store the image reference for later attachment.  Store the *index*
            # as part of the reference to be absolutely sure of the order.
            image_references.append({
                'filename': img_filename,
                'cid': cid,
                'alt_text': alt_text,
                'index': i  # Store the original index
            })

    # Reassemble the modified markdown text and convert it to HTML
    markdown_text = '\n'.join(lines)
    import markdown
    md = markdown.Markdown(extensions=['tables', 'fenced_code'])
    html_content = md.convert(markdown_text)

    # Wrap the converted HTML in a full HTML document with inline CSS
    html = f"""\
<html>
  <head>
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

    # Create a multipart/alternative container for the plain-text and HTML parts
    alternative_part = MIMEMultipart("alternative")
    # Provide a plain-text fallback
    plain_text = "This email contains HTML content. Please view in a HTML-supporting email client."
    alternative_part.attach(MIMEText(plain_text, "plain"))
    alternative_part.attach(MIMEText(html, "html"))

    # Create the outer multipart/related container and attach the alternative part
    message = MIMEMultipart("related")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = to
    message.attach(alternative_part)

    # Attach each image as an inline MIMEImage part. Sort by the stored index
    # to guarantee the correct order, even if something unexpected happens
    # during list processing.

    for img_ref in sorted(image_references, key=lambda x: x['index']):
        # use the filename from markdown
        img_path = img_ref['filename']

        try:
            with open(img_path, 'rb') as fp:
                img_data = fp.read()
            image = MIMEImage(img_data)
            # Set the Content-ID header so that the HTML can reference it (must be enclosed in angle brackets)
            image.add_header('Content-ID', f'<{img_ref["cid"]}>')
            # Set Content-Disposition to inline so that email clients treat it as an inline image
            image.add_header('Content-Disposition', 'inline', filename=os.path.basename(img_path))
            message.attach(image)
        except FileNotFoundError:
            print(f"Warning: Could not find image file: {img_path}")
            continue

    # Encode the entire message in base64url for the Gmail API
    import base64
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    return {'raw': raw_message}


def create_draft(service, user_id: str, message_body: dict) -> dict:
    """
    Create and insert a draft email using the Gmail API.
    
    Args:
        service: Authorized Gmail API service instance
        user_id: User's email address or 'me'
        message_body: Dict containing the raw base64url encoded message
    
    Returns:
        Created draft object or None if error occurs
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