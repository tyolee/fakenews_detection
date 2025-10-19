import os
import pandas as pd
import urllib.request

# Image Downloader
def download_image(row, img_dir):
    """
    Attempts to download the image from the URL and saves it using the post ID.
    Returns True if successful, False otherwise.
    """
    if row.get('hasImage') != True:
        return False
        
    img_url = row.get('image_url')
    img_id = row['id']
    img_path = os.path.join(img_dir, f"{img_id}.jpg")
    
    # Skip if the image file already exists
    if os.path.exists(img_path):
        return True

    if pd.isna(img_url) or not img_url:
        return False
        
    try:
        # Set a short timeout and user-agent to mimic a browser
        req = urllib.request.Request(
            img_url, 
            data=None, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        # Download and save the image
        with urllib.request.urlopen(req, timeout=10) as url:
            with open(img_path, 'wb') as f:
                f.write(url.read())
        return True
    except Exception as e:
        # Catch all common exceptions (Timeout, HTTPError, URLError, etc.)
        #print(e)
        return False