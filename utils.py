# ==============================================================================
# Helper Functions
# ==============================================================================

def download_pretrained_weights_manually():
    """
    Instructions for manually downloading YOLO-NAS pretrained weights.
    Use this if automatic download fails due to network issues.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  Manual Download Instructions for YOLO-NAS Pretrained Weights  ║
    ╚════════════════════════════════════════════════════════════════╝
    
    If automatic download fails, follow these steps:
    
    1. Download the pretrained weights manually:
       URL: https://sghub.deci.ai/models/yolo_nas_s_coco.pth
       
    2. Save the file to:
       C:\\Users\\quang\\.cache\\torch\\hub\\checkpoints\\yolo_nas_s_coco.pth
       
    3. Create the directory if it doesn't exist:
       mkdir "C:\\Users\\quang\\.cache\\torch\\hub\\checkpoints"
       
    4. Alternative: Use a download manager or wget:
       wget https://sghub.deci.ai/models/yolo_nas_s_coco.pth
       
    5. Or download via Python (if you have internet access):
       import requests
       url = "https://sghub.deci.ai/models/yolo_nas_s_coco.pth"
       r = requests.get(url)
       with open("yolo_nas_s_coco.pth", "wb") as f:
           f.write(r.content)
    
    After downloading, run the script again.
    
    Alternatively, you can train from scratch without pretrained weights
    by setting use_pretrained=False (this will take longer to converge).
    """)