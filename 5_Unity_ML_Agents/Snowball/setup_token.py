import os
from huggingface_hub import HfFolder, whoami

HfFolder.save_token(os.environ["TOKEN"])
print(whoami())
