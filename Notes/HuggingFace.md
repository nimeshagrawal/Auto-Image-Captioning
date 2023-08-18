# HuggingFace (HF)

1. Create a HF repo/space with your desinged template [streamlit/gradio/blank-docker]
2. There are issues in GitHub actions sync with HF-space (LFS not detected), so to bypass that, We clone HF space to system, transfer ONLY the inference files and web-app.py file and push to the designated HF-space.
3. We particularly have to EXPOSE port 7860 within the docker file.
4. Every push triggers a docker build and effectively runs the web-app.