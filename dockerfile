# 공통 베이스 이미지
FROM runpod/worker-comfyui:5.5.0-base

# ----- 커스텀 노드 설치 (두 워크플로우에서 사용하는 목록 합집합) -----
RUN comfy node install --exit-on-fail comfyui-easy-use@1.3.4 \
 && comfy node install --exit-on-fail comfyui_ultimatesdupscale@1.6.0 \
 && comfy node install --exit-on-fail comfyui-kjnodes@1.1.9 \
 && comfy node install --exit-on-fail rgthree-comfy@1.0.2511111955 \
 && comfy node install --exit-on-fail comfyui-dream-project@5.1.2 \
 && comfy node install --exit-on-fail comfyui-impact-pack@8.28.0 \
 && comfy node install --exit-on-fail ComfyUI_XISER_Nodes@1.3.1 \
 && comfy node install --exit-on-fail ComfyUI_LayerStyle_Advance \
 && comfy node install --exit-on-fail comfyui-custom-scripts@1.2.5 \
 && comfy node install --exit-on-fail mikey_nodes@1.0.5 \
 && comfy node install --exit-on-fail comfyui_essentials@1.1.0 \
 && comfy node install --exit-on-fail was-node-suite-comfyui@1.0.2

# ----- 모델 다운로드 -----
RUN mkdir -p /comfyui/models/{unet,vae,upscale_models,text_encoders,text_encoders/t5,long_clip}
RUN comfy model download --url https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth \
        --relative-path models/upscale_models --filename 8x_NMKD-Superscale_150000_G.pth \
 && comfy model download --url https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/resolve/main/ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors \
        --relative-path models/text_encoders/long_clip --filename ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors \
 && comfy model download --url https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors \
        --relative-path models/text_encoders --filename t5xxl_fp8_e4m3fn_scaled.safetensors \
 && comfy model download --url https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors \
        --relative-path models/vae --filename ae.safetensors

RUN python -m pip install --no-cache-dir gdown \
 && gdown --quiet --id 1pjmbNQeyViD67IxP9EfRuPqIMsMF5PaT \
        -O /comfyui/models/unet/flux1-dev-kontext.safetensors

# ----- 핸들러/워크플로 파일 포함 -----
WORKDIR /opt/flux
COPY handler_basic.py ./handler_basic.py
COPY CTY_FLUX_KONTEXT_V2.json ./CTY_FLUX_KONTEXT_V2.json
COPY handler_backplate.py ./handler_backplate.py
COPY CTY_FLUX_KONTEXT_W_BACKPLATE.json ./CTY_FLUX_KONTEXT_W_BACKPLATE.json
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 기본 워크플로 선택 값 (필요 시 Runpod 템플릿에서 override)
ENV WORKFLOW_TARGET=basic

CMD ["/entrypoint.sh"]