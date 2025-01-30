import os
import random
import string
import time
from dashscope import Generation, MultiModalConversation
from http import HTTPStatus
import requests
import torchvision.transforms as transforms
import torchvision.io as io
import torch
import torchaudio
import torchvision
import folder_paths
import numpy as np


def generate_random_image_name():
    random_number = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    return f"image_{random_number}.png"


def generate_random_video_name():
    random_number = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    return f"video_{random_number}.mp4"


def generate_random_audio_name():
    random_number = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    return f"audio_{random_number}.wav"


def get_image_url(image):
    image = image.squeeze(0)
    image = image.permute(2, 0, 1)
    image = transforms.ToPILImage()(image)

    image_name = generate_random_image_name()
    save_path = os.path.join(folder_paths.get_output_directory(), image_name)
    torchvision.utils.save_image(image, save_path)

    if os.path.exists(image_name):
        os.remove(image_name)

    return image_name


def get_audio_url(audio):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)

    audio_name = generate_random_audio_name()
    save_path = os.path.join(folder_paths.get_output_directory(), audio_name)
    torchaudio.save(save_path, waveform, sample_rate, format="wav")

    if os.path.exists(audio_name):
        os.remove(audio_name)

    return audio_name


def get_model_versions(model_type: str) -> list[str]:
    model_versions = []
    current_group = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_versions_file = os.path.join(
        current_dir, "model_versions", f"{model_type}.txt"
    )

    with open(model_versions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_group = line[1:-1]
                continue

            model_versions.append(line)

    return model_versions


VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov"]
MAX_VIDEO_SIZE_MB = 100


def video_to_images(video_path: str, fps: float = None) -> tuple:
    """将视频转换为图像序列，支持多种格式和内存优化"""

    if not os.path.exists(video_path):
        raise RuntimeError(f"视频文件不存在: {video_path}")

    if not any(video_path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
        raise RuntimeError(f"不支持的视频格式，支持: {", ".join(VIDEO_EXTENSIONS)}")

    if os.path.getsize(video_path) > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise RuntimeError(f"视频文件过大，最大支持 {MAX_VIDEO_SIZE_MB}MB")

    try:
        # 读取视频
        frames, audio_frames, metadata = io.read_video(video_path, pts_unit="sec")

        if frames.shape[0] == 0:
            raise RuntimeError("视频帧为空")

        # 获取信息
        frame_count = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]
        source_fps = metadata.get("video", {}).get("fps", 30.0)
        duration = frame_count / source_fps

        # 调整帧率(如果指定)
        if fps and fps > 0 and fps != source_fps:
            target_frames = int(duration * fps)
            indices = torch.linspace(0, frame_count - 1, target_frames).long()
            frames = frames[indices]
            frame_count = len(frames)

        # 格式转换
        frames = frames.float() / 255.0
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        frames = frames.permute(0, 3, 1, 2)

        # 返回信息
        video_info = {
            "source_fps": source_fps,
            "source_frame_count": frame_count,
            "source_duration": duration,
            "source_width": width,
            "source_height": height,
            "target_fps": fps if fps else source_fps,
        }

        return (frames, frame_count, video_info)

    except Exception as e:
        raise RuntimeError(f"视频转换失败: {str(e)}")


class DashscopeLLMLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_versions = get_model_versions("llm")
        return {
            "required": {
                "model_version": (
                    model_versions,
                    {"default": model_versions[0]},
                ),
            }
        }

    FUNCTION = "select_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def select_model(self, model_version) -> tuple[str]:
        if not model_version:
            raise ValueError("Model version cannot be empty")
        return (model_version,)


class DashscopeVLMLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_versions = get_model_versions("vlm")
        return {
            "required": {
                "model_version": (
                    model_versions,
                    {"default": model_versions[0]},
                ),
            }
        }

    FUNCTION = "select_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def select_model(self, model_version) -> tuple[str]:
        if not model_version:
            raise ValueError("Model version cannot be empty")
        return (model_version,)


# 关于OCR模型的最小像素值，根据阿里云文字识别服务的要求，图片的最短边不小于15像素，单字大小在10-50像素内时，识别效果较好。因此，28像素并不是OCR模型的最小像素值。
# 您好，关于OCR中min_pixels为3136（即28284）的28、28和4的含义如下：

# 28：表示图像的高度，单位为像素。
# 28：表示图像的宽度，单位为像素。
# 4：表示每个像素的通道数，通常为4个通道（R、G、B和Alpha）。


# 关于OCR中图片的宽高比要求，根据阿里云文字识别服务的要求，单张图片的最长边不超过4096像素，最短边不小于15像素，当长边超过1024像素时，长宽比不超过1:10。


# 这通常与OCR模型的设计和性能优化有关。1280这个值可能是为了平衡图像质量和处理速度，确保在大多数情况下能够高效地处理图像。
class DashscopeOCRCaller:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_versions = get_model_versions("ocr")
        return {
            "required": {
                "model_version": (
                    model_versions,
                    {"default": model_versions[0]},
                ),
                "min_pixels": (
                    "STRING",
                    {"default": "28 * 28 * 4"},
                ),  # width, height, channels
                "max_pixels": (
                    "STRING",
                    {"default": "28 * 28 * 1280"},
                ),  # larger channels to balance image quality and processing speed
                "image": ("IMAGE", {"default": None}),
            }
        }

    FUNCTION = "call_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def call_model(self, model_version, min_pixels, max_pixels, image) -> tuple[str]:
        if not model_version:
            raise ValueError("Model version cannot be empty")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

        print("Call the OCR model")
        image_url = get_image_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": image_url,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": "Read all the text in the image."},
                ],
            },
        ]
        response = MultiModalConversation.call(
            api_key=api_key,
            model=model_version,
            messages=messages,
            result_format="message",
        )

        if response.status_code == HTTPStatus.OK:
            print(response.output)
        else:
            print(response.code)
            print(response.message)

        message_content = response.output.choices[0].message.content[0]["text"]
        return (message_content,)


class DashscopeModelCaller:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": ("STRING", {"default": None}),
                "system_prompt": (
                    "STRING",
                    {"default": None},
                ),
                "user_prompt": ("STRING", {"default": None}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
            },
        }

    FUNCTION = "call_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)

    CATEGORY = "dashscope"

    def call_model(
        self, model_version, system_prompt, user_prompt, image
    ) -> tuple[str]:
        if not model_version:
            raise ValueError("Model version cannot be empty")
        if not system_prompt:
            raise ValueError("System prompt cannot be empty")
        if not user_prompt:
            raise ValueError("User prompt cannot be empty")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

        if image is None:
            print("Call the LLM model")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = Generation.call(
                api_key=api_key,
                model=model_version,
                messages=messages,
                result_format="message",
            )
        else:
            print("Call the VLM model")
            image_url = get_image_url(image)
            messages = [
                {"role": "system", "content": [{"text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": user_prompt},
                    ],
                },
            ]
            response = MultiModalConversation.call(
                api_key=api_key,
                model=model_version,
                messages=messages,
                result_format="message",
            )

        if response.status_code == HTTPStatus.OK:
            print(response.output)
        else:
            print(response.code)
            print(response.message)

        message_content = response.output.choices[0].message.content[0]["text"]
        return (message_content,)


# 输入限制
# 图像格式：格式为jpg，jpeg，png，bmp，webp。

# 图像分辨率：图像最小边长≥400像素，最大边长≤7000像素。

# 图像坐标框：图像需先通过EMO图像检测API，以获得正确的人脸区域和动态区域坐标信息。

# 音频格式：格式为wav、mp3。

# 音频限制：文件<15M，时长＜60s。

# 音频内容：音频中需包含清晰、响亮的人声语音，并去除了环境噪音、背景音乐等声音干扰信息。


# 上传图片、音频链接仅支持HTTP链接方式，不支持本地链接方式。
class DashscopeEmoCaller:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.face_detect_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/face-detect"
        self.video_systhesis_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "audio": ("AUDIO", {"default": None}),
            },
        }

    FUNCTION = "call_model"
    OUTPUT_NODE = True
    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "VHS_VIDEOINFO",
        "AUDIO",
    )
    RETURN_NAMES = ("frames", "frame_count", "video_info", "audio")

    CATEGORY = "dashscope"

    def detect_face(self, image, ratio) -> tuple[str]:
        ERROR_CODE_MAP = {
            "InvalidParameter": "The request parameters format is invalid.",
            "InvalidParameter.Ratio": "Invalid ratio parameter, must be '1:1' or '3:4'.",
            "InvalidURL": "Failed to download input image, please check network or input format.",
            "InvalidFile.NoHuman": "No human detected in the input image.",
            "InvalidFile.MultiHuman": "Multiple humans detected in the image.",
            "InvalidFile.BodyProportion": "The proportion of person in the image is invalid.",
            "InvalidFile.Resolution": "Image resolution must be between 400px and 7000px.",
            "InvalidFile.Value": "The image is too dark.",
            "InvalidFile.FrontBody": "Person must be facing the camera.",
            "InvalidFile.FullFace": "Face must be fully visible in the image.",
            "InvalidFile.FacePose": "Face orientation has severe deviation.",
            "InvalidFile.FullBody": "Full body or head must be completely visible based on ratio.",
        }
        ERROR_MESSAGE_MAP = {
            "The input image has no human body. Please upload other image with single person.": "未检测到人脸。",
            "The input image has multi human bodies. Please upload other image with single person.": "请上传单人照。",
            "The proportion of the detected person in the picture is too large or too small, please upload other image.": "上传图片中人脸占比过大/过小。",
            "The image resolution is invalid, please make sure that the largest length of image is smaller than 7000, and the smallest length of image is larger than 400.": "分辨率不得低于400*400。分辨率不得高于7000*7000。",
            "The value of the image is invalid, please upload other clearer image.": "请确保图片中人脸清晰。",
            "The pose of the detected person is invalid, please upload other image with the front view.": "请确保图片中人物正面朝向镜头。",
            "The pose of the detected face is invalid, please upload other image with whole face.": "请确保图片中人脸完整无遮挡。",
            "The pose of the detected face is invalid, please upload other image with the expected oriention.": "请确保图片中人脸朝向无偏斜。",
            "The pose of the detected person is invalid, please upload other image with whole body, or change the ratio parameter to 1:1.": "请确保图片中人脸完整可见（针对1:1画幅）\n请确保图片中人物上半身完整可见（针对3:4画幅）。",
        }

        img_url = get_image_url(image)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "emo-detect-v1",
            "input": {"image_url": img_url},
            "parameters": {"ratio": ratio},
        }

        response = requests.post(self.face_detect_url, headers=headers, json=payload)

        if response.status_code == HTTPStatus.OK:
            response_data = response.json()
            output = response_data.get("output", {})
            face_bbox = output.get("face_bbox", [])
            ext_bbox = output.get("ext_bbox", [])
            return (face_bbox, ext_bbox)
        elif response.status_code == HTTPStatus.BAD_REQUEST:
            error_response = response.json()
            error_output = error_response.get("output", {})
            error_code = error_output.get("code", "")
            error_message = error_output.get("message", "")
            error_message = ERROR_CODE_MAP.get(
                error_code, f"Unknown error: {error_code}"
            )
            friendly_message = ERROR_MESSAGE_MAP.get(error_message, error_message)
            print(friendly_message)
            return (error_message,)

    def _request_synthesis(self, image, audio):
        img_url = get_image_url(image)
        audio_url = get_audio_url(audio)
        face_bbox, ext_bbox = self.detect_face(image, "1:1")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

        payload = {
            "model": "emo-v1",
            "input": {
                "image_url": img_url,
                "audio_url": audio_url,
                "face_bbox": face_bbox,
                "ext_bbox": ext_bbox,
            },
            "parameters": {"style_level": "normal"},
        }

        response = requests.post(
            self.video_systhesis_url, headers=headers, json=payload
        )

        if response.status_code != 200:
            raise RuntimeError(f"Synthesis request failed: {response.text}")

        return response.json()

    def _check_task_status(self, task_id):
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.get(
            f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}", headers=headers
        )

        if response.status_code != 200:
            raise RuntimeError(f"Status check failed: {response.text}")

        return response.json()

    def call_model(self, image, audio):
        if image is None:
            raise ValueError("Image cannot be empty")
        if audio is None:
            raise ValueError("Audio cannot be empty")

        synthesis_response = self._request_synthesis(image, audio)
        task_id = synthesis_response.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError("Failed to get task_id")

        while True:
            status_response = self._check_task_status(task_id)
            task_status = status_response.get("output", {}).get("task_status")

            if task_status == "SUCCEEDED":
                video_url = (
                    status_response.get("output", {})
                    .get("results", {})
                    .get("video_url")
                )
                response = requests.get(video_url)
                if response.status_code == 200:
                    video_filename = generate_random_video_name()
                    with open(video_filename, "wb") as f:
                        f.write(response.content)
                    frames, frame_count, video_info = video_to_images(video_filename)
                    return (frames, frame_count, video_info, audio)
                else:
                    raise RuntimeError(
                        f"Failed to download video: {response.status_code}"
                    )
            elif task_status == "FAILED":
                error_message = status_response.get("output", {}).get(
                    "message", "Unknown error"
                )
                raise RuntimeError(f"Task failed: {error_message}")
            elif task_status in [
                "PENDING",
                "PRE-PROCESSING",
                "RUNNING",
                "POST-PROCESSING",
            ]:
                time.sleep(2)
            else:
                raise RuntimeError(f"Unknown task status: {task_status}")
