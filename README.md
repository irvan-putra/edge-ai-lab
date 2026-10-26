# Edge AI Lab

On-device sensor + small-model playground for Android, built for the
Edge AI Multimodal Demos talk. Test build 0.1 — **not phone-tested yet**;
the verified parts are listed below.

## What it does

| Screen | What runs | Model | Size |
| --- | --- | --- | --- |
| Sensor Lab | Every sensor Android reports on the device, with live values | - | - |
| Object Detection | Camera frames through YOLOv8n | yolov8n.onnx, 3.2M params | 12.9 MB, bundled |
| Pose Estimation | Camera frames through YOLOv8n-pose | yolov8n-pose.onnx, 3.3M params | 13.5 MB, bundled |
| Image Classification | Center-crop frames through MobileNetV2, top-3 ImageNet labels | mobilenetv2-7.onnx | ~14 MB, downloaded on tap |
| Speech to Text | Record ≤30 s, transcribe with Whisper tiny (8-bit) | onnx-community/whisper-tiny | ~72 MB, downloaded on tap |

Note on sizes: the YOLO ONNX exports are fp32 and larger than the .pt
sources (yolov8n-pose.pt is 6.83 MB vs 13.5 MB as ONNX). Format difference,
not different models.

## Privacy

All inference is on-device. Network is used only when you explicitly tap a
Download button (model files from their public sources, size shown first).
Camera/mic only inside their demo screens. No accounts, analytics, background
recording, or uploads. Models can be deleted from app storage.

## Honest limits

- Sensor Lab lists what Android reports, including software/virtual sensors.
  An absent sensor is unavailable to apps regardless of spec sheets.
- Small models make mistakes. That is the point of the demo.
- Inference timings shown in-app are measured on your device per run.

## Verified before shipping (on the build host, not on a phone)

- App compiles (debug APK, arm64-v8a, minSdk 26, target 34).
- yolov8n.onnx / yolov8n-pose.onnx exported from official Ultralytics .pt
  weights and smoke-tested in ONNX Runtime: output shapes [1,84,8400] /
  [1,56,8400].
- Whisper pipeline: the ACTUAL COMPILED Java classes (MelSpectrogram,
  WhisperTokenizer, WhisperEngine from this repo) were run on the desktop
  JVM against the real quantized onnx-community models and correctly
  transcribe whisper's JFK test clip ("And so my fellow Americans ask not
  what your country can do for you..."). The hand-written log-mel/FFT
  matches the numpy reference to 2.3e-6; the mel filterbank is whisper's
  own mel_filters.npz values, bundled as an asset.
- Not covered by host testing: Android UI behavior, camera/mic pipelines,
  Sensor Lab output, and on-device speed. Those need the phone.

## Build

Needs JDK 17, Android SDK 34, Gradle 8.7.

    gradle assembleDebug
    # APK: app/build/outputs/apk/debug/app-debug.apk

## Sources and licenses

- YOLOv8n / YOLOv8n-pose: Ultralytics, AGPL-3.0 (github.com/ultralytics/ultralytics)
- Whisper tiny ONNX (quantized): OpenAI Whisper, MIT; export by onnx-community (huggingface.co/onnx-community/whisper-tiny)
- MobileNetV2: ONNX Model Zoo, Apache-2.0 (github.com/onnx/models)
- ONNX Runtime: Microsoft, MIT
