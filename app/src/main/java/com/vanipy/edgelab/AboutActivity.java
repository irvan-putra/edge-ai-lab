package com.vanipy.edgelab;

import android.app.Activity;
import android.graphics.Typeface;
import android.os.Bundle;
import android.text.method.LinkMovementMethod;
import android.widget.ScrollView;
import android.widget.TextView;

public class AboutActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ScrollView scroll = new ScrollView(this);
        TextView tv = new TextView(this);
        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        tv.setPadding(pad, pad, pad, pad);
        tv.setTextSize(14);
        tv.setText(String.join("\n",
                "Edge AI Lab - on-device sensor and model playground",
                "Built for the Edge AI Multimodal Demos talk. Test build 0.1.",
                "",
                "MODELS",
                "- YOLOv8n object detection: 3.2M params, 12.9 MB, bundled in the app.",
                "  Source: Ultralytics (github.com/ultralytics), AGPL-3.0.",
                "- YOLOv8n-pose estimation: 3.3M params, 13.5 MB, bundled.",
                "  Source: Ultralytics, AGPL-3.0.",
                "- MobileNetV2 image classification: ~14 MB, downloaded on demand.",
                "  Source: ONNX Model Zoo (github.com/onnx/models), Apache-2.0.",
                "- Whisper tiny speech recognition: 39M params, ~72 MB quantized (8-bit: encoder + two-stage decoder),",
                "  downloaded on demand. Source: OpenAI Whisper (MIT), ONNX export by",
                "  onnx-community on Hugging Face.",
                "",
                "Runtime: ONNX Runtime (Microsoft, MIT).",
                "",
                "PRIVACY",
                "- All inference runs on this phone. No photo, audio, or sensor data",
                "  leaves the device.",
                "- Network is used only when you explicitly tap a Download button,",
                "  to fetch model files from their public sources. Downloads show",
                "  their size first. Models can be deleted from their screens.",
                "- Camera and microphone are used only inside their demo screens,",
                "  only while those screens are open.",
                "- No analytics, no accounts, no background recording, no uploads.",
                "",
                "HONEST LIMITS",
                "- Sensor Lab lists what Android reports on this device, including",
                "  software sensors. A sensor absent from the list is unavailable to",
                "  apps, whatever the marketing spec sheet says.",
                "- These are small models. Whisper tiny makes transcription mistakes,",
                "  especially in noisy audio. YOLOv8n misses objects. That is the",
                "  point of the demo: what fits in a few MB, running locally.",
                "- Inference speed depends on the phone. Timings shown per run are",
                "  measured on your device, not claimed from benchmarks."));
        tv.setMovementMethod(LinkMovementMethod.getInstance());
        tv.setTypeface(Typeface.DEFAULT);
        scroll.addView(tv);
        setContentView(scroll);
    }
}
