package com.vanipy.edgelab;

import android.graphics.Bitmap;

import com.vanipy.edgelab.vision.Yolo;

import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class DetectActivity extends CameraModelActivity {

    private static final int SIZE = 640;

    @Override
    protected OrtSession createSession(OrtEnvironment env) throws Exception {
        labels = loadLabels("coco.txt");
        return env.createSession(assetToFile("models/yolov8n.onnx").getAbsolutePath(),
                new OrtSession.SessionOptions());
    }

    @Override
    protected void runModel(Bitmap frame) throws Exception {
        float[] tp = new float[3];
        Bitmap lb = Yolo.letterbox(frame, SIZE, tp);
        float[] nchw = Yolo.bitmapToNchw(lb);
        long t0 = System.nanoTime();
        try (OnnxTensor input = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(nchw), new long[]{1, 3, SIZE, SIZE});
             OrtSession.Result res = session.run(Collections.singletonMap("images", input))) {
            float[][][] out = (float[][][]) res.get(0).getValue();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            List<Yolo.Det> dets = Yolo.parseDetect(flatten(out[0]), 8400, 80, 0.25f);
            for (Yolo.Det d : dets) {
                d.x1 = Yolo.unletterboxX(d.x1, tp); d.x2 = Yolo.unletterboxX(d.x2, tp);
                d.y1 = Yolo.unletterboxY(d.y1, tp); d.y2 = Yolo.unletterboxY(d.y2, tp);
            }
            String msg = String.format(Locale.US, "%d objects · %d ms/frame", dets.size(), ms);
            runOnUiThread(() -> {
                overlay.setResults(dets, null, labels, frame.getWidth(), frame.getHeight());
                status.setText(msg);
            });
        }
    }

    private static float[] flatten(float[][] a) {
        int c = a.length, n = a[0].length;
        float[] out = new float[c * n];
        for (int i = 0; i < c; i++) System.arraycopy(a[i], 0, out, i * n, n);
        return out;
    }
}
