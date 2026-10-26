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

public class PoseActivity extends CameraModelActivity {

    private static final int SIZE = 640;

    @Override
    protected OrtSession createSession(OrtEnvironment env) throws Exception {
        return env.createSession(assetToFile("models/yolov8n-pose.onnx").getAbsolutePath(),
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
            float[] flat = new float[56 * 8400];
            for (int c = 0; c < 56; c++) System.arraycopy(out[0][c], 0, flat, c * 8400, 8400);
            List<Yolo.Pose> poses = Yolo.parsePose(flat, 8400, 0.4f);
            for (Yolo.Pose p : poses) {
                p.x1 = Yolo.unletterboxX(p.x1, tp); p.x2 = Yolo.unletterboxX(p.x2, tp);
                p.y1 = Yolo.unletterboxY(p.y1, tp); p.y2 = Yolo.unletterboxY(p.y2, tp);
                for (int k = 0; k < 17; k++) {
                    p.kpts[k * 3] = Yolo.unletterboxX(p.kpts[k * 3], tp);
                    p.kpts[k * 3 + 1] = Yolo.unletterboxY(p.kpts[k * 3 + 1], tp);
                }
            }
            String msg = String.format(Locale.US, "%d person(s) · %d ms/frame", poses.size(), ms);
            runOnUiThread(() -> {
                overlay.setResults(null, poses, null, frame.getWidth(), frame.getHeight());
                status.setText(msg);
            });
        }
    }
}
