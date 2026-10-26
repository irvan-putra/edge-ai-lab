package com.vanipy.edgelab;

import android.graphics.Bitmap;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.vanipy.edgelab.ort.ModelStore;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Locale;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class ClassifyActivity extends CameraModelActivity {

    private static final int SIZE = 224;
    private boolean modelMissing;

    @Override
    protected OrtSession createSession(OrtEnvironment env) throws Exception {
        labels = loadImagenetLabels();
        if (!ModelStore.isDownloaded(this, ModelStore.MOBILENET)) {
            modelMissing = true;
            runOnUiThread(this::showDownloadUi);
            return null;
        }
        File f = ModelStore.file(this, "mobilenetv2.onnx");
        return env.createSession(f.getAbsolutePath(), new OrtSession.SessionOptions());
    }

    private void showDownloadUi() {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(40, 200, 40, 40);
        panel.setBackgroundColor(0xFFFFFFFF);
        TextView tv = new TextView(this);
        tv.setText("MobileNetV2 is not on the phone yet.\nSize: ~14 MB, one-time download from the ONNX Model Zoo (github.com/onnx/models). Tap Download to fetch it; it stays in app-private storage.");
        tv.setTextSize(14);
        panel.addView(tv);
        ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(100);
        bar.setVisibility(View.GONE);
        panel.addView(bar);
        Button btn = new Button(this);
        btn.setText("Download (~14 MB)");
        panel.addView(btn);
        TextView err = new TextView(this);
        err.setTextColor(0xFFCC0000);
        panel.addView(err);
        addContentView(panel, new LinearLayout.LayoutParams(
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.MATCH_PARENT));
        btn.setOnClickListener(v -> {
            btn.setEnabled(false);
            bar.setVisibility(View.VISIBLE);
            ModelStore.download(this, ModelStore.MOBILENET, new ModelStore.Callback() {
                @Override public void onProgress(long done, long total) {
                    bar.setProgress((int) (100 * done / Math.max(1, total)));
                }
                @Override public void onDone(File dir) { recreate(); }
                @Override public void onError(String message) {
                    btn.setEnabled(true);
                    err.setText("Download failed: " + message + "\nCheck connection and retry. Nothing else changed.");
                }
            });
        });
    }

    @Override
    protected void runModel(Bitmap frame) throws Exception {
        if (session == null) return;
        int w = frame.getWidth(), h = frame.getHeight();
        int side = Math.min(w, h);
        Bitmap crop = Bitmap.createBitmap(frame, (w - side) / 2, (h - side) / 2, side, side);
        Bitmap scaled = Bitmap.createScaledBitmap(crop, SIZE, SIZE, true);
        int[] px = new int[SIZE * SIZE];
        scaled.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE);
        float[] nchw = new float[3 * SIZE * SIZE];
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        for (int i = 0; i < px.length; i++) {
            int p = px[i];
            nchw[i] = (((p >> 16) & 0xFF) / 255f - mean[0]) / std[0];
            nchw[SIZE * SIZE + i] = (((p >> 8) & 0xFF) / 255f - mean[1]) / std[1];
            nchw[2 * SIZE * SIZE + i] = ((p & 0xFF) / 255f - mean[2]) / std[2];
        }
        long t0 = System.nanoTime();
        String inputName = session.getInputNames().iterator().next();
        try (OnnxTensor input = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(nchw), new long[]{1, 3, SIZE, SIZE});
             OrtSession.Result res = session.run(Collections.singletonMap(inputName, input))) {
            float[][] out = (float[][]) res.get(0).getValue();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            float[] logits = out[0];
            float max = Float.NEGATIVE_INFINITY;
            for (float v : logits) max = Math.max(max, v);
            float sum = 0;
            for (int i = 0; i < logits.length; i++) { logits[i] = (float) Math.exp(logits[i] - max); sum += logits[i]; }
            int b1 = -1, b2 = -1, b3 = -1;
            for (int i = 0; i < logits.length; i++) {
                if (b1 < 0 || logits[i] > logits[b1]) { b3 = b2; b2 = b1; b1 = i; }
                else if (b2 < 0 || logits[i] > logits[b2]) { b3 = b2; b2 = i; }
                else if (b3 < 0 || logits[i] > logits[b3]) { b3 = i; }
            }
            String msg = String.format(Locale.US, "%s %.0f%% · %s %.0f%% · %s %.0f%% · %d ms",
                    label(b1), 100 * logits[b1] / sum, label(b2), 100 * logits[b2] / sum,
                    label(b3), 100 * logits[b3] / sum, ms);
            runOnUiThread(() -> status.setText(msg));
        }
    }

    private String label(int i) {
        if (labels.length == 0 || i < 0 || i >= labels.length) return "class " + i;
        String l = labels[i];
        int comma = l.indexOf(',');
        return comma > 0 ? l.substring(0, comma).trim() : l;
    }

    private String[] loadImagenetLabels() {
        String[] raw = loadLabels("imagenet_synset.txt");
        String[] out = new String[raw.length];
        for (int i = 0; i < raw.length; i++) {
            String l = raw[i];
            int sp = l.indexOf(' ');
            out[i] = sp > 0 ? l.substring(sp + 1) : l;
        }
        return out;
    }
}
