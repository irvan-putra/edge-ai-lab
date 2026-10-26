package com.vanipy.edgelab;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Typeface;
import android.os.Bundle;
import android.util.Size;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.activity.ComponentActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.vanipy.edgelab.ort.OverlayView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public abstract class CameraModelActivity extends ComponentActivity {

    protected OrtEnvironment ortEnv;
    protected OrtSession session;
    protected OverlayView overlay;
    protected TextView status;
    protected String[] labels = new String[0];

    private final AtomicBoolean busy = new AtomicBoolean(false);
    private final AtomicBoolean live = new AtomicBoolean(true);
    private ExecutorService cameraExecutor;
    private PreviewView previewView;

    /** Load the ONNX session (from bundled asset name or downloaded file). */
    protected abstract OrtSession createSession(OrtEnvironment env) throws Exception;
    /** Run one frame; called on a background thread. Update overlay/status via runOnUiThread. */
    protected abstract void runModel(Bitmap frame) throws Exception;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ortEnv = OrtEnvironment.getEnvironment();

        FrameLayout root = new FrameLayout(this);
        previewView = new PreviewView(this);
        previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER);
        root.addView(previewView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        overlay = new OverlayView(this);
        root.addView(overlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(12), dp(8), dp(12), dp(8));
        panel.setBackgroundColor(0xCC000000);
        status = new TextView(this);
        status.setTextColor(0xFFFFFFFF);
        status.setTextSize(13);
        status.setTypeface(Typeface.MONOSPACE);
        status.setText("Loading model...");
        TextView hint = new TextView(this);
        hint.setTextColor(0xFFB0B0B0);
        hint.setTextSize(11);
        hint.setText("Tap preview to pause/resume analysis. Everything runs on-device.");
        panel.addView(status);
        panel.addView(hint);
        FrameLayout.LayoutParams plp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        plp.gravity = android.view.Gravity.BOTTOM;
        root.addView(panel, plp);

        previewView.setOnClickListener(v -> {
            live.set(!live.get());
            status.setText(live.get() ? "Resumed." : "Paused. Tap to resume.");
        });

        setContentView(root);
        getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraExecutor = Executors.newSingleThreadExecutor();

        new Thread(() -> {
            try {
                session = createSession(ortEnv);
                runOnUiThread(() -> status.setText("Model ready. Point the camera."));
            } catch (Exception e) {
                runOnUiThread(() -> status.setText("Model failed to load: " + e.getMessage()));
            }
        }).start();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 42);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 42 && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else if (requestCode == 42) {
            status.setText("Camera permission denied; this demo needs the camera.");
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build();
                analysis.setAnalyzer(cameraExecutor, this::analyze);
                provider.unbindAll();
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis);
            } catch (Exception e) {
                status.setText("Camera failed: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyze(ImageProxy image) {
        try {
            if (!live.get() || session == null || !busy.compareAndSet(false, true)) return;
            Bitmap bmp = rgbaToBitmap(image);
            int rot = image.getImageInfo().getRotationDegrees();
            Bitmap rotated = com.vanipy.edgelab.vision.Yolo.rotate(bmp, rot);
            long t0 = System.nanoTime();
            runModel(rotated);
        } catch (Exception e) {
            String msg = e.getMessage();
            runOnUiThread(() -> status.setText("Inference error: " + msg));
        } finally {
            busy.set(false);
            image.close();
        }
    }

    private static Bitmap rgbaToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int w = image.getWidth(), h = image.getHeight();
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        if (pixelStride == 4 && rowStride == w * 4) {
            bmp.copyPixelsFromBuffer(buffer);
        } else {
            int[] px = new int[w * h];
            byte[] row = new byte[rowStride];
            for (int y = 0; y < h; y++) {
                buffer.position(y * rowStride);
                buffer.get(row, 0, Math.min(rowStride, buffer.remaining()));
                for (int x = 0; x < w; x++) {
                    int off = x * pixelStride;
                    int r = row[off] & 0xFF, g = row[off + 1] & 0xFF, b = row[off + 2] & 0xFF, a = row[off + 3] & 0xFF;
                    px[y * w + x] = (a << 24) | (r << 16) | (g << 8) | b;
                }
            }
            bmp.setPixels(px, 0, w, 0, 0, w, h);
        }
        return bmp;
    }

    /** Copy a bundled asset to a cache file (session loads from path, keeping memory low). */
    protected File assetToFile(String assetName) throws Exception {
        File out = new File(getCacheDir(), assetName.replace('/', '_'));
        if (out.exists() && out.length() > 0) return out;
        try (InputStream in = getAssets().open(assetName);
             FileOutputStream fos = new FileOutputStream(out)) {
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) != -1) fos.write(buf, 0, n);
        }
        return out;
    }

    protected String[] loadLabels(String assetName) {
        try (InputStream in = getAssets().open(assetName);
             java.io.BufferedReader r = new java.io.BufferedReader(new java.io.InputStreamReader(in))) {
            java.util.List<String> lines = new java.util.ArrayList<>();
            String line;
            while ((line = r.readLine()) != null) lines.add(line.trim());
            return lines.toArray(new String[0]);
        } catch (Exception e) {
            return new String[0];
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) cameraExecutor.shutdown();
        try { if (session != null) session.close(); } catch (Exception ignored) {}
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }
}
