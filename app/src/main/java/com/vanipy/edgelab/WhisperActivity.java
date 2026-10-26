package com.vanipy.edgelab;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Process;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.vanipy.edgelab.ort.ModelStore;
import com.vanipy.edgelab.whisper.MelSpectrogram;
import com.vanipy.edgelab.whisper.WhisperEngine;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.Locale;

public class WhisperActivity extends Activity {

    private TextView status;
    private TextView resultView;
    private Button recordBtn;
    private RadioGroup langGroup;
    private WhisperEngine engine;
    private volatile boolean recording;
    private Thread recordThread;
    private ByteArrayOutputStream pcmBytes;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(16);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("Whisper tiny · speech to text · on-device");
        title.setTextSize(18);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);

        status = new TextView(this);
        status.setPadding(0, dp(8), 0, dp(8));
        root.addView(status);

        langGroup = new RadioGroup(this);
        langGroup.setOrientation(RadioGroup.HORIZONTAL);
        RadioButton en = new RadioButton(this);
        en.setText("English");
        en.setId(1);
        RadioButton idn = new RadioButton(this);
        idn.setText("Indonesian");
        idn.setId(2);
        langGroup.addView(en);
        langGroup.addView(idn);
        langGroup.check(1);
        root.addView(langGroup);

        recordBtn = new Button(this);
        recordBtn.setText("Record (up to 30 s)");
        root.addView(recordBtn);

        resultView = new TextView(this);
        resultView.setTextSize(15);
        resultView.setPadding(0, dp(12), 0, 0);
        resultView.setTextIsSelectable(true);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(resultView);
        root.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        recordBtn.setOnClickListener(v -> toggleRecording());
        setContentView(root);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshModelState();
    }

    private void refreshModelState() {
        if (ModelStore.isDownloaded(this, ModelStore.WHISPER)) {
            status.setText("Model on device (~72 MB). Loading engine...");
            recordBtn.setEnabled(false);
            new Thread(() -> {
                try {
                    engine = new WhisperEngine(ModelStore.dir(this), getAssets().open("mel_filters.txt"));
                    runOnUiThread(() -> {
                        status.setText("Ready. Pick a language, tap Record, speak, tap Stop.");
                        recordBtn.setEnabled(true);
                    });
                } catch (Exception e) {
                    runOnUiThread(() -> status.setText("Engine failed to load: " + e.getMessage()
                            + "\nFiles may be corrupt - delete and re-download."));
                }
            }).start();
        } else {
            status.setText("Whisper tiny is not on the phone yet.");
            recordBtn.setEnabled(false);
            showDownloadUi();
        }
    }

    private void showDownloadUi() {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        TextView tv = new TextView(this);
        long mb = ModelStore.totalBytes(ModelStore.WHISPER) / (1024 * 1024);
        tv.setText("One-time download: ~" + mb + " MB (quantized Whisper tiny: encoder, decoder, tokenizer) from onnx-community on Hugging Face. Runs fully on-device afterwards.");
        panel.addView(tv);
        ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(100);
        bar.setVisibility(View.GONE);
        panel.addView(bar);
        Button btn = new Button(this);
        btn.setText("Download (~" + mb + " MB)");
        panel.addView(btn);
        TextView err = new TextView(this);
        err.setTextColor(0xFFCC0000);
        panel.addView(err);
        int idx = recordBtn != null ? ((LinearLayout) recordBtn.getParent()).indexOfChild(recordBtn) : 1;
        ((LinearLayout) recordBtn.getParent()).addView(panel, idx);
        btn.setOnClickListener(v -> {
            btn.setEnabled(false);
            bar.setVisibility(View.VISIBLE);
            ModelStore.download(this, ModelStore.WHISPER, new ModelStore.Callback() {
                @Override public void onProgress(long done, long total) {
                    bar.setProgress((int) (100 * done / Math.max(1, total)));
                }
                @Override public void onDone(File dir) {
                    ((LinearLayout) panel.getParent()).removeView(panel);
                    refreshModelState();
                }
                @Override public void onError(String message) {
                    btn.setEnabled(true);
                    err.setText("Download failed: " + message + "\nCheck connection and retry. Nothing else changed.");
                }
            });
        });
    }

    private void toggleRecording() {
        if (recording) {
            recording = false;
            recordBtn.setText("Transcribing...");
            recordBtn.setEnabled(false);
            return;
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 7);
            return;
        }
        startRecording();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 7 && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording();
        } else if (requestCode == 7) {
            status.setText("Microphone permission denied; this demo needs the mic.");
        }
    }

    private void startRecording() {
        pcmBytes = new ByteArrayOutputStream();
        recording = true;
        recordBtn.setText("Stop");
        resultView.setText("Recording... up to 30 s. Tap Stop when done.");
        recordThread = new Thread(() -> {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
            int minBuf = AudioRecord.getMinBufferSize(MelSpectrogram.SR,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
            AudioRecord rec = new AudioRecord(MediaRecorder.AudioSource.MIC, MelSpectrogram.SR,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, Math.max(minBuf, 64000));
            try {
                rec.startRecording();
                byte[] buf = new byte[6400];
                while (recording && pcmBytes.size() < MelSpectrogram.N_SAMPLES * 2) {
                    int n = rec.read(buf, 0, buf.length);
                    if (n > 0) pcmBytes.write(buf, 0, n);
                }
            } catch (Exception e) {
                runOnUiThread(() -> status.setText("Recording failed: " + e.getMessage()));
            } finally {
                try { rec.stop(); } catch (Exception ignored) {}
                rec.release();
            }
            recording = false;
            runOnUiThread(this::transcribeCaptured);
        });
        recordThread.start();
    }

    private void transcribeCaptured() {
        byte[] raw = pcmBytes.toByteArray();
        int frames = raw.length / 2;
        float[] pcm = new float[frames];
        for (int i = 0; i < frames; i++) {
            short s = (short) ((raw[2 * i] & 0xFF) | (raw[2 * i + 1] << 8));
            pcm[i] = s / 32768f;
        }
        float seconds = frames / (float) MelSpectrogram.SR;
        if (frames < MelSpectrogram.SR / 4) {
            resultView.setText("Too short (" + String.format(Locale.US, "%.2f", seconds) + " s). Try again.");
            recordBtn.setText("Record (up to 30 s)");
            recordBtn.setEnabled(true);
            return;
        }
        int langToken = engine.languageToken(langGroup.getCheckedRadioButtonId() == 2 ? "<|id|>" : "<|en|>",
                langGroup.getCheckedRadioButtonId() == 2 ? 50275 : 50259);
        new Thread(() -> {
            try {
                WhisperEngine.Result r = engine.transcribe(pcm, langToken);
                String stats = String.format(Locale.US,
                        "\n\n---\n%.1f s audio · mel %d ms · encode %d ms · decode %d ms · %d tokens",
                        seconds, r.melMs, r.encodeMs, r.decodeMs, r.tokens);
                runOnUiThread(() -> {
                    resultView.setText((r.text.isEmpty() ? "(no speech recognized)" : r.text) + stats);
                    recordBtn.setText("Record (up to 30 s)");
                    recordBtn.setEnabled(true);
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    resultView.setText("Transcription error: " + e.getMessage());
                    recordBtn.setText("Record (up to 30 s)");
                    recordBtn.setEnabled(true);
                });
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        recording = false;
        try { if (engine != null) engine.close(); } catch (Exception ignored) {}
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }
}
