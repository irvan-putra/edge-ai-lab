package com.vanipy.edgelab;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(16);
        root.setPadding(pad, pad, pad, pad);
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("Edge AI Lab");
        title.setTextSize(26);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);

        ActivityManager am = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo mem = new ActivityManager.MemoryInfo();
        am.getMemoryInfo(mem);
        TextView device = new TextView(this);
        device.setText(Build.MANUFACTURER + " " + Build.MODEL
                + "  ·  Android " + Build.VERSION.RELEASE + " (SDK " + Build.VERSION.SDK_INT + ")"
                + "  ·  " + Build.HARDWARE + "  ·  RAM " + (mem.totalMem / (1024 * 1024)) + " MB");
        device.setTextSize(13);
        device.setPadding(0, dp(4), 0, dp(12));
        root.addView(device);

        addEntry(root, "Sensor Lab", "Every sensor your phone reports to Android, with live values", SensorsActivity.class);
        addEntry(root, "Object Detection", "YOLOv8n · 3.2M params · 12.9 MB bundled · camera, on-device", DetectActivity.class);
        addEntry(root, "Pose Estimation", "YOLOv8n-pose · 3.3M params · 13.5 MB bundled · camera, on-device", PoseActivity.class);
        addEntry(root, "Image Classification", "MobileNetV2 · 1000 ImageNet classes · ~14 MB, downloaded on demand", ClassifyActivity.class);
        addEntry(root, "Speech to Text", "Whisper tiny (quantized) · 39M params · ~72 MB, downloaded on demand", WhisperActivity.class);
        addEntry(root, "About & Privacy", "Models, sizes, licenses, and what leaves the phone (nothing)", AboutActivity.class);

        setContentView(scroll);
    }

    private void addEntry(LinearLayout root, String title, String subtitle, final Class<?> target) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(12), dp(12), dp(12), dp(12));
        TextView t = new TextView(this);
        t.setText(title);
        t.setTextSize(18);
        t.setTypeface(Typeface.DEFAULT_BOLD);
        TextView s = new TextView(this);
        s.setText(subtitle);
        s.setTextSize(13);
        card.addView(t);
        card.addView(s);
        card.setOnClickListener(v -> startActivity(new Intent(this, target)));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, 0, 0, dp(10));
        card.setLayoutParams(lp);
        card.setBackgroundColor(0xFFF0F0F0);
        root.addView(card);
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }
}
