package com.vanipy.edgelab;

import android.app.Activity;
import android.graphics.Typeface;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class SensorsActivity extends Activity implements SensorEventListener {

    private SensorManager sensorManager;
    private Sensor activeSensor;
    private TextView liveView;
    private TextView header;
    private long sampleCount;
    private long firstEventNanos;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);

        header = new TextView(this);
        header.setPadding(dp(16), dp(16), dp(16), dp(8));
        List<Sensor> sensors = sensorManager.getSensorList(Sensor.TYPE_ALL);
        header.setText(sensors.size() + " sensors reported by this device.\n"
                + "This is the on-device ground truth: what Android (and apps) can actually use, "
                + "including virtual/software sensors. Tap one for live values.");
        root.addView(header);

        liveView = new TextView(this);
        liveView.setPadding(dp(16), dp(4), dp(16), dp(8));
        liveView.setTypeface(Typeface.MONOSPACE);
        liveView.setText("No sensor selected.");
        root.addView(liveView);

        ListView list = new ListView(this);
        List<String> rows = new ArrayList<>();
        for (Sensor s : sensors) {
            rows.add(s.getName() + "\n"
                    + s.getVendor() + " v" + s.getVersion()
                    + " · type " + s.getType() + (s.getStringType() != null ? " (" + s.getStringType() + ")" : "")
                    + "\nrange ±" + s.getMaximumRange() + " · res " + s.getResolution()
                    + " · " + s.getPower() + " mA · delay " + s.getMinDelay() + " µs"
                    + (s.isWakeUpSensor() ? " · wake-up" : ""));
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,
                android.R.layout.simple_list_item_1, rows) {
            @Override
            public View getView(int position, View convertView, ViewGroup parent) {
                View v = super.getView(position, convertView, parent);
                ((TextView) v).setTextSize(12);
                return v;
            }
        };
        list.setAdapter(adapter);
        list.setOnItemClickListener((AdapterView<?> parent, View view, int position, long id) -> {
            if (activeSensor != null) sensorManager.unregisterListener(this);
            activeSensor = sensors.get(position);
            sampleCount = 0;
            firstEventNanos = 0;
            boolean ok = sensorManager.registerListener(this, activeSensor, SensorManager.SENSOR_DELAY_UI);
            liveView.setText("Listening: " + activeSensor.getName() + (ok ? "" : " (registerListener returned false)"));
        });
        root.addView(list, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (firstEventNanos == 0) firstEventNanos = event.timestamp;
        sampleCount++;
        StringBuilder sb = new StringBuilder();
        sb.append(event.sensor.getName()).append('\n');
        for (int i = 0; i < event.values.length; i++) {
            sb.append(String.format(Locale.US, "v[%d] = % .5f\n", i, event.values[i]));
        }
        sb.append("accuracy ").append(event.accuracy)
          .append(" · samples ").append(sampleCount)
          .append(String.format(Locale.US, " · %.1f Hz",
                  sampleCount / Math.max(0.001, (event.timestamp - firstEventNanos) / 1e9)));
        liveView.setText(sb.toString());
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }
}
