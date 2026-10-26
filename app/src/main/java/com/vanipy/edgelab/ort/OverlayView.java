package com.vanipy.edgelab.ort;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

import com.vanipy.edgelab.vision.Yolo;

import java.util.ArrayList;
import java.util.List;

public class OverlayView extends View {

    private final Paint boxPaint = new Paint();
    private final Paint textPaint = new Paint();
    private final Paint textBg = new Paint();
    private final Paint kptPaint = new Paint();
    private final Paint linePaint = new Paint();

    private List<Yolo.Det> dets = new ArrayList<>();
    private List<Yolo.Pose> poses = new ArrayList<>();
    private String[] labels = new String[0];
    private int srcW = 1, srcH = 1;

    private static final int[][] SKELETON = {
            {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 6}, {5, 11}, {6, 12},
            {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
            {0, 1}, {0, 2}, {1, 3}, {2, 4}
    };

    public OverlayView(Context ctx) {
        super(ctx);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);
        boxPaint.setColor(Color.GREEN);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(34f);
        textBg.setColor(Color.GREEN);
        textBg.setStyle(Paint.Style.FILL);
        kptPaint.setColor(Color.CYAN);
        kptPaint.setStyle(Paint.Style.FILL);
        linePaint.setColor(Color.YELLOW);
        linePaint.setStrokeWidth(3f);
    }

    public void setResults(List<Yolo.Det> dets, List<Yolo.Pose> poses, String[] labels, int srcW, int srcH) {
        this.dets = dets != null ? dets : new ArrayList<>();
        this.poses = poses != null ? poses : new ArrayList<>();
        this.labels = labels != null ? labels : new String[0];
        this.srcW = Math.max(1, srcW);
        this.srcH = Math.max(1, srcH);
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float scale = Math.min(getWidth() / (float) srcW, getHeight() / (float) srcH);
        float ox = (getWidth() - srcW * scale) / 2f;
        float oy = (getHeight() - srcH * scale) / 2f;

        for (Yolo.Det d : dets) {
            float x1 = ox + d.x1 * scale, y1 = oy + d.y1 * scale;
            float x2 = ox + d.x2 * scale, y2 = oy + d.y2 * scale;
            canvas.drawRect(new RectF(x1, y1, x2, y2), boxPaint);
            String label = (d.cls >= 0 && d.cls < labels.length ? labels[d.cls] : ("class " + d.cls))
                    + String.format(" %.2f", d.score);
            canvas.drawRect(x1, y1 - 40, x1 + textPaint.measureText(label) + 16, y1, textBg);
            canvas.drawText(label, x1 + 8, y1 - 8, textPaint);
        }

        for (Yolo.Pose p : poses) {
            canvas.drawRect(new RectF(ox + p.x1 * scale, oy + p.y1 * scale,
                    ox + p.x2 * scale, oy + p.y2 * scale), boxPaint);
            for (int k = 0; k < 17; k++) {
                float conf = p.kpts[k * 3 + 2];
                if (conf < 0.5f) continue;
                canvas.drawCircle(ox + p.kpts[k * 3] * scale, oy + p.kpts[k * 3 + 1] * scale, 6f, kptPaint);
            }
            for (int[] bone : SKELETON) {
                float c1 = p.kpts[bone[0] * 3 + 2], c2 = p.kpts[bone[1] * 3 + 2];
                if (c1 < 0.5f || c2 < 0.5f) continue;
                canvas.drawLine(ox + p.kpts[bone[0] * 3] * scale, oy + p.kpts[bone[0] * 3 + 1] * scale,
                        ox + p.kpts[bone[1] * 3] * scale, oy + p.kpts[bone[1] * 3 + 1] * scale, linePaint);
            }
        }
    }
}
