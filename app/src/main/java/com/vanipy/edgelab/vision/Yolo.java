package com.vanipy.edgelab.vision;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Yolo {

    public static class Det {
        public float x1, y1, x2, y2, score;
        public int cls;
    }

    public static class Pose {
        public float x1, y1, x2, y2, score;
        public float[] kpts = new float[51]; // 17 * (x, y, conf)
    }

    /** Letterbox src into size x size (gray 114), returns transform in tp = {scale, padX, padY}. */
    public static Bitmap letterbox(Bitmap src, int size, float[] tp) {
        float scale = Math.min(size / (float) src.getWidth(), size / (float) src.getHeight());
        int nw = Math.round(src.getWidth() * scale);
        int nh = Math.round(src.getHeight() * scale);
        int padX = (size - nw) / 2;
        int padY = (size - nh) / 2;
        Bitmap out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(out);
        c.drawColor(Color.rgb(114, 114, 114));
        Bitmap scaled = Bitmap.createScaledBitmap(src, nw, nh, true);
        c.drawBitmap(scaled, padX, padY, new Paint());
        tp[0] = scale; tp[1] = padX; tp[2] = padY;
        return out;
    }

    /** NCHW float array, RGB, pixels / 255. */
    public static float[] bitmapToNchw(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        int[] px = new int[w * h];
        bmp.getPixels(px, 0, w, 0, 0, w, h);
        float[] out = new float[3 * w * h];
        for (int i = 0; i < px.length; i++) {
            int p = px[i];
            out[i] = ((p >> 16) & 0xFF) / 255f;
            out[w * h + i] = ((p >> 8) & 0xFF) / 255f;
            out[2 * w * h + i] = (p & 0xFF) / 255f;
        }
        return out;
    }

    /** YOLOv8 detect output [1,84,8400] in letterbox space. */
    public static List<Det> parseDetect(float[] out, int numAnchors, int numClasses, float confTh) {
        List<Det> dets = new ArrayList<>();
        for (int i = 0; i < numAnchors; i++) {
            float best = 0;
            int bestC = -1;
            for (int c = 0; c < numClasses; c++) {
                float s = out[(4 + c) * numAnchors + i];
                if (s > best) { best = s; bestC = c; }
            }
            if (best < confTh) continue;
            float cx = out[i], cy = out[numAnchors + i];
            float w = out[2 * numAnchors + i], h = out[3 * numAnchors + i];
            Det d = new Det();
            d.x1 = cx - w / 2; d.y1 = cy - h / 2;
            d.x2 = cx + w / 2; d.y2 = cy + h / 2;
            d.score = best; d.cls = bestC;
            dets.add(d);
        }
        return nms(dets, 0.45f);
    }

    /** YOLOv8 pose output [1,56,8400] in letterbox space. */
    public static List<Pose> parsePose(float[] out, int numAnchors, float confTh) {
        List<Pose> poses = new ArrayList<>();
        for (int i = 0; i < numAnchors; i++) {
            float conf = out[4 * numAnchors + i];
            if (conf < confTh) continue;
            Pose p = new Pose();
            float cx = out[i], cy = out[numAnchors + i];
            float w = out[2 * numAnchors + i], h = out[3 * numAnchors + i];
            p.x1 = cx - w / 2; p.y1 = cy - h / 2;
            p.x2 = cx + w / 2; p.y2 = cy + h / 2;
            p.score = conf;
            for (int k = 0; k < 17; k++) {
                p.kpts[k * 3] = out[(5 + k * 3) * numAnchors + i];
                p.kpts[k * 3 + 1] = out[(5 + k * 3 + 1) * numAnchors + i];
                p.kpts[k * 3 + 2] = out[(5 + k * 3 + 2) * numAnchors + i];
            }
            poses.add(p);
        }
        // NMS on pose boxes
        List<Det> boxes = new ArrayList<>();
        for (Pose p : poses) {
            Det d = new Det();
            d.x1 = p.x1; d.y1 = p.y1; d.x2 = p.x2; d.y2 = p.y2; d.score = p.score;
            boxes.add(d);
        }
        List<Det> kept = nms(boxes, 0.45f);
        List<Pose> outPoses = new ArrayList<>();
        for (Det d : kept) {
            outPoses.add(poses.get(boxes.indexOf(d)));
        }
        return outPoses;
    }

    public static List<Det> nms(List<Det> in, float iouTh) {
        List<Det> sorted = new ArrayList<>(in);
        sorted.sort((a, b) -> Float.compare(b.score, a.score));
        List<Det> kept = new ArrayList<>();
        boolean[] removed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (removed[i]) continue;
            Det a = sorted.get(i);
            kept.add(a);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (removed[j]) continue;
                Det b = sorted.get(j);
                float ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
                float ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
                float iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
                float inter = iw * ih;
                float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
                float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
                float iou = inter / (areaA + areaB - inter + 1e-6f);
                if (iou > iouTh) removed[j] = true;
            }
        }
        return kept;
    }

    /** Map letterbox coordinate back to original bitmap space. */
    public static float unletterboxX(float x, float[] tp) { return (x - tp[1]) / tp[0]; }
    public static float unletterboxY(float y, float[] tp) { return (y - tp[2]) / tp[0]; }

    public static Bitmap rotate(Bitmap src, int degrees) {
        if (degrees == 0) return src;
        Matrix m = new Matrix();
        m.postRotate(degrees);
        return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
    }
}
