package com.vanipy.edgelab.whisper;

import java.util.Arrays;

/**
 * Whisper-compatible log-mel spectrogram: 16 kHz, n_fft=400, hop=160, 80 mels,
 * 30 s window (3000 frames), slaney-area-normalized filters, log10, clamp, scale.
 */
public class MelSpectrogram {

    public static final int SR = 16000;
    public static final int N_FFT = 400;
    public static final int HOP = 160;
    public static final int N_MELS = 80;
    public static final int N_SAMPLES = 480000; // 30 s
    public static final int N_FRAMES = 3000;

    private static float[] hann;
    private static float[][] melFilters;

    /** pcm: 16 kHz mono floats in [-1, 1], any length up to 30 s. Returns [80][3000]. */
    public static float[][] logMel(float[] pcm) {
        float[] audio = new float[N_SAMPLES];
        int n = Math.min(pcm.length, N_SAMPLES);
        System.arraycopy(pcm, 0, audio, 0, n);

        if (hann == null) {
            hann = new float[N_FFT];
            for (int i = 0; i < N_FFT; i++) {
                hann[i] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N_FFT));
            }
        }
        if (melFilters == null) throw new IllegalStateException("mel filters not loaded - call loadFilters first");

        float[][] mel = new float[N_MELS][N_FRAMES];
        double[] re = new double[N_FFT];
        double[] im = new double[N_FFT];
        float[] power = new float[N_FFT / 2 + 1];

        float max = 0;
        for (int t = 0; t < N_FRAMES; t++) {
            int center = t * HOP;
            Arrays.fill(re, 0);
            Arrays.fill(im, 0);
            for (int i = 0; i < N_FFT; i++) {
                int idx = center - N_FFT / 2 + i;
                if (idx >= 0 && idx < audio.length) re[i] = audio[idx] * hann[i];
            }
            fft(re, im, N_FFT);
            for (int k = 0; k <= N_FFT / 2; k++) {
                power[k] = (float) (re[k] * re[k] + im[k] * im[k]);
            }
            for (int m = 0; m < N_MELS; m++) {
                float[] f = melFilters[m];
                float sum = 0;
                for (int k = 0; k < f.length; k++) sum += f[k] * power[k];
                mel[m][t] = sum;
                if (sum > max) max = sum;
            }
        }

        float logMax = (float) Math.log10(Math.max(max, 1e-10));
        for (int m = 0; m < N_MELS; m++) {
            for (int t = 0; t < N_FRAMES; t++) {
                float v = (float) Math.log10(Math.max(mel[m][t], 1e-10));
                v = Math.max(v, logMax - 8.0f);
                mel[m][t] = (v + 4.0f) / 4.0f;
            }
        }
        return mel;
    }

    /** General mixed-radix DIT FFT for composite n (handles 400 = 2^4 * 5^2). */
    static void fft(double[] re, double[] im, int n) {
        if (n <= 1) return;
        int p = smallestFactor(n);
        if (p == n) { naiveDft(re, im, n); return; }
        int m = n / p;
        double[][] sre = new double[p][m];
        double[][] sim = new double[p][m];
        for (int r = 0; r < p; r++) {
            for (int t = 0; t < m; t++) {
                sre[r][t] = re[p * t + r];
                sim[r][t] = im[p * t + r];
            }
            fft(sre[r], sim[r], m);
        }
        double[] ore = new double[n];
        double[] oim = new double[n];
        for (int k = 0; k < n; k++) {
            double sr = 0, si = 0;
            for (int r = 0; r < p; r++) {
                double ang = -2 * Math.PI * r * k / n;
                double wr = Math.cos(ang), wi = Math.sin(ang);
                double yr = sre[r][k % m], yi = sim[r][k % m];
                sr += yr * wr - yi * wi;
                si += yr * wi + yi * wr;
            }
            ore[k] = sr;
            oim[k] = si;
        }
        System.arraycopy(ore, 0, re, 0, n);
        System.arraycopy(oim, 0, im, 0, n);
    }

    private static int smallestFactor(int n) {
        for (int p = 2; p * p <= n; p++) if (n % p == 0) return p;
        return n;
    }

    private static void naiveDft(double[] re, double[] im, int n) {
        double[] ore = new double[n];
        double[] oim = new double[n];
        for (int k = 0; k < n; k++) {
            for (int t = 0; t < n; t++) {
                double ang = -2 * Math.PI * k * t / n;
                ore[k] += re[t] * Math.cos(ang) - im[t] * Math.sin(ang);
                oim[k] += re[t] * Math.sin(ang) + im[t] * Math.cos(ang);
            }
        }
        System.arraycopy(ore, 0, re, 0, n);
        System.arraycopy(oim, 0, im, 0, n);
    }

    /** Load the exact whisper mel filterbank (dumped from openai-whisper's mel_filters.npz). */
    public static void loadFilters(java.io.InputStream in) throws java.io.IOException {
        java.io.BufferedReader r = new java.io.BufferedReader(new java.io.InputStreamReader(in));
        String[] dims = r.readLine().trim().split("\\s+");
        int rows = Integer.parseInt(dims[0]), cols = Integer.parseInt(dims[1]);
        float[][] f = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            String[] parts = r.readLine().trim().split("\\s+");
            for (int j = 0; j < cols; j++) f[i][j] = Float.parseFloat(parts[j]);
        }
        melFilters = f;
    }

}
